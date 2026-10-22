"""Single-pass photographic skin reconstruction at unchanged output resolution.

ClearReality reconstructs detail from a modestly reduced input. Its supersampled
output is antialiased back to the source grid; a bounded mid-frequency luminance
correction is then transferred inside the skin mask. No recursive enhancement,
random grain, procedural pores, or diffusion are used in this stage.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.utils

from .composite import align_box, blend_scaled, crop_image, gaussian_blur_image, paste_image
from .downloads import ModelDownloadError, ensure_model, find_in_folder
from .segment import mask_bboxes

log = logging.getLogger("SkinDetailer")

TEXTURE_FOLDER = "upscale_models"

#: Use the author's current safetensors release and published file checksum.
#: https://huggingface.co/Kim2091/ClearRealityV1 (Apache-2.0).
DEFAULT_TEXTURE_FILENAME = "4x-ClearRealityV1.safetensors"
AUTO_TEXTURE_LABEL = "auto (ClearReality photo detail)"
LEGACY_AUTO_TEXTURE_LABEL = "auto (1xSkinContrast-High-SuperUltraCompact)"
ITF_AUTO_TEXTURE_LABEL = "auto (1x-ITF-SkinDiffDetail-Lite-v1)"
ITF_TEXTURE_LABEL = "ITF SkinDiffDetail Lite v1 (legacy)"
TEXTURE_OFF_LABEL = "off"
TEXTURE_URLS = [
    "https://huggingface.co/Kim2091/ClearRealityV1/resolve/main/4x-ClearRealityV1.safetensors",
]
TEXTURE_SHA256 = "83a9c8e279e3d07548e2b0d736dc2c73c1736a07072a748ed4fafbba99b44693"
TEXTURE_MANUAL_HINT = (
    "ClearRealityV1 by Kim2091 (Apache-2.0, author's current release):\n"
    "https://huggingface.co/Kim2091/ClearRealityV1\n"
    "Save the file as 4x-ClearRealityV1.safetensors."
)
ITF_TEXTURE_FILENAME = "1x-ITF-SkinDiffDetail-Lite-v1.pth"
ITF_TEXTURE_URLS = [
    "https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o/1x-ITF-SkinDiffDetail-Lite-v1.pth",
]
ITF_TEXTURE_SHA256 = "94d368b633614958f84f335b129fd85abd30200e8fbc575b859ba6762116222b"

#: Context added around each skin region before running the texture model, so
#: tile seams and edge artefacts stay outside the blended area.
REGION_PAD = 96

_TEXTURE_CACHE: dict[str, Any] = {}


def texture_model_options() -> list[str]:
    """Combo options for the ``texture_model`` widget."""
    try:
        import folder_paths

        installed = list(folder_paths.get_filename_list(TEXTURE_FOLDER))
    except Exception:  # pragma: no cover - folder may not exist yet
        installed = []
    # ComfyUI validates combo values before calling the node. Keeping the old
    # auto label accepted also covers API workflows without a frontend migration.
    return [AUTO_TEXTURE_LABEL, ITF_TEXTURE_LABEL] + installed + [
        TEXTURE_OFF_LABEL, LEGACY_AUTO_TEXTURE_LABEL, ITF_AUTO_TEXTURE_LABEL,
    ]


def resolve_texture_path(selection: str) -> str | None:
    """Map a widget selection to a local file path (``None`` means disabled)."""
    if not selection or selection == TEXTURE_OFF_LABEL:
        return None
    if selection == ITF_TEXTURE_LABEL:
        return ensure_model(
            TEXTURE_FOLDER, ITF_TEXTURE_FILENAME, ITF_TEXTURE_URLS,
            "ITF SkinDiffDetail Lite v1",
            "https://openmodeldb.info/models/1x-ITF-SkinDiffDetail-Lite-v1 (CC-BY-NC-SA-4.0)",
            expected_sha256=ITF_TEXTURE_SHA256,
        )
    # Prefix match, not equality: workflows saved against an older build carry a
    # different "auto (...)" label and should keep working rather than erroring.
    if selection.startswith("auto"):
        return ensure_model(
            TEXTURE_FOLDER,
            DEFAULT_TEXTURE_FILENAME,
            TEXTURE_URLS,
            "ClearRealityV1 photographic detail",
            TEXTURE_MANUAL_HINT,
            expected_sha256=TEXTURE_SHA256,
        )

    path = find_in_folder(TEXTURE_FOLDER, selection)
    if path is None:
        raise ModelDownloadError(
            f"[Skin Detailer] texture model '{selection}' was not found in "
            f"models/{TEXTURE_FOLDER}/.\n"
            "Pick another entry, or select "
            f"'{AUTO_TEXTURE_LABEL}' to download the default automatically."
        )
    return path


def load_texture_model(path: str) -> Any:
    """Load (and cache) a spandrel image model by path."""
    cached = _TEXTURE_CACHE.get(path)
    if cached is not None:
        return cached

    try:
        from spandrel import ImageModelDescriptor, ModelLoader
    except ImportError as exc:  # pragma: no cover - spandrel ships with ComfyUI
        raise ModelDownloadError(
            "[Skin Detailer] spandrel is not installed, so the texture model "
            "cannot be loaded. Set texture_model to 'off', or repair the "
            "ComfyUI installation."
        ) from exc

    state_dict = comfy.utils.load_torch_file(path, safe_load=True)
    if "module.layers.0.residual_group.blocks.0.norm1.weight" in state_dict:
        state_dict = comfy.utils.state_dict_prefix_replace(state_dict, {"module.": ""})
    model = ModelLoader().load_from_state_dict(state_dict).eval()
    if not isinstance(model, ImageModelDescriptor):
        raise ModelDownloadError(
            f"[Skin Detailer] '{path}' is not a single-image model and cannot be "
            "used as a texture model."
        )
    if model.input_channels != 3 or model.output_channels != 3:
        raise ModelDownloadError("[Skin Detailer] texture models must accept and return RGB images.")
    model.model.requires_grad_(False)
    log.info("[Skin Detailer] texture model loaded (scale %sx): %s", model.scale, path)
    _TEXTURE_CACHE[path] = model
    return model


def purge_texture_cache() -> None:
    """Drop cached texture models."""
    _TEXTURE_CACHE.clear()
    comfy.model_management.soft_empty_cache()


@torch.inference_mode()
def _run_model(model: Any, image: torch.Tensor, output_scale: int = 1) -> torch.Tensor:
    """Tiled inference, reducing each SR tile before allocating the output canvas.

    A 4x model can return a 2x canvas without ever allocating a full 4x image.
    Averaging each aligned 2x2 block also rejects subpixel/shuffle artifacts.
    """
    device = comfy.model_management.get_torch_device()
    output_scale = max(1, min(int(model.scale), int(output_scale)))
    tile = 256 if model.scale > 1 else 512

    memory_required = comfy.model_management.module_size(model.model)
    memory_required += (tile * tile * 3) * image.element_size() * max(model.scale, 1.0) * 384.0
    memory_required += image.nelement() * image.element_size()
    comfy.model_management.free_memory(memory_required, device)

    output_device = comfy.model_management.intermediate_device()

    overlap = 64
    try:
        model.to(device)
        in_img = image.movedim(-1, -3).to(device)
        while True:
            try:
                steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(
                    in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap
                )
                pbar = comfy.utils.ProgressBar(steps)
                result = comfy.utils.tiled_scale(
                    in_img,
                    lambda a: _infer_tile(model, a, output_scale),
                    tile_x=tile,
                    tile_y=tile,
                    overlap=overlap,
                    upscale_amount=output_scale,
                    pbar=pbar,
                    output_device=output_device,
                )
                break
            except Exception as exc:
                comfy.model_management.raise_non_oom(exc)
                tile //= 2
                if tile < 128:
                    raise
    finally:
        model.to("cpu")

    return result.movedim(-3, -1).to(device=image.device, dtype=torch.float32)


def _infer_tile(model: Any, tile: torch.Tensor, output_scale: int) -> torch.Tensor:
    comfy.model_management.throw_exception_if_processing_interrupted()
    result = model(tile.float())
    size = (tile.shape[-2] * output_scale, tile.shape[-1] * output_scale)
    if result.shape[-2:] != size:
        result = F.interpolate(result, size=size, mode="area")
    return result


def _clean_model_input(image: torch.Tensor, amount: float) -> torch.Tensor:
    """Small edge-aware cleanup for inference only; never blur the output base."""
    amount = max(0.0, min(1.0, float(amount)))
    if amount == 0.0:
        return image
    height, width = image.shape[1:3]
    padded = F.pad(image.movedim(-1, 1), (1, 1, 1, 1), mode="replicate").movedim(1, -1)
    summed = torch.zeros_like(image)
    weights = torch.zeros_like(image[..., :1])
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            neighbor = padded[:, 1 + dy:1 + dy + height, 1 + dx:1 + dx + width]
            distance = (neighbor - image).square().mean(dim=-1, keepdim=True)
            weight = torch.exp(-distance / (2.0 * 0.025 ** 2) - (dx * dx + dy * dy) / 2.0)
            summed += neighbor * weight
            weights += weight
    return image.lerp(summed / weights.clamp_min(1e-8), amount)


def _resize_antialiased(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize RGB with a reconstruction filter instead of aliasing fine detail."""
    if image.shape[1:3] == (height, width):
        return image
    return F.interpolate(
        image.movedim(-1, 1), size=(height, width), mode="bicubic",
        align_corners=False, antialias=True,
    ).movedim(1, -1)


def _smoothstep(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(0.0, 1.0)
    return value.square() * (3.0 - 2.0 * value)


def _detail_confidence(luma: torch.Tensor) -> torch.Tensor:
    """Keep enhancement gentler at sharp contours and exposure extremes.

    This is based on source luminance/geometry, not a skin-colour classifier.
    It leaves most midtones at full strength across different skin tones.
    """
    shading = gaussian_blur_image(luma, 1.2)
    work = shading.movedim(-1, 1)
    padded = F.pad(work, (1, 1, 1, 1), mode="replicate")
    dx = 0.5 * (padded[:, :, 1:-1, 2:] - padded[:, :, 1:-1, :-2])
    dy = 0.5 * (padded[:, :, 2:, 1:-1] - padded[:, :, :-2, 1:-1])
    edge_weight = (1.0 / (1.0 + (dx.square() + dy.square()) / 0.04 ** 2)).movedim(1, -1)
    shadow_weight = _smoothstep((shading - 0.02) / 0.12)
    highlight_weight = _smoothstep((0.99 - shading) / 0.12)
    return edge_weight * shadow_weight * highlight_weight


def _transfer_detail(
    source: torch.Tensor, prediction: torch.Tensor, softness: float
) -> torch.Tensor:
    """Transfer photographic structure in a bounded band, rejecting pixel grain.

    Comparing the reconstruction to the ORIGINAL source avoids double-adding
    source detail after the internal resampling. No detail-energy normalization
    or random texture is used: a flat model residual remains flat.
    """
    prediction = torch.where(torch.isfinite(prediction), prediction, source)
    rgb_weights = source.new_tensor([0.2126, 0.7152, 0.0722])
    luma = (source * rgb_weights).sum(dim=-1, keepdim=True)
    residual = ((prediction - source) * rgb_weights).sum(dim=-1, keepdim=True)
    softness = max(0.0, min(1.0, float(softness)))

    # The previous high-pass selected exactly the frequencies that run away
    # under recurrence. Use a band-pass instead: reject ~1-pixel grain while
    # keeping several-pixel pore/fold structure and preserving broad shading.
    fine_sigma = 0.7 + 0.4 * softness
    detail = gaussian_blur_image(residual, fine_sigma) - gaussian_blur_image(residual, 4.0)

    # A small contribution from EXISTING structures improves local definition
    # without fabricating a noise floor on smooth skin. Avoid unsharp masking
    # the highest-frequency source band, which contains noise/compression.
    source_band = gaussian_blur_image(luma, 1.0) - gaussian_blur_image(luma, 3.0)
    detail = detail + source_band * (0.18 * (1.0 - 0.5 * softness))
    contrast = gaussian_blur_image(source_band.square(), 3.0).clamp_min(0.0).sqrt()
    limit = (0.025 + 0.75 * contrast).clamp(max=0.065) * (1.0 - 0.5 * softness)
    # Always enforce a real soft bound, including when softness=0. The previous
    # lerp with an unbounded residual still allowed arbitrarily harsh spikes.
    detail = limit * torch.tanh(detail / limit)
    detail = detail * _detail_confidence(luma)
    # A shared RGB offset preserves channel differences. Limit that offset to
    # available headroom instead of clipping channels independently.
    detail = torch.maximum(detail, -source.amin(dim=-1, keepdim=True))
    detail = torch.minimum(detail, 1.0 - source.amax(dim=-1, keepdim=True))
    return (source + detail).clamp(0.0, 1.0)


@torch.inference_mode()
def apply_texture(
    image: torch.Tensor,
    mask: torch.Tensor,
    model: Any,
    strength: float,
    softness: float = 0.5,
    cleanup: float = 0.15,
    detail_size: float = 1.5,
) -> torch.Tensor:
    """Reconstruct skin once, blended through ``mask * strength``.

    Only the bounding regions of the mask are pushed through the model, so a
    small patch of skin on a large canvas stays cheap.

    Args:
        image: ``[1, H, W, C]`` float image in 0..1.
        mask: ``[1, H, W]`` effective skin mask in 0..1.
        model: a loaded spandrel ``ImageModelDescriptor``.
        strength: 0..1 blend strength for the texture layer.
        softness: 0..1 suppression of harsh corrections; source remains the compositing base.
        cleanup: 0..1 edge-aware denoising of the model input only.
        detail_size: 1..2 reconstruction scale for SR models; 1x models stay native.

    Returns:
        ``[1, H, W, C]`` float32 image; pixels where ``mask == 0`` are unchanged.
    """
    if strength <= 0.0:
        return image.to(torch.float32)

    height, width = image.shape[1], image.shape[2]
    boxes = mask_bboxes(mask, threshold=1e-4, pad=REGION_PAD)
    if not boxes:
        return image.to(torch.float32)

    total_area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in boxes)
    if total_area >= 0.85 * height * width:
        boxes = [(0, 0, width, height)]

    out = image.to(torch.float32)
    for box in boxes:
        comfy.model_management.throw_exception_if_processing_interrupted()
        box = align_box(box, width, height, pad=0, stride=1)
        patch = crop_image(image, box).to(torch.float32)
        model_input = _clean_model_input(patch, cleanup)
        patch_h, patch_w = patch.shape[1:3]
        if model.scale > 1:
            # Give the photo SR model room to reconstruct features that survive
            # at the final size. This is one feed-forward reconstruction, not
            # feeding generated texture repeatedly back into the model.
            detail_size = max(1.0, min(2.0, float(detail_size)))
            work_h = max(1, round(patch_h / detail_size))
            work_w = max(1, round(patch_w / detail_size))
            model_input = _resize_antialiased(model_input, work_h, work_w).clamp(0.0, 1.0)
            prediction = _run_model(model, model_input, output_scale=2)
        else:
            prediction = _run_model(model, model_input)
        prediction = _resize_antialiased(prediction, patch_h, patch_w)
        textured = _transfer_detail(patch, prediction, softness)
        out = paste_image(out, textured, box)

    # The mask, not the crop rectangle, decides what actually changes.
    return blend_scaled(image, out, mask, strength)
