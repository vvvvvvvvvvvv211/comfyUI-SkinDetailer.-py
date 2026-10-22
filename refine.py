"""Masked, low-denoise diffusion refinement of the skin regions.

The refinement pass is model-family agnostic on purpose: it only ever performs
img2img with the MODEL / CLIP / VAE the workflow supplies, so FLUX, SDXL,
Qwen-Image, Krea 2 Turbo and Krea 2 Raw all work with no special-casing.

Two core mechanisms do the heavy lifting:

* **Differential Diffusion** is patched onto a *clone* of the incoming model, so
  the grayscale mask becomes a per-pixel denoise strength. That is what turns
  ``face_identity_protect`` into "face skin gets gently refined, body skin gets
  the full effect" rather than a binary include/exclude.
* Sampling goes through :func:`nodes.common_ksampler`, which builds the
  ``latent_preview`` callback internally. The live latent preview of the skin
  being detailed therefore streams onto *this* node for free.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

import comfy.model_management
import comfy.utils

from .composite import align_box, crop_image, crop_mask, merge_boxes, paste_image
from .segment import mask_bboxes

log = logging.getLogger("SkinDetailer")

#: Crops longer than this are cut down so a single region cannot blow up VRAM.
REFINE_MAX_SIDE = 1536
#: Only genuinely small crops are enlarged, and never past this target...
REFINE_MIN_SIDE = 1024
#: ...and only when they start below this size, by at most ``MAX_UPSCALE``.
#: Enlarging an already-detailed crop and shrinking it back afterwards is a
#: resample round-trip that turns invented pore noise into visible blotches, so
#: crops that are already large are refined at their native resolution.
REFINE_UPSCALE_BELOW = 768
MAX_UPSCALE = 1.5

#: Context kept around each skin region so the model sees surrounding structure.
CONTEXT_PAD = 48

#: "auto" switches to regional crops below this skin coverage...
AUTO_COVERAGE_LIMIT = 0.35
#: ...or above this canvas size, in pixels.
AUTO_MEGAPIXEL_LIMIT = 2.2e6

#: More regions than this and a single full-canvas pass is cheaper.
MAX_REGIONS = 8


def apply_differential_diffusion(model: Any) -> Any:
    """Return a clone of ``model`` with the Differential Diffusion patch applied."""
    from comfy_extras.nodes_differential_diffusion import DifferentialDiffusion

    return DifferentialDiffusion.execute(model, strength=1.0)[0]


def build_conditioning(
    clip: Any, skin_prompt: str, positive: list | None = None
) -> tuple[list, list]:
    """Build the (positive, negative) conditioning pair for the refine pass.

    ``negative`` is a zeroed-out copy of ``positive``, which is what core nodes
    do for families where CFG 1.0 is the norm and what every other family
    tolerates as an empty prompt.
    """
    from nodes import CLIPTextEncode, ConditioningZeroOut

    if positive is None:
        if clip is None:
            raise ValueError(
                "[Skin Detailer] no CLIP connected and no 'positive' conditioning "
                "supplied, so the skin prompt cannot be encoded. Connect the CLIP "
                "output of your loader, or wire a CONDITIONING into 'positive'."
            )
        positive = CLIPTextEncode().encode(clip, skin_prompt)[0]

    negative = ConditioningZeroOut().zero_out(positive)[0]
    return positive, negative


def _vae_stride(vae: Any) -> int:
    """Spatial compression factor of the VAE (never assume /8)."""
    try:
        stride = vae.spacial_compression_encode()
    except Exception:  # pragma: no cover - defensive
        stride = 8
    try:
        stride = int(stride)
    except (TypeError, ValueError):
        stride = 8
    return max(1, stride)


def _refine_tile(
    image: torch.Tensor,
    mask: torch.Tensor,
    patched_model: Any,
    vae: Any,
    positive: list,
    negative: list,
    seed: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    denoise: float,
) -> torch.Tensor:
    """Run one masked img2img pass over a whole tensor, returning pixels.

    The mask is handed to the sampler in pixel space; ``comfy.utils.reshape_mask``
    interpolates it onto whatever latent grid the model actually uses, so no
    stride assumption is baked in here.
    """
    from nodes import common_ksampler

    latent = {"samples": vae.encode(image[:, :, :, :3])}
    latent["noise_mask"] = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))

    result = common_ksampler(
        patched_model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent,
        denoise=denoise,
    )[0]

    decoded = vae.decode(result["samples"]).to(torch.float32)
    if decoded.ndim == 5:  # video VAEs collapse to a single frame here
        decoded = decoded.reshape(-1, *decoded.shape[-3:])
    if decoded.shape[1] != image.shape[1] or decoded.shape[2] != image.shape[2]:
        decoded = comfy.utils.common_upscale(
            decoded.movedim(-1, 1), image.shape[2], image.shape[1], "bicubic", crop="disabled"
        ).movedim(1, -1)
    return decoded[:, :, :, : image.shape[3]]


def _resample(image: torch.Tensor, width: int, height: int, method: str = "bicubic") -> torch.Tensor:
    return comfy.utils.common_upscale(
        image.movedim(-1, 1), width, height, method, crop="disabled"
    ).movedim(1, -1)


def _resample_mask(mask: torch.Tensor, width: int, height: int) -> torch.Tensor:
    return comfy.utils.common_upscale(
        mask.unsqueeze(1), width, height, "bilinear", crop="disabled"
    ).squeeze(1)


def _restore_residual(
    source: torch.Tensor, work: torch.Tensor, refined: torch.Tensor
) -> torch.Tensor:
    """Resize only the diffusion change, keeping native source detail intact."""
    refined = refined.to(device=source.device, dtype=torch.float32)
    work = work.to(device=source.device, dtype=torch.float32)
    residual = torch.where(torch.isfinite(refined), refined - work, torch.zeros_like(work))
    height, width = source.shape[1:3]
    if residual.shape[1:3] != source.shape[1:3]:
        residual = _resample(residual, width, height)
    return (source + residual).clamp(0.0, 1.0)


def _refine_target_size(width: int, height: int, stride: int) -> tuple[int, int]:
    """Pick the working resolution for a crop, snapped to the VAE stride."""
    longest = max(width, height)
    if longest > REFINE_MAX_SIDE:
        scale = REFINE_MAX_SIDE / longest
    elif longest < REFINE_UPSCALE_BELOW:
        scale = min(REFINE_MIN_SIDE / longest, MAX_UPSCALE)
    else:
        scale = 1.0

    target_w = max(stride, int(round(width * scale / stride)) * stride)
    target_h = max(stride, int(round(height * scale / stride)) * stride)
    return target_w, target_h


def choose_mode(
    mask: torch.Tensor, processing_mode: str
) -> tuple[str, list[tuple[int, int, int, int]]]:
    """Resolve ``processing_mode`` into a concrete mode plus its region boxes."""
    height, width = mask.shape[-2], mask.shape[-1]
    boxes = mask_bboxes(mask, threshold=1e-4, pad=CONTEXT_PAD)
    if not boxes:
        return "full_canvas", []

    if processing_mode == "full_canvas":
        return "full_canvas", boxes
    if processing_mode == "regional_crops":
        if len(boxes) > MAX_REGIONS:
            log.info(
                "[Skin Detailer] %d skin regions exceeds the %d-region budget; "
                "using a single full-canvas pass.",
                len(boxes),
                MAX_REGIONS,
            )
            return "full_canvas", boxes
        return "regional_crops", boxes

    coverage = sum((b[2] - b[0]) * (b[3] - b[1]) for b in boxes) / float(width * height)
    pixel_count = float(width * height)
    if len(boxes) <= MAX_REGIONS and (
        coverage < AUTO_COVERAGE_LIMIT or pixel_count > AUTO_MEGAPIXEL_LIMIT
    ):
        return "regional_crops", boxes
    return "full_canvas", boxes


def refine_image(
    image: torch.Tensor,
    mask: torch.Tensor,
    model: Any,
    clip: Any,
    vae: Any,
    *,
    skin_prompt: str,
    positive: list | None,
    seed: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    denoise: float,
    processing_mode: str = "auto",
) -> torch.Tensor:
    """Produce the refined *candidate* image for one input image.

    The return value is a full-canvas candidate, not the final output: the
    caller still composites it through the mask, which is what guarantees that
    non-skin pixels stay bit-identical to the input.

    Args:
        image: ``[1, H, W, C]`` float image in 0..1.
        mask: ``[1, H, W]`` effective skin mask (also the per-pixel denoise map).
        model: the workflow's MODEL; it is cloned, never mutated.
        clip: the workflow's CLIP, used when ``positive`` is not supplied.
        vae: the workflow's VAE.

    Returns:
        ``[1, H, W, C]`` float32 candidate image.
    """
    if mask.max().item() <= 1e-4:
        return image.to(torch.float32)

    patched = apply_differential_diffusion(model)
    positive, negative = build_conditioning(clip, skin_prompt, positive)
    stride = _vae_stride(vae)
    height, width = image.shape[1], image.shape[2]

    mode, boxes = choose_mode(mask, processing_mode)
    log.info(
        "[Skin Detailer] refine mode=%s regions=%d canvas=%dx%d stride=%d",
        mode,
        len(boxes) if mode == "regional_crops" else 1,
        width,
        height,
        stride,
    )

    if mode == "full_canvas":
        # Apply the same memory cap as regional mode, without enlarging small
        # full canvases. Restore only the change so downscaling cannot erase
        # the high-resolution texture layer that was just added.
        scale = min(1.0, REFINE_MAX_SIDE / max(width, height))
        work_w = max(stride, int(width * scale / stride) * stride)
        work_h = max(stride, int(height * scale / stride) * stride)
        work_image = image if (work_w, work_h) == (width, height) else _resample(image, work_w, work_h)
        work_mask = mask if (work_w, work_h) == (width, height) else _resample_mask(mask, work_w, work_h)
        refined = _refine_tile(
            work_image, work_mask, patched, vae, positive, negative,
            seed, steps, cfg, sampler_name, scheduler, denoise,
        )
        return _restore_residual(image, work_image, refined)

    candidate = image.to(torch.float32)
    boxes = merge_boxes([
        align_box(box, width, height, pad=0, stride=stride, min_size=stride * 8)
        for box in boxes
    ])
    for index, box in enumerate(boxes):
        comfy.model_management.throw_exception_if_processing_interrupted()
        crop = crop_image(image, box)
        crop_m = crop_mask(mask, box)

        crop_h, crop_w = crop.shape[1], crop.shape[2]
        target_w, target_h = _refine_target_size(crop_w, crop_h, stride)
        work_crop = crop if (target_w, target_h) == (crop_w, crop_h) else _resample(crop, target_w, target_h)
        work_mask = (
            crop_m if (target_w, target_h) == (crop_w, crop_h)
            else _resample_mask(crop_m, target_w, target_h)
        )

        refined_crop = _refine_tile(
            work_crop, work_mask, patched, vae, positive, negative,
            (seed + index) % (1 << 64), steps, cfg, sampler_name, scheduler, denoise,
        )
        refined_crop = _restore_residual(crop, work_crop, refined_crop)
        candidate = paste_image(candidate, refined_crop, box)

    return candidate
