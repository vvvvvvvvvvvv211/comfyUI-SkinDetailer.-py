"""Skin segmentation built on ComfyUI's native SAM 3.1 support.

ComfyUI 0.22.3 ships SAM 3 natively (``comfy_extras/nodes_sam3.py`` +
``comfy/text_encoders/sam3_clip.py``), so this module drives that machinery
directly and adds no third-party dependency.

Two details of the native implementation shape the design here:

* ``SAM3TokenizerWrapper`` splits a comma-separated prompt into one
  conditioning entry per concept (``sam3_multi_cond``) and ``SAM3_Detect``
  loops over those entries inside a single call, unioning the masks. One node
  call therefore covers a whole concept list -- exactly the batching the
  pipeline wants.
* Every concept defaults to ``max_detections = 1``. A bare ``"skin"`` prompt
  would keep only the single highest scoring instance, so every concept below
  carries an explicit ``":N"`` instance cap.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import torch

import comfy.model_management
import comfy.sd
import folder_paths

from .composite import gaussian_kernel1d, merge_boxes
from .downloads import ModelDownloadError, ensure_model, find_in_folder

log = logging.getLogger("SkinDetailer")

SAM3_FILENAME = "sam3.1_multiplex_fp16.safetensors"
SAM3_FOLDER = "checkpoints"
SAM3_URLS = [
    "https://huggingface.co/Comfy-Org/sam3.1/resolve/main/split_files/checkpoints/sam3.1_multiplex_fp16.safetensors",
    "https://huggingface.co/Comfy-Org/sam3.1/resolve/main/sam3.1_multiplex_fp16.safetensors",
    "https://huggingface.co/Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors",
]
SAM3_MANUAL_HINT = (
    "Source: https://huggingface.co/Comfy-Org/sam3.1 (file: sam3.1_multiplex_fp16.safetensors)"
)
SAM3_AUTO_LABEL = "auto (sam3.1_multiplex_fp16)"


def sam3_model_options() -> list[str]:
    """Combo options for the ``sam3_model`` widget: everything in models/checkpoints."""
    try:
        entries = list(folder_paths.get_filename_list(SAM3_FOLDER))
    except Exception:  # pragma: no cover - folder may be missing on a fresh install
        entries = []
    # Surface anything that looks like a SAM checkpoint first.
    likely = [e for e in entries if "sam3" in e.lower() or "sam_3" in e.lower()]
    rest = [e for e in entries if e not in likely]
    return [SAM3_AUTO_LABEL] + likely + rest

# Concept lists. Each concept gets an explicit instance cap because the SAM3
# tokenizer defaults to a single detection per concept.
SKIN_CONCEPTS = "skin:16"
NON_SKIN_CONCEPTS = (
    "eyes:12, eyebrows:8, lips:8, teeth:8, facial hair:8, hair:8, "
    "fingernails:24, clothing:12, glasses:4, jewelry:12"
)
FACE_CONCEPTS = "face:8"

#: Extra threshold applied to the subtract pass so over-eager negative masks
#: cannot eat true skin.
NEGATIVE_THRESHOLD_OFFSET = 0.05

#: Small-component cutoff, capped at 64 pixels so high-resolution images do
#: not lose small but real patches of exposed skin.
MIN_BLOB_FRACTION = 0.0002  # 0.02 %

# Module level singleton: SAM 3.1 is expensive to load, and the node is meant to
# be re-run repeatedly on the same session.
_SAM3_CACHE: dict[str, tuple[Any, Any]] = {}


# --------------------------------------------------------------------------- #
# mask utilities
# --------------------------------------------------------------------------- #

def _to_bhw(mask: torch.Tensor) -> torch.Tensor:
    """Normalise any mask tensor to ``[B, H, W]`` float32 on CPU."""
    mask = mask.detach().to(device="cpu", dtype=torch.float32)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    elif mask.ndim == 4:
        mask = mask.reshape(-1, mask.shape[-2], mask.shape[-1])
    elif mask.ndim != 3:
        mask = mask.reshape(-1, mask.shape[-2], mask.shape[-1])
    return mask


def _binary_kernel() -> np.ndarray:
    """The tapered 3x3 structuring element core ``GrowMask`` uses."""
    return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)


def morph_mask(mask: torch.Tensor, expand: int) -> torch.Tensor:
    """Dilate (``expand > 0``) or erode (``expand < 0``) a mask by N pixels.

    Mirrors ``comfy_extras.nodes_mask.GrowMask`` with tapered corners.
    """
    if expand == 0:
        return mask
    try:
        import scipy.ndimage
    except ImportError:  # pragma: no cover - scipy ships with ComfyUI
        log.warning("[Skin Detailer] scipy unavailable; skipping mask grow/shrink.")
        return mask

    kernel = _binary_kernel()
    out = []
    for m in _to_bhw(mask):
        arr = m.numpy()
        for _ in range(abs(int(expand))):
            if expand < 0:
                arr = scipy.ndimage.grey_erosion(arr, footprint=kernel)
            else:
                arr = scipy.ndimage.grey_dilation(arr, footprint=kernel)
        out.append(torch.from_numpy(np.ascontiguousarray(arr)))
    return torch.stack(out, dim=0)


def fill_pinholes(mask: torch.Tensor) -> torch.Tensor:
    """Fill only small enclosed holes; keep background gaps between limbs open."""
    try:
        import scipy.ndimage
    except ImportError:  # pragma: no cover
        return mask
    out = []
    for m in _to_bhw(mask):
        binary = m.numpy() > 0.5
        holes = scipy.ndimage.binary_fill_holes(binary) & ~binary
        labels, _ = scipy.ndimage.label(holes)
        areas = np.bincount(labels.ravel())
        small = areas <= 64
        small[0] = False
        out.append(torch.from_numpy((binary | small[labels]).astype(np.float32)))
    return torch.stack(out, dim=0)


def remove_small_blobs(mask: torch.Tensor, min_fraction: float = MIN_BLOB_FRACTION) -> torch.Tensor:
    """Drop small connected components using a canvas-relative, capped cutoff."""
    try:
        import scipy.ndimage
    except ImportError:  # pragma: no cover
        return mask
    mask = _to_bhw(mask)
    _, height, width = mask.shape
    min_area = min(64.0, max(4.0, min_fraction * height * width))
    out = []
    for m in mask:
        binary = m.numpy() > 0.5
        labels, count = scipy.ndimage.label(binary)
        if count == 0:
            out.append(m)
            continue
        areas = np.bincount(labels.ravel())
        keep = np.zeros(areas.shape[0], dtype=bool)
        keep[1:] = areas[1:] >= min_area
        cleaned = keep[labels]
        out.append(torch.from_numpy(cleaned.astype(np.float32)))
    return torch.stack(out, dim=0)


def feather_mask(mask: torch.Tensor, feather_px: float) -> torch.Tensor:
    """Separable gaussian blur of a mask, expressed in pixels of feathering."""
    if feather_px <= 0.0:
        return mask
    mask = _to_bhw(mask)
    sigma = max(0.3, float(feather_px) / 2.0)
    kernel = gaussian_kernel1d(sigma, mask.dtype)
    pad = (kernel.numel() - 1) // 2
    work = mask.unsqueeze(1)  # [B, 1, H, W]
    work = torch.nn.functional.pad(work, (pad, pad, 0, 0), mode="replicate")
    work = torch.nn.functional.conv2d(work, kernel.view(1, 1, 1, -1))
    work = torch.nn.functional.pad(work, (0, 0, pad, pad), mode="replicate")
    work = torch.nn.functional.conv2d(work, kernel.view(1, 1, -1, 1))
    return work.squeeze(1).clamp_(0.0, 1.0)


def feather_skin_inward(mask: torch.Tensor, feather_px: float) -> torch.Tensor:
    """Smoothstep from the boundary into skin, with zero support outside it."""
    mask = _to_bhw(mask)
    if feather_px <= 0.0:
        return mask
    try:
        import scipy.ndimage
    except ImportError:  # Keep the exclusion contract even without SciPy.
        return feather_mask(mask, feather_px) * mask
    out = []
    for m in mask:
        binary = m.numpy() > 0.5
        if binary.all() or not binary.any():
            out.append(m)
            continue
        distance = scipy.ndimage.distance_transform_edt(binary)
        t = np.clip(distance / (float(feather_px) + 1.0), 0.0, 1.0)
        out.append(torch.from_numpy((t * t * (3.0 - 2.0 * t)).astype(np.float32)))
    return torch.stack(out)


def mask_bboxes(
    mask: torch.Tensor, threshold: float = 0.01, pad: int = 0
) -> list[tuple[int, int, int, int]]:
    """Bounding boxes ``(x1, y1, x2, y2)`` of the connected regions of one mask.

    Boxes that overlap after padding are merged, so nearby limbs end up in a
    single refinement crop instead of several overlapping ones.
    """
    mask = _to_bhw(mask)[0]
    height, width = mask.shape
    try:
        import scipy.ndimage

        labels, count = scipy.ndimage.label(mask.numpy() > threshold)
        if count == 0:
            return []
        slices = scipy.ndimage.find_objects(labels)
    except ImportError:  # pragma: no cover
        nonzero = (mask > threshold).nonzero()
        if nonzero.numel() == 0:
            return []
        ys, xs = nonzero[:, 0], nonzero[:, 1]
        return [(max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
                 min(width, int(xs.max()) + 1 + pad), min(height, int(ys.max()) + 1 + pad))]

    boxes: list[list[int]] = []
    for sl in slices:
        if sl is None:
            continue
        y_sl, x_sl = sl
        boxes.append(
            [
                max(0, x_sl.start - pad),
                max(0, y_sl.start - pad),
                min(width, x_sl.stop + pad),
                min(height, y_sl.stop + pad),
            ]
        )

    return merge_boxes([tuple(int(v) for v in box) for box in boxes])


# --------------------------------------------------------------------------- #
# SAM 3.1 plumbing
# --------------------------------------------------------------------------- #

def resolve_sam3_path(selection: str = SAM3_AUTO_LABEL) -> str:
    """Resolve the ``sam3_model`` widget selection to a file in models/checkpoints.

    An explicit selection is never downloaded -- if the user picked a file it is
    expected to exist. Only the ``auto`` entry falls back to fetching the
    official weights.
    """
    # Prefix match so workflows saved against an older "auto (...)" label still work.
    if selection and not selection.startswith("auto"):
        path = find_in_folder(SAM3_FOLDER, selection)
        if path is None:
            raise ModelDownloadError(
                f"[Skin Detailer] the selected SAM 3 checkpoint '{selection}' is no "
                f"longer in models/{SAM3_FOLDER}/.\n"
                f"Pick another entry, or select '{SAM3_AUTO_LABEL}' to download the "
                "official weights automatically."
            )
        return path

    existing = find_in_folder(SAM3_FOLDER, SAM3_FILENAME)
    if existing is not None:
        return existing
    return ensure_model(
        SAM3_FOLDER, SAM3_FILENAME, SAM3_URLS, "SAM 3.1 (multiplex fp16)", SAM3_MANUAL_HINT
    )


def load_sam3(selection: str = SAM3_AUTO_LABEL) -> tuple[Any, Any]:
    """Load (and cache) the SAM 3.1 detector plus its CLIP text encoder."""
    path = resolve_sam3_path(selection)
    cached = _SAM3_CACHE.get(path)
    if cached is not None:
        log.info("[Skin Detailer] SAM 3.1 already resident: %s", os.path.basename(path))
        return cached

    log.info(
        "[Skin Detailer] loading SAM 3.1 from models/%s/%s (first run takes a while, "
        "the node stays cached afterwards)",
        SAM3_FOLDER,
        os.path.relpath(path, os.path.dirname(os.path.dirname(path))),
    )
    out = comfy.sd.load_checkpoint_guess_config(
        path,
        output_vae=False,
        output_clip=True,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
    )
    model, clip = out[0], out[1]
    if model is None or clip is None:
        raise ModelDownloadError(
            f"[Skin Detailer] '{path}' did not load as a SAM 3.1 checkpoint "
            "(missing detector or text encoder).\n"
            "Delete the file and re-run to re-download, or replace it with the "
            f"official one.\n{SAM3_MANUAL_HINT}"
        )
    _SAM3_CACHE[path] = (model, clip)
    return model, clip


def unload_sam3() -> None:
    """Push SAM 3.1 off the compute device without dropping the CPU cache.

    Called before the diffusion pass so the sampler gets the VRAM back.
    """
    if not _SAM3_CACHE:
        return
    mm = comfy.model_management
    try:
        ours = set()
        for model, clip in _SAM3_CACHE.values():
            ours.add(id(model))
            patcher = getattr(clip, "patcher", None)
            if patcher is not None:
                ours.add(id(patcher))
        keep = [lm for lm in list(mm.current_loaded_models) if id(getattr(lm, "model", None)) not in ours]
        mm.free_memory(1e30, mm.get_torch_device(), keep_loaded=keep)
    except Exception as exc:  # pragma: no cover - defensive, internals may drift
        log.debug("[Skin Detailer] targeted SAM3 offload failed (%s); soft cache flush.", exc)
    mm.soft_empty_cache()


def purge_sam3_cache() -> None:
    """Drop the cached SAM 3.1 weights entirely (frees system RAM too)."""
    _SAM3_CACHE.clear()
    comfy.model_management.soft_empty_cache()


def _encode_concepts(clip: Any, concepts: str) -> list:
    """Encode a comma-separated concept list into SAM3 multi-concept conditioning."""
    tokens = clip.tokenize(concepts)
    return clip.encode_from_tokens_scheduled(tokens)


def _detect(model: Any, conditioning: list, image: torch.Tensor, threshold: float) -> torch.Tensor:
    """Run one ``SAM3_Detect`` pass, returning a unioned ``[B, H, W]`` mask."""
    from comfy_extras.nodes_sam3 import SAM3_Detect

    result = SAM3_Detect.execute(
        model=model,
        image=image,
        conditioning=conditioning,
        threshold=float(max(0.0, min(1.0, threshold))),
        refine_iterations=2,
        individual_masks=False,
    )
    return _to_bhw(result[0])


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #

class SkinSegmentation:
    """Result of one segmentation run over an image batch."""

    __slots__ = ("skin", "face", "raw_skin", "non_skin")

    def __init__(
        self,
        skin: torch.Tensor,
        face: torch.Tensor,
        raw_skin: torch.Tensor,
        non_skin: torch.Tensor,
    ) -> None:
        self.skin = skin           # cleaned, grown, feathered [B, H, W]
        self.face = face           # feathered face mask       [B, H, W]
        self.raw_skin = raw_skin   # binary pass A             [B, H, W]
        self.non_skin = non_skin   # binary pass B             [B, H, W]

    def is_empty(self, index: int) -> bool:
        return bool(self.skin[index].max().item() <= 1e-4)


def segment_skin(
    image: torch.Tensor,
    threshold: float = 0.45,
    mask_expand: int = 0,
    mask_feather: float = 12.0,
    sam3_model: str = SAM3_AUTO_LABEL,
) -> SkinSegmentation:
    """Find exposed skin (and only skin) across an image batch.

    Exactly three SAM 3.1 passes are issued for the whole batch:

    ``A`` positive skin, ``B`` one batched query for every non-skin feature to
    subtract, and ``C`` the face region used for identity protection.

    Args:
        image: ``[B, H, W, C]`` float image in 0..1, ComfyUI IMAGE layout.
        threshold: detection threshold for the positive and face passes.
        mask_expand: pixels to dilate the positive mask by before subtraction.
        mask_feather: gaussian feather radius in pixels for the final mask.
        sam3_model: ``sam3_model`` widget selection, or the auto label.

    Returns:
        A :class:`SkinSegmentation` with all masks on CPU as ``[B, H, W]``.
    """
    model, clip = load_sam3(sam3_model)

    pos_cond = _encode_concepts(clip, SKIN_CONCEPTS)
    neg_cond = _encode_concepts(clip, NON_SKIN_CONCEPTS)
    face_cond = _encode_concepts(clip, FACE_CONCEPTS)

    comfy.model_management.throw_exception_if_processing_interrupted()
    skin_raw = _detect(model, pos_cond, image, threshold)
    comfy.model_management.throw_exception_if_processing_interrupted()
    non_skin = _detect(model, neg_cond, image, threshold + NEGATIVE_THRESHOLD_OFFSET)
    comfy.model_management.throw_exception_if_processing_interrupted()
    face_raw = _detect(model, face_cond, image, threshold)

    skin_bin = (skin_raw > 0.5).float()
    non_skin_bin = (non_skin > 0.5).float()
    face_bin = (face_raw > 0.5).float()

    # Grow before subtracting: expanding afterwards would push the skin mask
    # back over the eyes / lips / hair we just excluded.
    grown = morph_mask(fill_pinholes(skin_bin), int(mask_expand))
    subtracted = (grown * (1.0 - non_skin_bin)).clamp_(0.0, 1.0)
    cleaned = remove_small_blobs(subtracted)

    # Feather inward: Gaussian tails previously re-enabled background and thin
    # excluded features, including eyebrows/nails that had no eroded core.
    skin_final = feather_skin_inward(cleaned, float(mask_feather))

    # Keep all detected face pixels fully protected at strength=1; soften only
    # the transition beyond the detected face boundary.
    face_soft = torch.maximum(face_bin, feather_mask(face_bin, float(mask_feather)))

    return SkinSegmentation(skin_final, face_soft, skin_bin, non_skin_bin)


def apply_face_protection(
    skin: torch.Tensor, face: torch.Tensor, strength: float
) -> torch.Tensor:
    """Attenuate the skin mask inside the face region.

    Because refinement runs under Differential Diffusion the mask value *is* the
    per-pixel denoise strength, so scaling face skin by ``1 - strength`` makes
    the diffusion pass proportionally gentler there (``1.0`` leaves face skin
    untouched by diffusion) while body skin keeps the full effect.
    """
    if strength <= 0.0:
        return skin
    factor = 1.0 - float(min(1.0, strength)) * face.clamp(0.0, 1.0)
    return (skin * factor).clamp_(0.0, 1.0)
