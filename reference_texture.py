"""Deterministic skin-detail transfer from a photographed texture exemplar.

Patch selection varies the arrangement of REAL reference pixels; it does not
generate white noise or stamp analytic pores. Overlap matching and crossfades
avoid a repeating atlas. Only local luminance relief is transferred, never the
reference's colour, broad illumination, or body/face geometry.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch

import comfy.model_management
import comfy.utils

from .composite import blend_scaled, crop_image, crop_mask, gaussian_blur_image, paste_image
from .segment import mask_bboxes
from .texture import _detail_confidence, _resize_antialiased

PATCH = 48
OVERLAP = 16


@lru_cache(maxsize=1)
def _bundled_reference() -> torch.Tensor:
    path = Path(__file__).parent / "assets" / "skin_reference_cc0.jpg"
    if not path.is_file():
        raise RuntimeError(
            "[Skin Detailer] missing assets/skin_reference_cc0.jpg. Copy the entire "
            "node folder, including assets, or connect a skin_reference image."
        )
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        # Cache only the useful sampling resolution, not a ~110 MB float copy
        # of the original macro photograph. Keep the distributed asset intact.
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        rgb = np.array(image, dtype=np.float32) / 255.0
    return torch.from_numpy(rgb).unsqueeze(0)


@lru_cache(maxsize=4)
def _bundled_bank(detail_size: float) -> torch.Tensor:
    return _reference_bank(_bundled_reference(), detail_size)


def _reference_bank(reference: torch.Tensor, detail_size: float) -> torch.Tensor:
    """Photograph -> illumination-free, contrast-calibrated patch library on CPU."""
    reference = reference.detach().to(device="cpu", dtype=torch.float32)
    if reference.ndim != 4 or reference.shape[0] != 1 or reference.shape[-1] < 3:
        raise ValueError("[Skin Detailer] skin_reference must be one RGB skin close-up per image.")
    if min(reference.shape[1:3]) < PATCH:
        raise ValueError("[Skin Detailer] skin_reference must be at least 48 x 48 pixels.")
    if not bool(torch.isfinite(reference).all()):
        raise ValueError("[Skin Detailer] skin_reference contains non-finite pixels.")
    # The stock macro photo is 3693 pixels wide. Reducing it to ~512 brings
    # photographed microstructure to several output pixels, rather than using
    # macro-sized creases. User references should likewise be skin close-ups.
    size = max(1.0, min(2.0, float(detail_size)))
    h, w = reference.shape[1:3]
    # Preserve custom reference proportions; crop extreme panoramas before
    # resizing so neither dimension becomes an oversized allocation.
    if w > h * 8:
        left = (w - h * 8) // 2
        reference = reference[:, :, left:left + h * 8]
    elif h > w * 8:
        top = (h - w * 8) // 2
        reference = reference[:, top:top + w * 8]
    h, w = reference.shape[1:3]
    scale = min(512 * size / (1.5 * w), 1024 / max(h, w))
    scale = max(scale, PATCH / min(h, w))
    target_w, target_h = max(PATCH, round(w * scale)), max(PATCH, round(h * scale))
    reference = _resize_antialiased(reference[..., :3], target_h, target_w).clamp(0.0, 1.0)
    weights = reference.new_tensor([0.2126, 0.7152, 0.0722])
    luma = (reference * weights).sum(dim=-1, keepdim=True)
    relief = gaussian_blur_image(luma, 0.45) - gaussian_blur_image(luma, 6.0)
    rms = relief.square().mean().sqrt()
    if float(rms) < 0.001:
        raise ValueError(
            "[Skin Detailer] skin_reference has too little visible texture. "
            "Use a sharp crop of bare skin, or disconnect it to use the bundled photograph."
        )
    # Calibration applies to the explicitly chosen reference, never to a model
    # residual. This supplies detail even when the SR model is almost identity.
    relief = 3.0 * torch.tanh(relief / (3.0 * rms))
    plane = relief[0, ..., 0]
    patches = plane.unfold(0, PATCH, 16).unfold(1, PATCH, 16).reshape(-1, PATCH, PATCH)
    # Reject extreme patches (e.g. a strongly marked crease or a blank area).
    energy = patches.square().mean(dim=(-1, -2))
    median = energy.median().clamp_min(1e-6)
    usable = (energy > median * 0.3) & (energy < median * 2.5)
    if bool(usable.any()):
        patches = patches[usable]
    return patches.contiguous()


def _quilt(bank: torch.Tensor, height: int, width: int, seed: int) -> torch.Tensor:
    """Choose patches by overlap agreement, with deterministic nonrepeating origins."""
    stride = PATCH - OVERLAP
    canvas = torch.zeros((height, width), dtype=torch.float32)
    coverage = torch.zeros_like(canvas)
    generator = torch.Generator(device="cpu").manual_seed(int(seed) % (1 << 63))
    axis = torch.ones(PATCH)
    ramp = torch.sin(torch.linspace(0.0, torch.pi / 2, OVERLAP + 2)[1:-1]).square()
    axis[:OVERLAP] = ramp
    axis[-OVERLAP:] = ramp.flip(0)
    window = axis[:, None] * axis[None, :]
    steps = ((height + stride - 1) // stride) * ((width + stride - 1) // stride)
    progress = comfy.utils.ProgressBar(steps)
    for y in range(0, height, stride):
        comfy.model_management.throw_exception_if_processing_interrupted()
        for x in range(0, width, stride):
            ph, pw = min(PATCH, height - y), min(PATCH, width - x)
            ids = torch.randint(len(bank), (8,), generator=generator)
            candidates = bank[ids]
            # Different rotations/origins break atlas repetition. The variation
            # is in patch placement, not in the texture values themselves.
            rotation = int(torch.randint(4, (), generator=generator))
            candidates = torch.rot90(candidates, rotation, dims=(-2, -1))[:, :ph, :pw]
            target = canvas[y:y + ph, x:x + pw]
            covered = coverage[y:y + ph, x:x + pw]
            known = covered > 0.0
            if bool(known.any()):
                existing = target / covered.clamp_min(1e-8)
                error = ((candidates - existing).square() * known).sum(dim=(-1, -2))
                chosen = candidates[int(error.argmin())]
            else:
                chosen = candidates[0]
            weight = window[:ph, :pw]
            target.add_(chosen * weight)
            covered.add_(weight)
            progress.update(1)
    return (canvas / coverage.clamp_min(1e-8))[None, ..., None]


@torch.inference_mode()
def apply_reference_texture(
    source: torch.Tensor,
    candidate: torch.Tensor,
    mask: torch.Tensor,
    *,
    amount: float,
    strength: float,
    softness: float,
    detail_size: float,
    reference: torch.Tensor | None = None,
) -> torch.Tensor:
    """Supply missing skin structure, using the same effective mask as restoration.

    ``amount`` controls transferred reference contrast independently of the
    upscaler's residual. Each region receives one synthesis, not repeated model
    inference. Regions already containing fine detail receive less added detail.
    """
    amount = max(0.0, min(1.0, float(amount)))
    if amount == 0.0 or strength <= 0.0:
        return candidate
    bank = (_bundled_bank(float(detail_size)) if reference is None
            else _reference_bank(reference, detail_size))
    out = candidate.to(torch.float32)
    softness = max(0.0, min(1.0, float(softness)))
    for box in mask_bboxes(mask, threshold=1e-4, pad=32):
        comfy.model_management.throw_exception_if_processing_interrupted()
        base = crop_image(source, box).to(torch.float32)
        patch = crop_image(candidate, box).to(torch.float32)
        crop_m = crop_mask(mask, box)
        h, w = patch.shape[1:3]
        # Image content does not seed noise: stable coordinates choose the same
        # arrangement when the workflow is rerun. No global RNG state changes.
        seed = box[0] * 73856093 + box[1] * 19349663 + h * 83492791 + w
        relief = _quilt(bank, h, w, seed).to(device=patch.device)
        relief = gaussian_blur_image(relief, 0.35 + softness * 0.25)
        relief = relief - gaussian_blur_image(relief, 6.0)
        support = (crop_m.to(device=patch.device) > 0.0).unsqueeze(-1)
        energy = (relief.square() * support).sum() / support.sum().clamp_min(1)
        if float(energy) < 1e-6:
            continue
        relief = relief / energy.sqrt()

        luma = (base * base.new_tensor([0.2126, 0.7152, 0.0722])).sum(dim=-1, keepdim=True)
        existing = gaussian_blur_image(luma, 0.6) - gaussian_blur_image(luma, 2.0)
        existing_rms = gaussian_blur_image(existing.square(), 4.0).clamp_min(0.0).sqrt()
        needs_detail = 1.0 / (1.0 + (existing_rms / 0.012).square())
        response = _detail_confidence(luma) * needs_detail
        # Reference contrast is an explicit artistic control, independent of
        # how much a restoration network happens to change this particular skin.
        detail = relief * (0.026 * amount * (1.0 - 0.3 * softness))
        detail = 0.055 * torch.tanh(detail / 0.055) * response
        detail = torch.maximum(detail, -patch.amin(dim=-1, keepdim=True))
        detail = torch.minimum(detail, 1.0 - patch.amax(dim=-1, keepdim=True))
        transferred = (patch + detail).clamp(0.0, 1.0)
        blended = blend_scaled(patch, transferred, crop_m, strength)
        out = paste_image(out, blended, box)
    return out
