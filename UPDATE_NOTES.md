# Stable skin detail — 1.5

## User result: 1.4 was insufficient

The user's new updated/original comparison confirms that the previous candidate
did not deliver the intended visible detail. Both supplied clipboard PNGs are
1080 x 1920 and contain no workflow metadata. About 9.02% of RGB pixels differ.
In the same abdomen and thigh areas used below, luminance difference RMS is
only 0.7065 and 0.6962 8-bit levels respectively. Their mean changes are -0.4884
and -0.4800 levels. The forehead sample is 0.5287 levels RMS; a background sample
is unchanged. These are measurements of user-provided files, not model runs.
The absence of metadata means the exact model/widget settings cannot be recovered
from these PNGs. The result nevertheless fails the user's visual requirement.

## New approach

Version 1.5 supplies missing structure from a real skin photograph, independently
of the upscaler's small residual. The bundled reference is CC0; see
[asset provenance](assets/README.md). An optional `skin_reference` IMAGE input
can replace it with a crop appropriate to the target skin area.

The implementation builds a luminance-detail patch library, varies source patch
origins and orientations deterministically, and chooses patches by agreement
with the existing overlap. Smooth crossfades remove patch boundaries. Broad
reference illumination and colour are removed, and the result is masked with
the existing skin/face protection. Detail is attenuated where the source already
contains texture and near strong contours or exposure extremes.

This uses the general idea of exemplar texture transfer; it is not the full
minimum-cut quilting implementation described in
[Efros and Freeman image quilting](https://www.ipol.im/pub/art/2017/171/).
The output is synthesized detail borrowed from photographed skin, not verified
recovery of the subject's original pores. It is a 2D texture transfer, without
estimated 3D geometry or physically reconstructed directional lighting. The
bundled reference includes fine creases as well as pore-scale structure; a
custom reference allows a different texture character.

`skin_detail_amount` defaults to 0.65 and controls reference contrast. Zero
disables it for comparison with model-only processing. `texture_strength` still
controls the final blend; zero bypasses both stages. `texture_model=off` can now
run reference-only enhancement. The upscaler never loops. Experimental diffusion
is retained and does not use the new reference transfer.

The node logs skin-region RMS change before and after reference transfer during
the user's run. This helps distinguish a weak model residual, weak mask, and
missing reference contribution without relying solely on a full-image preview.

No application tests or model inference have been run for 1.5. The new output
still needs visual assessment; the user's 1.4 result must not be presented as
evidence that this version succeeds.

## Historical 1.4 investigation

The update targets the supplied comparison: image 1 is the six-pass result,
image 2 is the untreated source, and image 3 is the one-pass result. All three
are 1080 × 1920 RGB images. The six-pass result visibly develops a fine regular
mesh over the body. The one-pass change is much weaker.

## Reference measurements

These are measurements of the user's supplied outputs. The custom node, its
tests, and its models were **not executed** to produce them.

Luminance differences were computed with RGB weights (0.2126, 0.7152, 0.0722)
after decoding the PNGs into float RGB in 0–1. Differences are in the encoded
image domain, not linear-light luminance. Coordinates below are fractions of
image width/height: (left, top, right, bottom).

| Crop | Result | Difference RMS, 8-bit equivalent | Fine-frequency share |
| --- | --- | --- | --- |
| Abdomen (0.41, 0.62, 0.58, 0.68) | Six passes | 16.4015 | 92.32% |
| Abdomen (0.41, 0.62, 0.58, 0.68) | One pass | 1.3729 | 41.79% |
| Left thigh (0.35, 0.85, 0.43, 0.95) | Six passes | 13.1659 | 82.45% |
| Left thigh (0.35, 0.85, 0.43, 0.95) | One pass | 1.3883 | 28.85% |

For the frequency share, each difference crop had its mean removed and a
separable Hann window applied. Squared FFT magnitudes were summed above radial
frequency 0.35 cycles/pixel, divided by energy above 0.04 cycles/pixel. Radial
frequencies include diagonal components. This measures where the change occurs;
it does not directly measure realism or identify the model's internal cause.
The large high-frequency concentration supports removing recurrent enhancement
and rejecting near-pixel-scale changes instead of increasing their amplitude.

## Why this candidate

The previous texture filter was largely a high-pass of the ITF residual. It
emphasized the frequencies that expanded dramatically in the repeated output.
The new correction uses a band-pass: at default softness it subtracts a sigma-4
blur from a sigma-0.9 blur of the reconstruction difference. It therefore passes
intermediate structure while suppressing both broad tone changes and very fine
grain. It never normalizes low-energy model output up to a target grain level.
Only a small contribution from existing source structure is added separately.

The default model is now ClearRealityV1, not ITF. The
[author's model card](https://huggingface.co/Kim2091/ClearRealityV1) describes
training for photographic imagery and a natural, softer result. ITF's
[model description](https://openmodeldb.info/models/1x-ITF-SkinDiffDetail-Lite-v1)
describes an early model for 3D-character skin diffuse textures. This is a
domain-based reason to try ClearReality, not proof it outperforms ITF here.

ClearReality receives a modestly reduced source crop (default divisor 1.5) and
reconstructs it once. Its native 4x tiles are averaged down to 2x before assembly,
then antialiased to the original crop dimensions. Comparing that reconstruction
to the original source, rather than adding a residual to a resampled base, avoids
duplicating source texture. The final masked correction preserves the source
as its base. A shared RGB offset retains channel differences, while a smooth
contrast bound and source-derived contour/exposure weights limit harsh changes.

The legacy pass input is accepted for workflow compatibility but cannot trigger
another pass. The backend contains one texture call per image. The frontend
hides and resets the old widget. The new detail-size widget is appended after
that retained slot. Both examples use one pass and the current photo default.
The diffusion stage remains behind the existing experimental-mode branch.

## Verification status

The implementation and migration paths were reviewed by reading their source.
The image measurements and model-source metadata above informed the changes.
No test suite, syntax-check command, ComfyUI run, or model inference was executed.
Visual quality on these photographs is still unverified. Use the untreated
image as input for the next comparison, rather than feeding the old six-pass
output into the new node; this update is not a removal tool for baked-in grain.

Model file: `4x-ClearRealityV1.safetensors` from Kim2091's repository.
Published SHA-256:
`83a9c8e279e3d07548e2b0d736dc2c73c1736a07072a748ed4fafbba99b44693`.
The downloader validates this before installing the file.

### Follow-up source inspection

The remote safetensors header was read directly without loading weights or
running inference. It contains 204 tensors and an 18,768-byte JSON header.
The expected SPAN detection keys are present, including `conv_1.sk.weight`,
`block_1.c1_r.eval_conv.weight`, `block_1.c3_r.eval_conv.weight`,
`conv_cat.weight`, `conv_2.eval_conv.weight`, and `upsampler.0.weight`.
The first convolution has shape [48, 3, 1, 1]; the output convolution has shape
[48, 48, 3, 3], consistent with RGB at 4x pixel-shuffle scale. Stored tensors
are F16. This confirms file-format/architecture metadata, not a successful
runtime load or visual improvement.

The follow-up source review also confirmed that saved pass counts cannot cause
recurrence: the backend accepts but ignores that value, the frontend hides and
resets its retained slot, and only one `apply_texture` call is made per image.
Runtime image comparison remains pending the user's own result or authorization
to perform it under the earlier instruction not to run tests.
