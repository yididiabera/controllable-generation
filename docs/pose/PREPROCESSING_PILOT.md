# One-shot pose preprocessing pilot

Status: diagnostic data-path pilot completed on the server; training eligibility is not established.

## Scope and representation

Branch: `feat/pose-preprocessing-pilot`, separate from the pose-record pipeline PR.
The annotated-shot run used implementation commit `4bc2f45`.

The pilot renders grouped Gaussian keypoint maps (head, torso, limbs) for all
person candidates. It retains every available in-frame position without a raw-score
threshold, confidence conversion, temporal tracking, interpolation, or smoothing.
These maps are not `pose-pilot-v1` records. They do not satisfy its selected-subject
and validity contract. Their `pose_encoded` key is a loader diagnostic interface;
the three channels are raw maps, not the legacy 256-channel learned pose features.
No trained pose adapter or end-to-end model forward pass was tested.

Source coordinates use an explicit pixel-centre resize transform. Spatial size is
128 × 128, matching the loader's direct resize for this check. That resize changes
aspect ratio for the 1280 × 720 source; it is not a final training resize policy.
Gaussian sigma is 2 output pixels. The three map channels use a maximum across
all points in each group, so individual joints and people cannot be recovered
from the maps alone. Raw predictions remain the audit source.

## Server verification reported on 2026-11-14

- Source: `/mnt/d1/snet/controllable-generation/veo-data/02.mp4`.
- Video ID `02`, shot ID `02_shot_0000`.
- Original metadata: `/mnt/d1/snet/jedidiah/data_depth_repro_docx/shots_metadata.json`.
- Frozen split: `/mnt/d1/snet/jedidiah/data_depth_repro/split_manifest.json`.
- Partition `train`; full annotations contain 524 shots, of which 314 are training shots.
- Selected source indices: `[0, 27, 54, 81, 109, 136, 163, 191]`.
- Candidate counts: `[8, 4, 8, 7, 2, 7, 5, 7]`.
- Three renderer/sampler tests passed.
- The real `ControllableVideoDataset` constructor discovered exactly one sample.
- `dataset[0]` loaded the correct video ID, shot ID and caption.
- Video and control shapes were each `[3, 8, 128, 128]`.
- Saved selected indices matched the shared frame sampler.

Server output directory:
`/home/jovyan/jedidiah/controllable-generation/artifacts/pose-preprocessing-pilot/02-annotated-shot-v1/`.

Artifacts include `manifest.json`, `raw_predictions.json`, and
`controls/02/shot_02_shot_0000_controls_encoded.npz`. The manifest stores actual
source, annotation, split and raw-prediction hashes, extractor model information,
code revision and rendering policy. Hash values were not supplied in the terminal
report, so this document does not claim independent verification of those values.

## Visual review

The user supplied `annotated-shot-overlay.png`, generated from the fully loaded
sample. Points broadly align with people across all eight inspected frames.
The main woman's head and upper-body estimates generally follow her position.
Overlapping people and cropped lower-body estimates remain uncertain.
This is a qualitative alignment check at 128 × 128, not a labelled landmark
accuracy benchmark or proof of temporal stability between sampled frames.

The source image is preserved locally under
`artifacts/pose-preprocessing-pilot/02-annotated-shot-v1/review/` with an evidence
manifest. Overlay SHA-256: `818799525e031171b3eba5a92cb01175bc773fd139661343c6f42988116bad6b`.

## Remaining work

Before using these outputs for training, establish the intended people policy,
visibility/score handling, spatial and temporal sampling, and the exact pose
encoder/adapter interface. The current eight-frame loader diagnostic is not a
WAN generation frame-count or model-training contract. Keep this evidence when
judging any later training results.

The preprocessing PR has not yet been opened. This pilot can be reviewed as a
diagnostic preprocessing contribution with the above limits stated explicitly.
