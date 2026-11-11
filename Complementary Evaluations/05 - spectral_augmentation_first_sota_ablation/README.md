# Augmentation-first, joint classifier-feature ablation

## Protocol

1. **Augmentation:** fixed, prespecified Residual MLP + multi-resolution
   spectral pooling ranks real only, latent flow matching, latent DDPM, and an
   EDM-style score model, with one or more synthetic sample counts. Only inner
   original-patient validation is used for selection.
2. **Joint ablation:** selected augmentation is held fixed while Residual MLP,
   ConvNeXt1D, Patch Transformer, and Spectral Mixer are crossed with Raw SNV,
   multi-resolution patches, a masked spectral autoencoder, and a supervised
   contrastive embedding. TabPFN is optional, requiring independent weights.
3. **Outer evaluation:** each originally held-out patient is predicted once
   per arm, matching patients across comparisons. Changes to either the
   classifier or the representation are evaluated in isolation.

## Figures and interpretation

- `01_augmentation_inner_validation.pdf`: inner validation BA/SD and number
  of folds choosing each augmentation; **selection evidence only**.
- `02_augmentation_paired_heldout.pdf`: matched out-of-fold augmentation
  delta BA versus real-only using the identical fixed anchor; uncertainty.
- `03_joint_classifier_feature_heatmap.pdf`: conditional inner joint search.
- `04_component_ablation_paired_effects.pdf`: paired held-out degradation or
  improvement from replacing exactly one of the selected components.
- `05_patient_gain_loss_map.pdf`: which original patients were recovered/lost
  by the selected pipeline versus same model/features trained on real only.
- `06_outer_roc_precision_recall.pdf`: descriptive, uncalibrated cross-fold
  ROC/PR discrimination; curves are unreliable at this cohort size.
- `07_accuracy_computation_tradeoff.pdf`: resource vs held-out BA tradeoff.
- `08_synthetic_proximity_diversity.pdf`: train-only similarity/diversity check
  **not** proof of biochemical validity.
- `09_vs_modern_raw_real_only.pdf`: selected full pipeline versus contemporary
  classifier baselines using no augmentation and raw SNV only, with paired CIs.

CSV files contain per-patient outer predictions, stage-wise validation,
selection counts, uncertainty intervals, exact exploratory McNemar tests and
resource utilization. Five LaTeX tables are combined in `paper_tables.tex`.

### Limitations

The synthetic generator learns from 6–10 ORIGINAL examples per class, often
fewer in inner folds. It may memorize or distort biological signals. Every
augmented observation is generated only from original TRAIN patients. Do not
report synthetic samples as independent patients, pick final designs from
outer-test rankings, describe these compact networks as published pretrained
methods, or claim clinical significance without independent patient data.
For complete runs (all 17 held-out patients), the effective resolution of raw
accuracy is one patient. With only 17 total originals, inferential statistics
and effect-size intervals are extremely unstable.

## Install and run

`pip install numpy pandas scipy scikit-learn matplotlib torch`

Place `saudaveis` and `doentes` beside this file, set `RUN_MODE` at the top,
and run `python spectral_augmentation_first_sota_ablation.py`.
Set `RUN_MODE = 'complete'` for all 17 outer patients and the larger grid.
No ZIP archives or GPU are required. CPU training can be lengthy.
