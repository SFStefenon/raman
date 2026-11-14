# Figure interpretation guide

1. **Augmentation probe gain** — Inner-CV balanced accuracy difference from no augmentation, using the SAME fixed Patch Transformer on original-patient validation folds. Not test performance.
2. **Joint ablation heatmap** — Once augmentation is chosen, mean inner-CV balanced accuracy for each neural classifier and spectral representation. Selection count is not accuracy.
3. **Outer held-out balanced accuracy** — Top two methods have IDENTICAL classifier and features: real only versus best synthetic (even if real only was selected by the probe). The third arm is the preferred real/synthetic option. All remaining modern methods are real-only.
4. **Paired patient error transitions** — Shows how many original patients change correctness as a result of augmentation (recovered vs newly introduced errors).
5. **Paired effect sizes** — Forced best-synthetic versus paired alternatives (including runner-up and per-generator controls). Exploratory 95% bootstrap intervals, paired class-balanced sign-flip tests (exact up to 20 discordances, otherwise Monte Carlo), McNemar tests, Holm adjustment. These tests have limited power and fold fits overlap; they cannot prove superiority with few original patients.
6. **ROC and precision–recall** — Secondary descriptive views of held-out discrimination; separate models across folds are not probability calibrated.
7. **Spectral plausibility** — Real versus best-synthetic class-conditional median and 10–90% range on ONE outer training fold; never held-out data. Always plotted for the best-synthetic candidate; it may lose to the real-only option.
8. **Accuracy vs compute** — Training-only inner-CV score versus standalone cold-start time, generator fit included.
9. **Matched generator comparison** — Each generator retrains the SAME selected fold-specific classifier and representation; outer labels are never used for any choice. This is DESCRIPTIVE: do not select the best generator using these test scores.
10. **Seed robustness** — Retrain models and generators from independent seeds for the inner-selected configurations, without re-running selection. Assess reliability of predictions, not selection across seeds.
11. **Generator quality** — Train-only class-conditional MMD, diversity, nearest-patient distance, and near-copy rate. No test samples used. Good synthetic fidelity does not imply better clinical performance.
12. **Selection stability** — Frequency of each classifier/feature pair selected by inner CV across outer patient folds; inspect top-runner margins in model_selection_stability.csv.

Extra CSVs: outer_generator_comparison, generator_quality_by_train_fold, generator_quality_summary, refit_seed_metrics,
refit_seed_patient_agreement, outer_threshold_sensitivity_descriptive. Threshold
sweeps are post-hoc checks, NOT thresholds selected for deployment.

Critical controls: no held-out original patient participates in generation, classifier fit, representation construction, augmentation selection, or model/feature selection. Synthetic examples never count as independent held-out patients. Quick mode is a smoke test. With few original patients all inferential claims require independent external validation.
The conditional DDPM and conditional GAN/WGAN-GP operate in TRAIN-ONLY PCA latent space;
they are compact conditional baselines, not official paper implementation checkpoints.
Do not use outer-test generator rankings or threshold sweeps to revise hyperparameters
and then report the same outer scores as an unbiased final test.
