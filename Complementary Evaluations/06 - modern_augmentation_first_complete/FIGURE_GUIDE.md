# Figure interpretation guide

1. **Augmentation probe gain** — Inner-CV balanced accuracy difference from no augmentation, using the SAME fixed Patch Transformer on original-patient validation folds. Not test performance.
2. **Joint ablation heatmap** — Once augmentation is chosen, mean inner-CV balanced accuracy for each neural classifier and spectral representation. Selection count is not accuracy.
3. **Outer held-out balanced accuracy** — Top two methods have IDENTICAL classifier and features: real only versus best synthetic (even if real only was selected by the probe). The third arm is the preferred real/synthetic option. All remaining modern methods are real-only.
4. **Paired patient error transitions** — Shows how many original patients change correctness as a result of augmentation (recovered vs newly introduced errors).
5. **Paired effect sizes** — Forced best-synthetic versus paired real-only alternatives (and the preferred CV choice). Point and exploratory 95% bootstrap interval. CSV also includes exploratory exact class-balanced paired sign-flip and exact McNemar tests, Holm corrected. These tests are low-power and fold fits overlap; not proof of superiority with 17 patients.
6. **ROC and precision–recall** — Secondary descriptive views of held-out discrimination; separate models across folds are not probability calibrated.
7. **Spectral plausibility** — Real versus best-synthetic class-conditional median and 10–90% range on ONE outer training fold; never held-out data. Always plotted for the best-synthetic candidate; it may lose to the real-only option.
8. **Accuracy vs compute** — Training-only inner-CV score versus standalone cold-start time, generator fit included.

Critical controls: no held-out original patient participates in generation, classifier fit, representation construction, augmentation selection, or model/feature selection. Synthetic examples never count as independent held-out patients. Quick mode is a smoke test. With 17 patients all inferential claims require independent external validation.
