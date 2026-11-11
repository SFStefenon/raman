# Raman-Based Renal Disorder 

This repository describes an experimental pipeline for distinguishing urine Raman spectra from healthy individuals and patients with renal disorders using a small clinical cohort.

## Dataset
 
### Origin and collection

The spectral dataset is associated with **LUMINA (Light Based Urine Monitoring and Analysis)**, a collaboration involving the **Instituto Superior de Engenharia de Lisboa (ISEL)** and the **Unidade Local de Saúde da Arrábida (ULSA)**, Portugal. 

Urine specimens were obtained from healthy volunteers and individuals receiving nephrology care at **Hospital São Bernardo**, under an ethics-approved collection protocol and informed consent procedures. Samples were coded to protect participant identities, collected in sterile containers without preservatives, and stored at approximately **4 °C** before analysis. For spectroscopy, approximately **4 mL** of each sample was placed in a quartz cuvette and measured at room temperature (approximately **20 °C**).

The Raman system was assembled at ISEL and used a **532 nm excitation laser** (Thorlabs CPS532), optical components for rejecting Rayleigh scattering, and a digital imaging detector. Instrument calibration was checked using distilled water. The thesis describes storing Raman measurements in **CSV** format.

### Dataset composition

The cohort used in the current machine-learning study consists of **17 individuals**:

| Group | Patients | Binary label |
| --- | ---: | ---: |
| Healthy volunteers | 7 | `0` |
| Patients with renal disorders | 10 | `1` |
| **Total** | **17** | |

The positive group includes patients with **acute kidney injury (AKI)** and **chronic kidney disease (CKD)**, as well as other renal conditions such as **glomerulopathy** and **lupus nephritis**. Therefore, the classification target is **healthy versus renal disorder**, not an AKI-specific diagnosis.

> Each patient is represented in the manuscript's classification analysis by a Raman spectrum with **2,048 intensity features**, indexed by Raman shift (wavenumber, cm⁻¹). The complete input can be represented as a matrix of shape (`17, 2048`) and a binary target vector of length 17. Wavenumber coordinates should be retained alongside the intensity values to interpret selected spectral positions.

### Data interpretation and constraints

- Original patient spectra are the **only** observations counted in test-set performance metrics.
- Synthetic spectra are generated **within training folds** and are not additional patients.
- The dataset is small and heterogeneous. Results are **exploratory**, not evidence of clinical diagnostic validation.

## Proposed method

The proposed method combines **class-conditional latent flow augmentation**, **ANOVA-based spectral feature selection**, and **partial least squares discriminant analysis (PLS-DA)**. It is designed to address the combination of few labeled patients and many correlated Raman measurements.

### 1. Standard Normal Variate normalization

Standard Normal Variate (SNV) normalization is applied independently to each measured spectrum by subtracting its mean intensity and dividing the resulting values by the spectrum's standard deviation. A small numerical constant of `10^{-10}` is used as a lower bound for the denominator to prevent division by zero. This procedure reduces additive baseline offsets and multiplicative intensity variations while preserving the relative spectral structure, thereby improving the consistency of the spectral data for subsequent augmentation, feature selection, and classification.

### 2. Class-conditional latent flow augmentation

Because the original spectra contain 2,048 correlated features but the cohort contains only 17 people, generating realistic spectra directly in the original feature space is difficult. The proposed approach instead learns a compact latent representation and generates new training observations there.

1. Fit **principal component analysis (PCA)** on the real training spectra, using up to **4 components**. Standardize the resulting latent coordinates using training-derived statistics.
2. Train a **class-conditional flow-matching model** to transform Gaussian latent noise into latent coordinates resembling the selected class (`healthy` or `renal disorder`). Its velocity network is a multilayer perceptron with **two 64-unit hidden layers** and SiLU activations; the input includes the latent state, class label, and time encoding.
3. Optimize the flow model with **AdamW**, learning rate `1e-3`, weight decay `1e-4`, minibatch size `64`, and **350 gradient updates**.
4. Sample the learned flow using a **32-step Heun ordinary differential equation solver**. Inverse-transform the generated latent vectors through the PCA mapping to recover spectra in the original feature space, followed by SNV normalization.
5. Generate **30 synthetic spectra for each class**, adding **60 synthetic observations** to the real training data in each fold.

The flow is fit afresh in every cross-validation fold. No held-out spectrum contributes to PCA fitting, latent-space statistics, generative-model training, or synthetic-data construction.

### 3. ANOVA feature selection

A one-way analysis of variance (**ANOVA**) computes a class-discrimination score for each of the **2,048 spectral coordinates**. The score compares between-class differences with within-class variation. The **256 highest-scoring features** are retained, reducing the representation by **87.5%** while preserving original wavenumber locations.

ANOVA ranking is calculated on the combined **real and synthetic training spectra**. Selected indices are then applied unchanged to the held-out patient's original spectrum. Feature ranking is repeated independently for every fold.

### 4. PLS-DA classification

**Partial least squares discriminant analysis (PLS-DA)** is fit using the selected 256-dimensional training spectra. The manuscript specifies **4 latent components** and a **0.5 decision threshold**. PLS-DA is suited to spectral predictors with correlated variables and limited observations.

At inference, the held-out real spectrum receives the same per-spectrum SNV transformation, the current fold's 256 selected feature indices, and the trained classifier. **No synthetic spectrum is ever used as test data.**

## Evaluation protocol

The manuscript uses **patient-level leave-one-out cross-validation (LOOCV)**:

1. Hold out one original patient for testing.
2. Fit the PCA, latent flow, ANOVA selector, and PLS-DA exclusively with the other 16 patients and training-only synthetic spectra.
3. Produce exactly one out-of-fold prediction for the held-out real patient.
4. Repeat until all **17 patients** have been evaluated.

The study reports accuracy, balanced accuracy, sensitivity (renal-disorder group), specificity (healthy group), and area under the receiver operating characteristic curve (ROC-AUC). Its benchmarking experiment compares the full method with real-only classifiers, including SVM variants, elastic-net logistic regression, random forest, XGBoost, SE-ResNet1D, a Patch Transformer, and PLS-DA.

## Research limitations

The cohort contains only 17 patients and combines multiple renal conditions into one positive class. Synthetic observations cannot replace independent clinical recruitment. The thesis also describes instrumental sensitivity and calibration limitations associated with the experimental Raman setup. Accordingly, the repository documents an exploratory computational study rather than a validated diagnostic tool.

## Source documents

1. **Moreno, Ana Filipa Basílio (2024).** *Diagnóstico de lesão renal aguda através da análise optoelectrónica de urina*. Master's dissertation in Biomedical Engineering, Instituto Superior de Engenharia de Lisboa / Escola Superior de Tecnologia da Saúde de Lisboa.
2. **Mansour, Rima; Jesus, Rui; Seman, Laio O.; and Stefenon, Stefano F.** *Class-Conditional Latent Flow Augmentation and ANOVA-Guided PLS-DA for Small-Sample Spectral Classification of Acute Kidney Injury*. IEEE Access manuscript (working version supplied with this project).

**Availability:** Implementation commands, repository paths, raw-data download links, and a software license are not specified in the source documents and should be added only after the corresponding repository materials and permissions are confirmed.
