# Childhood Trauma and Academic Achievement: Analysis Code

This repository contains statistical analysis and data preprocessing code for a study of childhood trauma, white matter structural connectivity, inhibitory control, and academic achievement. The study uses a cross-sectional design to examine trauma–achievement associations, age-related differences in white matter structural connectivity, candidate indirect pathways, and conditional associations.

## Statistical Analysis: `statistic_ana/`

### Main Analysis Scripts

| Script | Purpose |
| --- | --- |
| `0_env_bhv_partial_cor.py` | Tests partial correlations between childhood trauma and five academic measures, controlling for sex and family socioeconomic status, with Benjamini–Hochberg false discovery rate (BH-FDR) correction. |
| `1_bhv_group_diff.py` | Compares academic performance between matched trauma groups across four school stages using two-sided Mann–Whitney U tests, with FDR correction across the four stage-specific tests for each measure. |
| `2_env_brain_partial_cor.py` | Tests the partial correlation between childhood trauma and within-grade standardized fractional anisotropy (FA) in the Orbito-Affective network. |
| `3_gam_devp_diff_culmutive.ipynb` | Uses generalized additive models (GAMs) in R to examine Orbito-Affective network FA in relation to age in months in the two matched groups, compare curve shapes and intercepts, and estimate between-group smooth differences with pointwise 95% confidence intervals. |
| `plot_gam_fig.py` | Reads exported GAM curves, confidence intervals, and test results to plot age-related group curves and smooth differences. |
| `4_mediation.py` | Tests statistical mediation by within-grade standardized Orbito-Affective network FA in the associations of trauma with mathematics and Chinese reading composite scores, using 5,000 bootstrap resamples. |
| `5_age_moderate_mediation.py` | Models age in months as a moderator of the a, b, and direct c′ paths, estimating conditional indirect effects and contrasts between selected ages. |
| `6_moderation_new.py` | Tests whether inhibitory control, measured by stop-signal reaction time (SSRT), moderates trauma–achievement associations for mathematics and Chinese reading, and reports interaction terms, simple slopes, and Johnson–Neyman intervals. |
| `8_double_moderated_env-bhv.py` | Tests trauma × network FA × SSRT interactions across five networks and two academic composite scores, reporting multiple-comparison corrections, conditional effects, and corresponding plots. |

GAMs use FA after ComBat-GAM harmonization without within-grade Z standardization. Brain partial correlation and mediation analyses use within-grade FA Z scores for the Orbito-Affective network; three-way interaction analyses use within-grade FA Z scores for five networks.

## Data Preprocessing: `data_generate/`

| Script | Purpose |
| --- | --- |
| `sc2net.py` | Uses CAB-NP parcel-to-network labels to aggregate nonzero connections in individual structural connectivity matrices into within- and between-network mean FA, MD, RD, and AD measures. |
| `generate_combat_cov.py` | Prepares subject identifiers, age in months, sex, and site–scanner combinations as ComBat-GAM covariates, and flags configured sites for exclusion. |
| `encode_QC.py` | Combines diffusion imaging quality-control fields to generate subject inclusion codes for subsequent filtering. |
| `combat_gam.py` | Filters subjects based on brain-measure completeness, covariates, and imaging quality control, applies ComBat-GAM harmonization with `neuroHarmonize`, and saves harmonized measures and the fitted model. |
| `ses_pca.py` | Standardizes parental occupation and education, performs principal component analysis to derive the first component as a family socioeconomic status score, and reports loadings and explained variance. |
| `match_twogroup.ipynb` | Uses R to define groups by mean childhood trauma scores > 2 versus ≤ 2, performs 1:1 nearest-neighbor propensity score matching on age in months and sex (caliper = 0.25), and exports matched groups and covariate balance results. |

Imaging preprocessing proceeds from network-level aggregation to covariate and quality-control preparation, followed by ComBat-GAM harmonization. Socioeconomic status estimation and trauma-group matching require their respective input datasets.

## Running the Code

Python scripts use libraries including `pandas`, `numpy`, `scipy`, `statsmodels`, and `matplotlib`. Principal component analysis uses `scikit-learn`, and imaging harmonization uses `neuroHarmonize`. Both notebooks use an R kernel: matching requires `MatchIt` and `cobalt`, while GAM analysis uses packages including `mgcv`, `gratia`, `ggplot2`, `dplyr`, `tidyr`, `readr`, and `patchwork`.

All configured data, output, and cache paths use placeholders under `/path/to/data/` or `/path/to/output/`. Replace these paths with locations in your environment before running the scripts and prepare the required behavioral, environmental, covariate, and brain-measure datasets. Site exclusion identifiers use `SITE_TO_EXCLUDE_1`, `SITE_TO_EXCLUDE_2`, and `SITE_TO_EXCLUDE_3`; replace them with the applicable site–scanner labels before preprocessing. The scripts retain their existing variable and model settings. File numbering organizes the analysis stages rather than defining an automated pipeline. Run `plot_gam_fig.py` after exporting the GAM results.

The repository excludes participant data, matching outputs, notebook execution outputs, caches, and historical backups. Within-grade FA Z scores describe white matter structural connectivity relative to same-grade peers. Cross-sectional associations do not establish causal relationships or within-person developmental trajectories.
