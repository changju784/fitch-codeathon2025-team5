# FitchGroup Codeathon 2025 – Team 5 Solution

This repository contains our end-to-end solution for the FitchGroup Codeathon 2025, where we develop machine learning models to estimate corporate Scope 1 and Scope 2 greenhouse gas emissions from partial financial and ESG disclosures.  
The full technical report, code snippets, and detailed analysis are documented in [`final.md`](final.md); this README provides a concise, section-by-section overview and links to the corresponding parts of the report.

Live demo video link - https://www.youtube.com/watch?v=g0D5z0ApbIA

---

## Notebook Section Overview

### [1. Problem Overview](final.md#1-problem-overview)

This section frames the business problem of estimating Scope 1 (direct) and Scope 2 (indirect, purchased energy) emissions for companies with incomplete disclosures. It clarifies key definitions, explains why these targets are critical for portfolio risk, valuation, and regulatory reporting, and motivates the need for robust, out-of-sample prediction. The discussion also highlights why Scope 1 and Scope 2 should be modelled separately, given their different operational and geographic drivers.

### [2. Target Distribution and ML Objectives](final.md#2-target-distribution-and-ml-objectives)

Here we analyze the empirical distributions of the Scope 1 and Scope 2 targets, showing that both are highly right-skewed with heavy tails. The section motivates the use of log-transformed targets to stabilize variance and outlines the primary evaluation metrics (e.g., MAE and log-MAE) that focus on median-oriented, robust error. It also sets the high-level modelling objective: accurate, stable prediction under realistic data sparsity and disclosure constraints.

### [3. Feature Engineering — Overview](final.md#3-feature-engineering--overview)

This section introduces the main feature families used in the models: scale (revenue), geography, and behavioural/ESG signals. For each family, it explains the economic and climate rationale, the specific variables derived (e.g., log revenue, sector-partitioned revenues, adjusted ESG scores), and the intended role in capturing emissions-relevant heterogeneity. It serves as a roadmap for the more detailed, hypothesis-driven analyses that follow in Sections 4–6.

### [4. Scale of Business](final.md#4-scale-of-business)

This section tests whether firm size, proxied by total revenue and sector-partitioned revenues, is a first-order driver of emissions. It documents how log transformations and scope-specific revenue buckets improve distributional properties and strengthen correlations with log emissions, and how “materiality” flags based on non-zero scope-specific revenues help distinguish high- from low-exposure firms. The analysis concludes that scale features (log-revenue plus scope-specific revenues and flags) form a robust core signal, but are not sufficient on their own to fully explain emissions, motivating the addition of geography and behavioural features.

### [5. Geography](final.md#5-geography)

This section evaluates how location-based variables—primarily country, with regions as a coarser alternative—help explain residual variation in emissions after controlling for scale and sector. It applies quantile-based grouping and statistical comparisons of mean emissions across countries and regions, focusing on whether observed differences align with expectations from grid mix and fuel-use patterns, especially for Scope 2. The results support using country codes as carefully encoded categorical features (e.g., frequency grouping and regularized target encoding), while discarding sparse and weakly informative region indicators from the final feature set.

### [6. Behavioral Features](final.md#6-behavioral-features)

This section analyzes ESG-derived behavioural indicators, including environmental, social, governance, and adjusted scores, as well as SDG activity flags, as proxies for mitigation practices and process maturity. It reports global and size-conditional correlations between these scores and both raw and log emissions, and tests whether SDG-related activities are associated with systematically lower emissions within revenue quantiles. The evidence justifies including adjusted environmental scores and selected behavioural flags as complementary features alongside scale and geography, while excluding redundant composite scores (such as the overall ESG score) to avoid multicollinearity.

### [7. Experiments](final.md#7-experiments)

This section describes the experimental setup used to validate the proposed feature engineering and preprocessing strategies. It compares baseline and enriched feature sets, and evaluates different model families (e.g., linear regression, median regression, CatBoost) using cross-validated metrics on both raw and log targets. The results show that combining the proposed features with median-focused objectives and tree-based methods yields substantial improvements over the baseline.

### [8. CatBoost Hyperparameter Optimization](final.md#8-catboost-hyperparameter-optimization)

Here we conduct a lightweight random search over key CatBoost hyperparameters (iterations, depth, learning rate, and regularization) on top of the proposed feature set and log-target formulation. The section explains the search space, cross-validation strategy, and the combined optimization objective across Scope 1 and Scope 2. It concludes that even modest tuning further improves performance, while the base configuration is already strong enough for competition use.

### [9. Final Prediction (`submission.csv`)](final.md#9-final-prediction-submissioncsv)

This section provides the final training and inference pipeline used to generate the `submission.csv` file for the Codeathon leaderboard. It shows how the proposed preprocessor and CatBoost models are instantiated, how train and test data are loaded, and how predictions are written out in the required format. Practical notes are also provided on common pandas warnings and how to interpret them when running the pipeline.

### [10. Conclusion](final.md#10-conclusion)

The conclusion synthesizes the main findings: domain-informed feature engineering, log-target transformations, and robust tree-based models materially improve emissions prediction quality. It quantifies the gains in MAE for both Scope 1 and Scope 2 relative to the baseline and discusses the implications for portfolio-level carbon estimation and ESG risk management. The section also outlines how the approach can be extended with richer sectoral data, updated grid factors, or more advanced model families in future work.

---

## Using This Repository

The code referenced in `final.ipynb` and `final.md` assumes a standard structure with a `data/` directory (containing `train.csv` and `test.csv`) and a `src/` package with modules such as `preprocessor.py`, `models.py`, and `trainer.py`.  
To reproduce the experiments and final submission:

1. Use Python 3.10+ (or similar) and install all required packages via:
   ```bash
   pip install -r requirements.txt
   ```
2. Prepare the input data under `data/` with the same schema as used in the Codeathon.
3. Follow the code snippets in [Section 7](final.md#7-experiments), [Section 8](final.md#8-catboost-hyperparameter-optimization), and [Section 9](final.md#9-final-prediction-submissioncsv) to run experiments and generate `submission.csv`.

For a full narrative, implementation details, and discussion of limitations, please refer to [`final.ipynb`](final.ipynb) and [`final.md`](final.md).

