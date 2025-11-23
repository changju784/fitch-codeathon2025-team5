# Experiment Report

Generated on: 2025-11-22 19:31:48

## Summary Metrics (Scope 1)

| Experiment              | Model                 | Preprocessor                   | Status   | RMSE       | MAE       | Log RMSE   | Log MAE   |
|:------------------------|:----------------------|:-------------------------------|:---------|:-----------|:----------|:-----------|:----------|
| baseline_config         | LinearRegressionModel | CodeathonBaselinePreprocessor  | Success  | 106867.53  | 64409.11  | 3.51       | 2.58      |
| xgb_abs_revenue         | XGBoostModel          | AbsoluteSectorRevenuePrep.     | Success  | 108067.20  | 49652.96  | 2.02       | 1.59      |
| **catboost_abs_rev_new**| **CatBoostModel**     | **AbsoluteRevenuePreprocessor**| **Success**| **108608.60**| **48092.99**| **1.91**   | **1.50**  |
| catboost_tree           | CatBoostModel         | TreePreprocessor               | Success  | 109481.97  | 48556.96  | 1.90       | 1.49      |
| catboost_abs_revenue    | CatBoostModel         | AbsoluteSectorRevenuePrep.     | Success  | 109714.91  | 49311.80  | 1.97       | 1.53      |
| rf_tree                 | RandomForestModel     | TreePreprocessor               | Success  | 109983.44  | 48709.65  | 2.00       | 1.57      |
| xgb_tree                | XGBoostModel          | TreePreprocessor               | Success  | 111933.49  | 52328.54  | 2.10       | 1.66      |
| linear_top_features     | LinearRegressionModel | TopFeaturesPreprocessor        | Success  | 112325.65  | 51272.13  | 2.02       | 1.59      |
| ridge_top_features      | RidgeModel            | TopFeaturesPreprocessor        | Success  | 114592.94  | 51429.62  | 2.01       | 1.58      |
| lasso_comprehensive     | LassoModel            | ComprehensivePreprocessor      | Success  | 118353.68  | 52955.13  | 2.35       | 1.94      |
| lasso_top_features      | LassoModel            | TopFeaturesPreprocessor        | Success  | 118353.68  | 52955.13  | 2.35       | 1.94      |
| ridge_comprehensive     | RidgeModel            | ComprehensivePreprocessor      | Success  | 4653074.38 | 552526.17 | 1.97       | 1.55      |
| lasso_abs_revenue       | LassoModel            | AbsoluteSectorRevenuePrep.     | Failed   |            |           |            |           |
| linear_abs_revenue      | LinearRegressionModel | AbsoluteSectorRevenuePrep.     | Failed   |            |           |            |           |
| linear_comprehensive    | LinearRegressionModel | ComprehensivePreprocessor      | Failed   |            |           |            |           |
| ridge_abs_revenue       | RidgeModel            | AbsoluteSectorRevenuePrep.     | Failed   |            |           |            |           |
| baseline_abs_revenue    | LinearRegressionModel | AbsoluteRevenuePreprocessor    | Failed   |            |           |            |           |

## Detailed Experiment Info

### baseline_config
- **Status**: Success
- **Model**: LinearRegressionModel
- **Preprocessor**: CodeathonBaselinePreprocessor
- **RMSE**: 106867.53
- **MAE**: 64409.11
- **Features (13)**: revenue, overall_score, environmental_score, social_score, governance_score, sect_C_pct, sect_G_pct, sect_J_pct, env_score_adjustment, sdg_id_3, sdg_id_9, region_code_NAM, region_code_WEU

---
### xgb_abs_revenue
- **Status**: Success
- **Model**: XGBoostModel
- **Preprocessor**: AbsoluteSectorRevenuePreprocessor
- **RMSE**: 108067.20
- **MAE**: 49652.96
- **Features (18)**: environmental_score, social_score, governance_score, region_group, env_score_adjustment, sdg_3, sdg_7, sdg_9, sdg_11, sdg_12, sdg_OTHER, sect_C_revenue, sect_G_revenue, sect_J_revenue, sect_M_revenue, sect_N_revenue, sect_OTHER_revenue, revenue_log

---
### catboost_tree
- **Status**: Success
- **Model**: CatBoostModel
- **Preprocessor**: TreePreprocessor
- **RMSE**: 109481.97
- **MAE**: 48556.96
- **Features (18)**: environmental_score, social_score, governance_score, region_group, sect_C_pct, sect_G_pct, sect_J_pct, sect_M_pct, sect_N_pct, sect_OTHER_pct, env_score_adjustment, sdg_3, sdg_7, sdg_9, sdg_11, sdg_12, sdg_OTHER, revenue_log

---
### catboost_abs_revenue
- **Status**: Success
- **Model**: CatBoostModel
- **Preprocessor**: AbsoluteSectorRevenuePreprocessor
- **RMSE**: 109714.91
- **MAE**: 49311.80
- **Features (18)**: environmental_score, social_score, governance_score, region_group, env_score_adjustment, sdg_3, sdg_7, sdg_9, sdg_11, sdg_12, sdg_OTHER, sect_C_revenue, sect_G_revenue, sect_J_revenue, sect_M_revenue, sect_N_revenue, sect_OTHER_revenue, revenue_log

---
### catboost_abs_rev_new
- **Status**: Success
- **Model**: CatBoostModel
- **Preprocessor**: AbsoluteRevenuePreprocessor
- **Scope 1 Metrics**:
  - RMSE: 108608.60 (±15959.43)
  - MAE: 48092.99 (±6085.57)
  - Log RMSE: 1.91 (±0.20)
  - Log MAE: 1.50 (±0.14)
- **Scope 2 Metrics**:
  - RMSE: 155689.07 (±87837.99)
  - MAE: 51238.62 (±19658.80)
  - Log RMSE: 2.42 (±0.16)
  - Log MAE: 1.75 (±0.06)
- **Features (18)**: environmental_score, social_score, governance_score, region_group, sect_C_abs, sect_G_abs, sect_J_abs, sect_M_abs, sect_N_abs, sect_OTHER_abs, env_score_adjustment, sdg_3, sdg_7, sdg_9, sdg_11, sdg_12, sdg_OTHER, revenue_log
- **Note**: Uses absolute sector revenue amounts (sect_*_abs) instead of percentages (sect_*_pct)

---
### rf_tree
- **Status**: Success
- **Model**: RandomForestModel
- **Preprocessor**: TreePreprocessor
- **RMSE**: 109983.44
- **MAE**: 48709.65
- **Features (18)**: environmental_score, social_score, governance_score, region_group, sect_C_pct, sect_G_pct, sect_J_pct, sect_M_pct, sect_N_pct, sect_OTHER_pct, env_score_adjustment, sdg_3, sdg_7, sdg_9, sdg_11, sdg_12, sdg_OTHER, revenue_log

---
### xgb_tree
- **Status**: Success
- **Model**: XGBoostModel
- **Preprocessor**: TreePreprocessor
- **RMSE**: 111933.49
- **MAE**: 52328.54
- **Features (18)**: environmental_score, social_score, governance_score, region_group, sect_C_pct, sect_G_pct, sect_J_pct, sect_M_pct, sect_N_pct, sect_OTHER_pct, env_score_adjustment, sdg_3, sdg_7, sdg_9, sdg_11, sdg_12, sdg_OTHER, revenue_log

---
### linear_top_features
- **Status**: Success
- **Model**: LinearRegressionModel
- **Preprocessor**: TopFeaturesPreprocessor
- **RMSE**: 112325.65
- **MAE**: 51272.13
- **Features (20)**: environmental_score, social_score, governance_score, sect_C_pct, sect_G_pct, sect_J_pct, sect_M_pct, sect_N_pct, sect_OTHER_pct, env_score_adjustment, sdg_3, sdg_7, sdg_9, sdg_11, sdg_12, sdg_OTHER, revenue_log, region_group_NAM, region_group_OTHER, region_group_WEU

---
### ridge_top_features
- **Status**: Success
- **Model**: RidgeModel
- **Preprocessor**: TopFeaturesPreprocessor
- **RMSE**: 114592.94
- **MAE**: 51429.62
- **Features (20)**: environmental_score, social_score, governance_score, sect_C_pct, sect_G_pct, sect_J_pct, sect_M_pct, sect_N_pct, sect_OTHER_pct, env_score_adjustment, sdg_3, sdg_7, sdg_9, sdg_11, sdg_12, sdg_OTHER, revenue_log, region_group_NAM, region_group_OTHER, region_group_WEU

---
### lasso_comprehensive
- **Status**: Success
- **Model**: LassoModel
- **Preprocessor**: ComprehensivePreprocessor
- **RMSE**: 118353.68
- **MAE**: 52955.13
- **Features (80)**: environmental_score, social_score, governance_score, sect_A_pct, sect_B_pct, sect_C_pct, sect_D_pct, sect_E_pct, sect_F_pct, sect_G_pct, sect_H_pct, sect_I_pct, sect_J_pct, sect_K_pct, sect_L_pct, sect_M_pct, sect_N_pct, sect_O_pct, sect_P_pct, sect_Q_pct, sect_R_pct, sect_S_pct, sect_T_pct, env_score_adjustment, env_act_Disposal , env_act_End-use, env_act_Farming, env_act_Manufacturing, env_act_Operation, env_act_Other, env_act_Raw materials, env_act_Transportation, sdg_id_2, sdg_id_3, sdg_id_4, sdg_id_5, sdg_id_6, sdg_id_7, sdg_id_8, sdg_id_9, sdg_id_11, sdg_id_12, sdg_id_13, sdg_id_16, revenue_log, region_code_ANZ, region_code_CAR, region_code_EA, region_code_EEU, region_code_LATAM, region_code_NAM, region_code_WEU, country_code_AT, country_code_BE, country_code_CA, country_code_CH, country_code_CN, country_code_CZ, country_code_DE, country_code_DK, country_code_EE, country_code_ES, country_code_FI, country_code_FR, country_code_GB, country_code_GI, country_code_IE, country_code_IT, country_code_JE, country_code_JP, country_code_LU, country_code_MT, country_code_MX, country_code_NL, country_code_NO, country_code_NZ, country_code_PR, country_code_PT, country_code_SE, country_code_US

---
### lasso_top_features
- **Status**: Success
- **Model**: LassoModel
- **Preprocessor**: TopFeaturesPreprocessor
- **RMSE**: 118353.68
- **MAE**: 52955.13
- **Features (20)**: environmental_score, social_score, governance_score, sect_C_pct, sect_G_pct, sect_J_pct, sect_M_pct, sect_N_pct, sect_OTHER_pct, env_score_adjustment, sdg_3, sdg_7, sdg_9, sdg_11, sdg_12, sdg_OTHER, revenue_log, region_group_NAM, region_group_OTHER, region_group_WEU

---
### ridge_comprehensive
- **Status**: Success
- **Model**: RidgeModel
- **Preprocessor**: ComprehensivePreprocessor
- **RMSE**: 4653074.38
- **MAE**: 552526.17
- **Features (80)**: environmental_score, social_score, governance_score, sect_A_pct, sect_B_pct, sect_C_pct, sect_D_pct, sect_E_pct, sect_F_pct, sect_G_pct, sect_H_pct, sect_I_pct, sect_J_pct, sect_K_pct, sect_L_pct, sect_M_pct, sect_N_pct, sect_O_pct, sect_P_pct, sect_Q_pct, sect_R_pct, sect_S_pct, sect_T_pct, env_score_adjustment, env_act_Disposal , env_act_End-use, env_act_Farming, env_act_Manufacturing, env_act_Operation, env_act_Other, env_act_Raw materials, env_act_Transportation, sdg_id_2, sdg_id_3, sdg_id_4, sdg_id_5, sdg_id_6, sdg_id_7, sdg_id_8, sdg_id_9, sdg_id_11, sdg_id_12, sdg_id_13, sdg_id_16, revenue_log, region_code_ANZ, region_code_CAR, region_code_EA, region_code_EEU, region_code_LATAM, region_code_NAM, region_code_WEU, country_code_AT, country_code_BE, country_code_CA, country_code_CH, country_code_CN, country_code_CZ, country_code_DE, country_code_DK, country_code_EE, country_code_ES, country_code_FI, country_code_FR, country_code_GB, country_code_GI, country_code_IE, country_code_IT, country_code_JE, country_code_JP, country_code_LU, country_code_MT, country_code_MX, country_code_NL, country_code_NO, country_code_NZ, country_code_PR, country_code_PT, country_code_SE, country_code_US

---
### lasso_abs_revenue
- **Status**: Failed
- **Model**: LassoModel
- **Preprocessor**: AbsoluteSectorRevenuePreprocessor
- **Error**: 
```

  File "/Users/jj/fitch-codeathon2025/.venv/lib/python3.12/site-packages/sklearn/utils/validation.py", line 2929, in validate_data
    _check_feature_names(_estimator, X, reset=reset)
  File "/Users/jj/fitch-codeathon2025/.venv/lib/python3.12/site-packages/sklearn/utils/validation.py", line 2787, in _check_feature_names
    raise ValueError(message)
ValueError: The feature names should match those that were passed during fit.
Feature names seen at fit time, yet now missing:
- region_group_OTHER
```

---
### linear_abs_revenue
- **Status**: Failed
- **Model**: LinearRegressionModel
- **Preprocessor**: AbsoluteSectorRevenuePreprocessor
- **Error**: 
```

  File "/Users/jj/fitch-codeathon2025/.venv/lib/python3.12/site-packages/sklearn/utils/validation.py", line 2929, in validate_data
    _check_feature_names(_estimator, X, reset=reset)
  File "/Users/jj/fitch-codeathon2025/.venv/lib/python3.12/site-packages/sklearn/utils/validation.py", line 2787, in _check_feature_names
    raise ValueError(message)
ValueError: The feature names should match those that were passed during fit.
Feature names seen at fit time, yet now missing:
- region_group_OTHER
```

---
### linear_comprehensive
- **Status**: Failed
- **Model**: LinearRegressionModel
- **Preprocessor**: ComprehensivePreprocessor
- **Error**: 
```
utils/validation.py", line 1105, in check_array
    _assert_all_finite(
  File "/Users/jj/fitch-codeathon2025/.venv/lib/python3.12/site-packages/sklearn/utils/validation.py", line 120, in _assert_all_finite
    _assert_all_finite_element_wise(
  File "/Users/jj/fitch-codeathon2025/.venv/lib/python3.12/site-packages/sklearn/utils/validation.py", line 169, in _assert_all_finite_element_wise
    raise ValueError(msg_err)
ValueError: Input contains infinity or a value too large for dtype('float64').
```

---
### ridge_abs_revenue
- **Status**: Failed
- **Model**: RidgeModel
- **Preprocessor**: AbsoluteSectorRevenuePreprocessor
- **Error**: 
```

  File "/Users/jj/fitch-codeathon2025/.venv/lib/python3.12/site-packages/sklearn/utils/validation.py", line 2929, in validate_data
    _check_feature_names(_estimator, X, reset=reset)
  File "/Users/jj/fitch-codeathon2025/.venv/lib/python3.12/site-packages/sklearn/utils/validation.py", line 2787, in _check_feature_names
    raise ValueError(message)
ValueError: The feature names should match those that were passed during fit.
Feature names seen at fit time, yet now missing:
- region_group_OTHER
```

---
