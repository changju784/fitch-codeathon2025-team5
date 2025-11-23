from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class BasePreprocessor(ABC):
    @abstractmethod
    def fit(self, X, y=None):
        pass

    @abstractmethod
    def transform(self, X):
        pass

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

class BaselinePreprocessor(BasePreprocessor):
    def __init__(self):
        self.features = []

    def fit(self, X, y=None):
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        exclude_cols = ['target_scope_1', 'target_scope_2', 'entity_id']
        self.features = [c for c in numeric_cols if c not in exclude_cols]
        return self

    def transform(self, X):
        X_transformed = X[self.features].fillna(0)
        return X_transformed

class CodeathonBaselinePreprocessor(BasePreprocessor):
    def __init__(self):
        self.features = []
        self.one_hot_encoder = None
        self.selected_features = []

    def fit(self, X, y=None):
        sect = pd.read_csv("data/revenue_distribution_by_sector.csv")
        env = pd.read_csv("data/environmental_activities.csv")
        sdg = pd.read_csv("data/sustainable_development_goals.csv")

        level_1_sect = sect.pivot_table(
            values='revenue_pct',
            index='entity_id',
            columns='nace_level_1_code',
            aggfunc='sum',
            fill_value=0
        ).add_prefix('sect_').add_suffix('_pct').reset_index()

        env_adjustment = env.groupby('entity_id')['env_score_adjustment'].sum().reset_index()

        sdg['present'] = 1
        sdg_features = sdg.pivot_table(
            values='present',
            index='entity_id',
            columns='sdg_id',
            aggfunc='max',
            fill_value=0
        ).add_prefix('sdg_id_').reset_index()

        X_temp = X.copy()
        
        self.one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.one_hot_encoder.fit(X_temp[['region_code']])
        
        region_encoded = self.one_hot_encoder.transform(X_temp[['region_code']])
        region_feature_names = self.one_hot_encoder.get_feature_names_out(['region_code'])
        region_df = pd.DataFrame(region_encoded, columns=region_feature_names, index=X_temp.index)
        
        X_temp = X_temp.merge(level_1_sect, on='entity_id', how='left')
        X_temp = X_temp.merge(env_adjustment, on='entity_id', how='left')
        X_temp = X_temp.merge(sdg_features, on='entity_id', how='left')
        
        if 'env_score_adjustment' in X_temp.columns:
            X_temp['env_score_adjustment'] = X_temp['env_score_adjustment'].fillna(0)
        
        cols_to_fill = [c for c in X_temp.columns if c.startswith('sect_') or c.startswith('sdg_id_')]
        X_temp[cols_to_fill] = X_temp[cols_to_fill].fillna(0)

        X_temp = pd.concat([X_temp, region_df], axis=1)

        exclude_cols = ['entity_id', 'target_scope_1', 'target_scope_2', 'region_code', 'country_code', 'region_name', 'country_name', 'name', 'isin', 'lei', 'bvd_id', 'ticker']
        
        numeric_cols = X_temp.select_dtypes(include=['number']).columns.tolist()
        candidate_features = [c for c in numeric_cols if c not in exclude_cols]
        
        variances = X_temp[candidate_features].var()
        self.selected_features = variances[variances > 0.05].index.tolist()
        
        print(f"Selected {len(self.selected_features)} features with variance > 0.05")
        
        return self

    def transform(self, X):
        sect = pd.read_csv("data/revenue_distribution_by_sector.csv")
        env = pd.read_csv("data/environmental_activities.csv")
        sdg = pd.read_csv("data/sustainable_development_goals.csv")

        level_1_sect = sect.pivot_table(
            values='revenue_pct',
            index='entity_id',
            columns='nace_level_1_code',
            aggfunc='sum',
            fill_value=0
        ).add_prefix('sect_').add_suffix('_pct').reset_index()

        env_adjustment = env.groupby('entity_id')['env_score_adjustment'].sum().reset_index()

        sdg['present'] = 1
        sdg_features = sdg.pivot_table(
            values='present',
            index='entity_id',
            columns='sdg_id',
            aggfunc='max',
            fill_value=0
        ).add_prefix('sdg_id_').reset_index()

        X_temp = X.copy()
        
        region_encoded = self.one_hot_encoder.transform(X_temp[['region_code']])
        region_feature_names = self.one_hot_encoder.get_feature_names_out(['region_code'])
        region_df = pd.DataFrame(region_encoded, columns=region_feature_names, index=X_temp.index)
        
        X_temp = X_temp.merge(level_1_sect, on='entity_id', how='left')
        X_temp = X_temp.merge(env_adjustment, on='entity_id', how='left')
        X_temp = X_temp.merge(sdg_features, on='entity_id', how='left')
        
        if 'env_score_adjustment' in X_temp.columns:
            X_temp['env_score_adjustment'] = X_temp['env_score_adjustment'].fillna(0)
            
        cols_to_fill = [c for c in X_temp.columns if c.startswith('sect_') or c.startswith('sdg_id_')]
        X_temp[cols_to_fill] = X_temp[cols_to_fill].fillna(0)
        
        X_temp = pd.concat([X_temp, region_df], axis=1)
        
        for feature in self.selected_features:
            if feature not in X_temp.columns:
                X_temp[feature] = 0
                
        X_final = X_temp[self.selected_features].fillna(0)
        
        return X_final

class ComprehensivePreprocessor(BasePreprocessor):
    def __init__(self):
        self.features = []
        self.one_hot_encoder = None
        self.scaler = None
        self.feature_names = []

    def fit(self, X, y=None):
        # 1. Load auxiliary data
        sect = pd.read_csv("data/revenue_distribution_by_sector.csv")
        env = pd.read_csv("data/environmental_activities.csv")
        sdg = pd.read_csv("data/sustainable_development_goals.csv")

        # 2. Process Sector Data (Revenue percentages)
        level_1_sect = sect.pivot_table(
            values='revenue_pct',
            index='entity_id',
            columns='nace_level_1_code',
            aggfunc='sum',
            fill_value=0
        ).add_prefix('sect_').add_suffix('_pct').reset_index()

        # 3. Process Environmental Data
        # a) env_score_adjustment sum
        env_adjustment = env.groupby('entity_id')['env_score_adjustment'].sum().reset_index()
        
        # b) activity_type binary features
        env['present'] = 1
        env_activities = env.pivot_table(
            values='present',
            index='entity_id',
            columns='activity_type',
            aggfunc='max',
            fill_value=0
        ).add_prefix('env_act_').reset_index()

        # 4. Process SDG Data (Binary)
        sdg['present'] = 1
        sdg_features = sdg.pivot_table(
            values='present',
            index='entity_id',
            columns='sdg_id',
            aggfunc='max',
            fill_value=0
        ).add_prefix('sdg_id_').reset_index()

        X_temp = X.copy()
        
        # 5. One Hot Encode Region and Country
        self.one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        # Fill NaNs in categorical columns before fitting
        X_temp['region_code'] = X_temp['region_code'].fillna('Unknown')
        X_temp['country_code'] = X_temp['country_code'].fillna('Unknown')
        
        self.one_hot_encoder.fit(X_temp[['region_code', 'country_code']])
        
        cat_encoded = self.one_hot_encoder.transform(X_temp[['region_code', 'country_code']])
        cat_feature_names = self.one_hot_encoder.get_feature_names_out(['region_code', 'country_code'])
        cat_df = pd.DataFrame(cat_encoded, columns=cat_feature_names, index=X_temp.index)
        
        # 6. Merge all auxiliary features
        X_temp = X_temp.merge(level_1_sect, on='entity_id', how='left')
        X_temp = X_temp.merge(env_adjustment, on='entity_id', how='left')
        X_temp = X_temp.merge(env_activities, on='entity_id', how='left')
        X_temp = X_temp.merge(sdg_features, on='entity_id', how='left')
        
        # 7. Handle Missing Values for Merged Features
        if 'env_score_adjustment' in X_temp.columns:
            X_temp['env_score_adjustment'] = X_temp['env_score_adjustment'].fillna(0)
            
        # Fill sector, env activity, and sdg NaNs with 0
        cols_to_fill = [c for c in X_temp.columns if c.startswith('sect_') or c.startswith('sdg_id_') or c.startswith('env_act_')]
        X_temp[cols_to_fill] = X_temp[cols_to_fill].fillna(0)
        
        # 8. Log Transform Revenue
        if 'revenue' in X_temp.columns:
            X_temp['revenue_log'] = np.log1p(X_temp['revenue'])
        
        # 9. Select Features
        # Include ESG scores: environmental_score, social_score, governance_score
        # Exclude overall_score as requested
        esg_cols = ['environmental_score', 'social_score', 'governance_score']
        # Fill ESG NaNs with mean or 0? Let's use 0 for now to be safe/simple, or mean if we want. 
        # Given standard scaler later, mean imputation might be better, but let's stick to 0 (missing info) or median.
        # Let's use 0 for consistency with other features unless specified.
        X_temp[esg_cols] = X_temp[esg_cols].fillna(0)
        
        # Concatenate categorical features
        X_temp = pd.concat([X_temp, cat_df], axis=1)
        
        # Define feature list
        # Start with numeric columns from original X (excluding targets/ids/overall_score/revenue raw)
        exclude_cols = ['entity_id', 'target_scope_1', 'target_scope_2', 'region_code', 'country_code', 'region_name', 'country_name', 'name', 'isin', 'lei', 'bvd_id', 'ticker', 'revenue', 'overall_score']
        
        # We want to keep: revenue_log, esg_cols, sect_*, env_score_adjustment, env_act_*, sdg_id_*, cat_feature_names
        # And any other numeric columns in X that are not excluded
        
        # Get all numeric columns from current X_temp
        numeric_cols = X_temp.select_dtypes(include=['number']).columns.tolist()
        self.feature_names = [c for c in numeric_cols if c not in exclude_cols]
        
        print(f"Comprehensive Preprocessor: Selected {len(self.feature_names)} features.")
        
        return self

    def transform(self, X):
        # Re-load aux data (same as fit)
        sect = pd.read_csv("data/revenue_distribution_by_sector.csv")
        env = pd.read_csv("data/environmental_activities.csv")
        sdg = pd.read_csv("data/sustainable_development_goals.csv")

        level_1_sect = sect.pivot_table(
            values='revenue_pct',
            index='entity_id',
            columns='nace_level_1_code',
            aggfunc='sum',
            fill_value=0
        ).add_prefix('sect_').add_suffix('_pct').reset_index()

        env_adjustment = env.groupby('entity_id')['env_score_adjustment'].sum().reset_index()
        
        env['present'] = 1
        env_activities = env.pivot_table(
            values='present',
            index='entity_id',
            columns='activity_type',
            aggfunc='max',
            fill_value=0
        ).add_prefix('env_act_').reset_index()

        sdg['present'] = 1
        sdg_features = sdg.pivot_table(
            values='present',
            index='entity_id',
            columns='sdg_id',
            aggfunc='max',
            fill_value=0
        ).add_prefix('sdg_id_').reset_index()

        X_temp = X.copy()
        
        # Categorical
        X_temp['region_code'] = X_temp['region_code'].fillna('Unknown')
        X_temp['country_code'] = X_temp['country_code'].fillna('Unknown')
        
        cat_encoded = self.one_hot_encoder.transform(X_temp[['region_code', 'country_code']])
        cat_feature_names = self.one_hot_encoder.get_feature_names_out(['region_code', 'country_code'])
        cat_df = pd.DataFrame(cat_encoded, columns=cat_feature_names, index=X_temp.index)
        
        # Merge
        X_temp = X_temp.merge(level_1_sect, on='entity_id', how='left')
        X_temp = X_temp.merge(env_adjustment, on='entity_id', how='left')
        X_temp = X_temp.merge(env_activities, on='entity_id', how='left')
        X_temp = X_temp.merge(sdg_features, on='entity_id', how='left')
        
        # Fill NaNs
        if 'env_score_adjustment' in X_temp.columns:
            X_temp['env_score_adjustment'] = X_temp['env_score_adjustment'].fillna(0)
            
        cols_to_fill = [c for c in X_temp.columns if c.startswith('sect_') or c.startswith('sdg_id_') or c.startswith('env_act_')]
        X_temp[cols_to_fill] = X_temp[cols_to_fill].fillna(0)
        
        # Log Revenue
        if 'revenue' in X_temp.columns:
            X_temp['revenue_log'] = np.log1p(X_temp['revenue'])
            
        # ESG Fill
        esg_cols = ['environmental_score', 'social_score', 'governance_score']
        X_temp[esg_cols] = X_temp[esg_cols].fillna(0)
        
        # Concat
        X_temp = pd.concat([X_temp, cat_df], axis=1)
        
        # Ensure all features exist
        for feature in self.feature_names:
            if feature not in X_temp.columns:
                X_temp[feature] = 0
                
        X_final = X_temp[self.feature_names].fillna(0)
        
        return X_final

class TopFeaturesPreprocessor(BasePreprocessor):
    def __init__(self):
        self.features = []
        self.one_hot_encoder = None
        self.feature_names = []
        self.top_5_sdgs = [3, 9, 11, 12, 7]
        self.top_5_sectors = ['C', 'J', 'G', 'M', 'N']

    def fit(self, X, y=None):
        # 1. Load auxiliary data
        sect = pd.read_csv("data/revenue_distribution_by_sector.csv")
        env = pd.read_csv("data/environmental_activities.csv")
        sdg = pd.read_csv("data/sustainable_development_goals.csv")

        # 2. Process Sector Data (Top 5 + OTHER)
        sect['nace_group'] = sect['nace_level_1_code'].apply(lambda x: x if x in self.top_5_sectors else 'OTHER')
        level_1_sect = sect.pivot_table(
            values='revenue_pct',
            index='entity_id',
            columns='nace_group',
            aggfunc='sum',
            fill_value=0
        ).add_prefix('sect_').add_suffix('_pct').reset_index()

        # 3. Process Environmental Data (Sum only)
        env_adjustment = env.groupby('entity_id')['env_score_adjustment'].sum().reset_index()

        # 4. Process SDG Data (Top 5 + OTHER, Binary)
        sdg['sdg_group'] = sdg['sdg_id'].apply(lambda x: x if x in self.top_5_sdgs else 'OTHER')
        sdg['present'] = 1
        sdg_features = sdg.pivot_table(
            values='present',
            index='entity_id',
            columns='sdg_group',
            aggfunc='max',
            fill_value=0
        ).add_prefix('sdg_').reset_index()

        X_temp = X.copy()
        
        # 5. Region (WEU, NAM, OTHER)
        X_temp['region_group'] = X_temp['region_code'].apply(lambda x: x if x in ['WEU', 'NAM'] else 'OTHER')
        
        self.one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.one_hot_encoder.fit(X_temp[['region_group']])
        
        region_encoded = self.one_hot_encoder.transform(X_temp[['region_group']])
        region_feature_names = self.one_hot_encoder.get_feature_names_out(['region_group'])
        region_df = pd.DataFrame(region_encoded, columns=region_feature_names, index=X_temp.index)
        
        # 6. Merge all
        X_temp = X_temp.merge(level_1_sect, on='entity_id', how='left')
        X_temp = X_temp.merge(env_adjustment, on='entity_id', how='left')
        X_temp = X_temp.merge(sdg_features, on='entity_id', how='left')
        
        # 7. Handle Missing Values
        if 'env_score_adjustment' in X_temp.columns:
            X_temp['env_score_adjustment'] = X_temp['env_score_adjustment'].fillna(0)
            
        cols_to_fill = [c for c in X_temp.columns if c.startswith('sect_') or c.startswith('sdg_')]
        X_temp[cols_to_fill] = X_temp[cols_to_fill].fillna(0)
        
        # 8. Log Transform Revenue
        if 'revenue' in X_temp.columns:
            X_temp['revenue_log'] = np.log1p(X_temp['revenue'])
            
        # 9. Select Features
        # ESG: E, S, G only
        esg_cols = ['environmental_score', 'social_score', 'governance_score']
        X_temp[esg_cols] = X_temp[esg_cols].fillna(0)
        
        # Concatenate region features
        X_temp = pd.concat([X_temp, region_df], axis=1)
        
        # Define feature list
        # Exclude original raw columns and IDs
        exclude_cols = ['entity_id', 'target_scope_1', 'target_scope_2', 'region_code', 'country_code', 
                        'region_name', 'country_name', 'name', 'isin', 'lei', 'bvd_id', 'ticker', 
                        'revenue', 'overall_score', 'region_group']
        
        numeric_cols = X_temp.select_dtypes(include=['number']).columns.tolist()
        self.feature_names = [c for c in numeric_cols if c not in exclude_cols]
        
        print(f"TopFeatures Preprocessor: Selected {len(self.feature_names)} features.")
        print(f"Features: {self.feature_names}")
        
        return self

    def transform(self, X):
        # Re-load aux data (same as fit)
        sect = pd.read_csv("data/revenue_distribution_by_sector.csv")
        env = pd.read_csv("data/environmental_activities.csv")
        sdg = pd.read_csv("data/sustainable_development_goals.csv")

        # Process Sector
        sect['nace_group'] = sect['nace_level_1_code'].apply(lambda x: x if x in self.top_5_sectors else 'OTHER')
        level_1_sect = sect.pivot_table(
            values='revenue_pct',
            index='entity_id',
            columns='nace_group',
            aggfunc='sum',
            fill_value=0
        ).add_prefix('sect_').add_suffix('_pct').reset_index()

        # Process Env
        env_adjustment = env.groupby('entity_id')['env_score_adjustment'].sum().reset_index()

        # Process SDG
        sdg['sdg_group'] = sdg['sdg_id'].apply(lambda x: x if x in self.top_5_sdgs else 'OTHER')
        sdg['present'] = 1
        sdg_features = sdg.pivot_table(
            values='present',
            index='entity_id',
            columns='sdg_group',
            aggfunc='max',
            fill_value=0
        ).add_prefix('sdg_').reset_index()

        X_temp = X.copy()
        
        # Region
        X_temp['region_group'] = X_temp['region_code'].apply(lambda x: x if x in ['WEU', 'NAM'] else 'OTHER')
        
        region_encoded = self.one_hot_encoder.transform(X_temp[['region_group']])
        region_feature_names = self.one_hot_encoder.get_feature_names_out(['region_group'])
        region_df = pd.DataFrame(region_encoded, columns=region_feature_names, index=X_temp.index)
        
        # Merge
        X_temp = X_temp.merge(level_1_sect, on='entity_id', how='left')
        X_temp = X_temp.merge(env_adjustment, on='entity_id', how='left')
        X_temp = X_temp.merge(sdg_features, on='entity_id', how='left')
        
        # Fill NaNs
        if 'env_score_adjustment' in X_temp.columns:
            X_temp['env_score_adjustment'] = X_temp['env_score_adjustment'].fillna(0)
            
        cols_to_fill = [c for c in X_temp.columns if c.startswith('sect_') or c.startswith('sdg_')]
        X_temp[cols_to_fill] = X_temp[cols_to_fill].fillna(0)
        
        # Log Revenue
        if 'revenue' in X_temp.columns:
            X_temp['revenue_log'] = np.log1p(X_temp['revenue'])
            
        # ESG Fill
        esg_cols = ['environmental_score', 'social_score', 'governance_score']
        X_temp[esg_cols] = X_temp[esg_cols].fillna(0)
        
        # Concat
        X_temp = pd.concat([X_temp, region_df], axis=1)
        
        # Ensure all features exist
        for feature in self.feature_names:
            if feature not in X_temp.columns:
                X_temp[feature] = 0
                
        X_final = X_temp[self.feature_names].fillna(0)
        
        return X_final


class TreePreprocessor(BasePreprocessor):
    def __init__(self):
        self.features = []
        self.feature_names = []
        self.cat_features = []
        self.top_5_sdgs = [3, 9, 11, 12, 7]
        self.top_5_sectors = ['C', 'J', 'G', 'M', 'N']

    def fit(self, X, y=None):
        # 1. Load auxiliary data
        sect = pd.read_csv("data/revenue_distribution_by_sector.csv")
        env = pd.read_csv("data/environmental_activities.csv")
        sdg = pd.read_csv("data/sustainable_development_goals.csv")

        # 2. Process Sector Data (Top 5 + OTHER)
        sect['nace_group'] = sect['nace_level_1_code'].apply(lambda x: x if x in self.top_5_sectors else 'OTHER')
        level_1_sect = sect.pivot_table(
            values='revenue_pct',
            index='entity_id',
            columns='nace_group',
            aggfunc='sum',
            fill_value=0
        ).add_prefix('sect_').add_suffix('_pct').reset_index()

        # 3. Process Environmental Data (Sum only)
        env_adjustment = env.groupby('entity_id')['env_score_adjustment'].sum().reset_index()

        # 4. Process SDG Data (Top 5 + OTHER, Binary)
        sdg['sdg_group'] = sdg['sdg_id'].apply(lambda x: x if x in self.top_5_sdgs else 'OTHER')
        sdg['present'] = 1
        sdg_features = sdg.pivot_table(
            values='present',
            index='entity_id',
            columns='sdg_group',
            aggfunc='max',
            fill_value=0
        ).add_prefix('sdg_').reset_index()

        X_temp = X.copy()
        
        # 5. Region (WEU, NAM, OTHER) - Keep as Categorical
        X_temp['region_group'] = X_temp['region_code'].apply(lambda x: x if x in ['WEU', 'NAM'] else 'OTHER')
        X_temp['region_group'] = X_temp['region_group'].astype('category')
        
        # 6. Merge all
        X_temp = X_temp.merge(level_1_sect, on='entity_id', how='left')
        X_temp = X_temp.merge(env_adjustment, on='entity_id', how='left')
        X_temp = X_temp.merge(sdg_features, on='entity_id', how='left')
        
        # 7. Handle Missing Values
        if 'env_score_adjustment' in X_temp.columns:
            X_temp['env_score_adjustment'] = X_temp['env_score_adjustment'].fillna(0)
            
        cols_to_fill = [c for c in X_temp.columns if c.startswith('sect_') or c.startswith('sdg_')]
        X_temp[cols_to_fill] = X_temp[cols_to_fill].fillna(0)
        
        # 8. Log Transform Revenue
        if 'revenue' in X_temp.columns:
            X_temp['revenue_log'] = np.log1p(X_temp['revenue'])
            
        # 9. Select Features
        # ESG: E, S, G only
        esg_cols = ['environmental_score', 'social_score', 'governance_score']
        X_temp[esg_cols] = X_temp[esg_cols].fillna(0)
        
        # Define feature list
        # Exclude original raw columns and IDs
        exclude_cols = ['entity_id', 'target_scope_1', 'target_scope_2', 'region_code', 'country_code', 
                        'region_name', 'country_name', 'name', 'isin', 'lei', 'bvd_id', 'ticker', 
                        'revenue', 'overall_score']
        
        # Select numeric and categorical features
        # Note: region_group is categorical
        
        all_cols = X_temp.columns.tolist()
        self.feature_names = [c for c in all_cols if c not in exclude_cols]
        
        # Identify categorical features
        self.cat_features = ['region_group']
        
        print(f"Tree Preprocessor: Selected {len(self.feature_names)} features.")
        print(f"Categorical Features: {self.cat_features}")
        
        return self

    def transform(self, X):
        # Re-load aux data (same as fit)
        sect = pd.read_csv("data/revenue_distribution_by_sector.csv")
        env = pd.read_csv("data/environmental_activities.csv")
        sdg = pd.read_csv("data/sustainable_development_goals.csv")

        # Process Sector
        sect['nace_group'] = sect['nace_level_1_code'].apply(lambda x: x if x in self.top_5_sectors else 'OTHER')
        level_1_sect = sect.pivot_table(
            values='revenue_pct',
            index='entity_id',
            columns='nace_group',
            aggfunc='sum',
            fill_value=0
        ).add_prefix('sect_').add_suffix('_pct').reset_index()

        # Process Env
        env_adjustment = env.groupby('entity_id')['env_score_adjustment'].sum().reset_index()

        # Process SDG
        sdg['sdg_group'] = sdg['sdg_id'].apply(lambda x: x if x in self.top_5_sdgs else 'OTHER')
        sdg['present'] = 1
        sdg_features = sdg.pivot_table(
            values='present',
            index='entity_id',
            columns='sdg_group',
            aggfunc='max',
            fill_value=0
        ).add_prefix('sdg_').reset_index()

        X_temp = X.copy()
        
        # Region
        X_temp['region_group'] = X_temp['region_code'].apply(lambda x: x if x in ['WEU', 'NAM'] else 'OTHER')
        X_temp['region_group'] = X_temp['region_group'].astype('category')
        
        # Merge
        X_temp = X_temp.merge(level_1_sect, on='entity_id', how='left')
        X_temp = X_temp.merge(env_adjustment, on='entity_id', how='left')
        X_temp = X_temp.merge(sdg_features, on='entity_id', how='left')
        
        # Fill NaNs
        if 'env_score_adjustment' in X_temp.columns:
            X_temp['env_score_adjustment'] = X_temp['env_score_adjustment'].fillna(0)
            
        cols_to_fill = [c for c in X_temp.columns if c.startswith('sect_') or c.startswith('sdg_')]
        X_temp[cols_to_fill] = X_temp[cols_to_fill].fillna(0)
        
        # Log Revenue
        if 'revenue' in X_temp.columns:
            X_temp['revenue_log'] = np.log1p(X_temp['revenue'])
            
        # ESG Fill
        esg_cols = ['environmental_score', 'social_score', 'governance_score']
        X_temp[esg_cols] = X_temp[esg_cols].fillna(0)
        
        # Ensure all features exist
        for feature in self.feature_names:
            if feature not in X_temp.columns:
                if feature in self.cat_features:
                    X_temp[feature] = 'OTHER' # Default for categorical
                    X_temp[feature] = X_temp[feature].astype('category')
                else:
                    X_temp[feature] = 0
        
        X_final = X_temp[self.feature_names]
        
        # Ensure categorical type is preserved
        for cat in self.cat_features:
            X_final[cat] = X_final[cat].astype('category')
            
        # Fill NaNs in numeric columns
        numeric_cols = [c for c in self.feature_names if c not in self.cat_features]
        X_final[numeric_cols] = X_final[numeric_cols].fillna(0)
        
        return X_final


class AbsoluteRevenuePreprocessor(BasePreprocessor):
    """
    Similar to TreePreprocessor, but converts sector revenue percentages 
    to absolute revenue amounts by multiplying with total revenue.
    """
    def __init__(self):
        self.features = []
        self.feature_names = []
        self.cat_features = []
        self.top_5_sdgs = [3, 9, 11, 12, 7]
        self.top_5_sectors = ['C', 'J', 'G', 'M', 'N']

    def fit(self, X, y=None):
        # 1. Load auxiliary data
        sect = pd.read_csv("data/revenue_distribution_by_sector.csv")
        env = pd.read_csv("data/environmental_activities.csv")
        sdg = pd.read_csv("data/sustainable_development_goals.csv")

        X_temp = X.copy()
        
        # 2. Process Sector Data (Top 5 + OTHER) - Convert to ABSOLUTE revenue
        sect['nace_group'] = sect['nace_level_1_code'].apply(lambda x: x if x in self.top_5_sectors else 'OTHER')
        level_1_sect_pct = sect.pivot_table(
            values='revenue_pct',
            index='entity_id',
            columns='nace_group',
            aggfunc='sum',
            fill_value=0
        ).reset_index()
        
        # Merge with revenue to compute absolute values
        level_1_sect = level_1_sect_pct.merge(X_temp[['entity_id', 'revenue']], on='entity_id', how='left')
        level_1_sect['revenue'] = level_1_sect['revenue'].fillna(0)
        
        # Convert percentages to absolute revenue amounts
        for col in [c for c in level_1_sect.columns if c not in ['entity_id', 'revenue']]:
            level_1_sect[f'sect_{col}_abs'] = level_1_sect[col] * level_1_sect['revenue'] / 100.0
        
        # Keep only entity_id and absolute revenue columns
        abs_cols = ['entity_id'] + [c for c in level_1_sect.columns if c.endswith('_abs')]
        level_1_sect = level_1_sect[abs_cols]

        # 3. Process Environmental Data (Sum only)
        env_adjustment = env.groupby('entity_id')['env_score_adjustment'].sum().reset_index()

        # 4. Process SDG Data (Top 5 + OTHER, Binary)
        sdg['sdg_group'] = sdg['sdg_id'].apply(lambda x: x if x in self.top_5_sdgs else 'OTHER')
        sdg['present'] = 1
        sdg_features = sdg.pivot_table(
            values='present',
            index='entity_id',
            columns='sdg_group',
            aggfunc='max',
            fill_value=0
        ).add_prefix('sdg_').reset_index()
        
        # 5. Region (WEU, NAM, OTHER) - Keep as Categorical
        X_temp['region_group'] = X_temp['region_code'].apply(lambda x: x if x in ['WEU', 'NAM'] else 'OTHER')
        X_temp['region_group'] = X_temp['region_group'].astype('category')
        
        # 6. Merge all
        X_temp = X_temp.merge(level_1_sect, on='entity_id', how='left')
        X_temp = X_temp.merge(env_adjustment, on='entity_id', how='left')
        X_temp = X_temp.merge(sdg_features, on='entity_id', how='left')
        
        # 7. Handle Missing Values
        if 'env_score_adjustment' in X_temp.columns:
            X_temp['env_score_adjustment'] = X_temp['env_score_adjustment'].fillna(0)
            
        cols_to_fill = [c for c in X_temp.columns if c.startswith('sect_') or c.startswith('sdg_')]
        X_temp[cols_to_fill] = X_temp[cols_to_fill].fillna(0)
        
        # 8. Log Transform Revenue (keep original revenue for reference)
        if 'revenue' in X_temp.columns:
            X_temp['revenue_log'] = np.log1p(X_temp['revenue'])
            
        # 9. Select Features
        # ESG: E, S, G only
        esg_cols = ['environmental_score', 'social_score', 'governance_score']
        X_temp[esg_cols] = X_temp[esg_cols].fillna(0)
        
        # Define feature list
        # Exclude original raw columns and IDs
        exclude_cols = ['entity_id', 'target_scope_1', 'target_scope_2', 'region_code', 'country_code', 
                        'region_name', 'country_name', 'name', 'isin', 'lei', 'bvd_id', 'ticker', 
                        'revenue', 'overall_score']
        
        # Select numeric and categorical features
        # Note: region_group is categorical
        
        all_cols = X_temp.columns.tolist()
        self.feature_names = [c for c in all_cols if c not in exclude_cols]
        
        # Identify categorical features
        self.cat_features = ['region_group']
        
        print(f"AbsoluteRevenue Preprocessor: Selected {len(self.feature_names)} features.")
        print(f"Categorical Features: {self.cat_features}")
        print(f"Sector absolute revenue features: {[c for c in self.feature_names if 'sect_' in c]}")
        
        return self

    def transform(self, X):
        # Re-load aux data (same as fit)
        sect = pd.read_csv("data/revenue_distribution_by_sector.csv")
        env = pd.read_csv("data/environmental_activities.csv")
        sdg = pd.read_csv("data/sustainable_development_goals.csv")

        X_temp = X.copy()
        
        # Process Sector - Convert to ABSOLUTE revenue
        sect['nace_group'] = sect['nace_level_1_code'].apply(lambda x: x if x in self.top_5_sectors else 'OTHER')
        level_1_sect_pct = sect.pivot_table(
            values='revenue_pct',
            index='entity_id',
            columns='nace_group',
            aggfunc='sum',
            fill_value=0
        ).reset_index()
        
        # Merge with revenue to compute absolute values
        level_1_sect = level_1_sect_pct.merge(X_temp[['entity_id', 'revenue']], on='entity_id', how='left')
        level_1_sect['revenue'] = level_1_sect['revenue'].fillna(0)
        
        # Convert percentages to absolute revenue amounts
        for col in [c for c in level_1_sect.columns if c not in ['entity_id', 'revenue']]:
            level_1_sect[f'sect_{col}_abs'] = level_1_sect[col] * level_1_sect['revenue'] / 100.0
        
        # Keep only entity_id and absolute revenue columns
        abs_cols = ['entity_id'] + [c for c in level_1_sect.columns if c.endswith('_abs')]
        level_1_sect = level_1_sect[abs_cols]

        # Process Env
        env_adjustment = env.groupby('entity_id')['env_score_adjustment'].sum().reset_index()

        # Process SDG
        sdg['sdg_group'] = sdg['sdg_id'].apply(lambda x: x if x in self.top_5_sdgs else 'OTHER')
        sdg['present'] = 1
        sdg_features = sdg.pivot_table(
            values='present',
            index='entity_id',
            columns='sdg_group',
            aggfunc='max',
            fill_value=0
        ).add_prefix('sdg_').reset_index()
        
        # Region
        X_temp['region_group'] = X_temp['region_code'].apply(lambda x: x if x in ['WEU', 'NAM'] else 'OTHER')
        X_temp['region_group'] = X_temp['region_group'].astype('category')
        
        # Merge
        X_temp = X_temp.merge(level_1_sect, on='entity_id', how='left')
        X_temp = X_temp.merge(env_adjustment, on='entity_id', how='left')
        X_temp = X_temp.merge(sdg_features, on='entity_id', how='left')
        
        # Fill NaNs
        if 'env_score_adjustment' in X_temp.columns:
            X_temp['env_score_adjustment'] = X_temp['env_score_adjustment'].fillna(0)
            
        cols_to_fill = [c for c in X_temp.columns if c.startswith('sect_') or c.startswith('sdg_')]
        X_temp[cols_to_fill] = X_temp[cols_to_fill].fillna(0)
        
        # Log Revenue
        if 'revenue' in X_temp.columns:
            X_temp['revenue_log'] = np.log1p(X_temp['revenue'])
            
        # ESG Fill
        esg_cols = ['environmental_score', 'social_score', 'governance_score']
        X_temp[esg_cols] = X_temp[esg_cols].fillna(0)
        
        # Ensure all features exist
        for feature in self.feature_names:
            if feature not in X_temp.columns:
                if feature in self.cat_features:
                    X_temp[feature] = 'OTHER' # Default for categorical
                    X_temp[feature] = X_temp[feature].astype('category')
                else:
                    X_temp[feature] = 0
        
        X_final = X_temp[self.feature_names]
        
        # Ensure categorical type is preserved
        for cat in self.cat_features:
            X_final[cat] = X_final[cat].astype('category')
            
        # Fill NaNs in numeric columns
        numeric_cols = [c for c in self.feature_names if c not in self.cat_features]
        X_final[numeric_cols] = X_final[numeric_cols].fillna(0)
        
        return X_final


class MedianRegressionPreprocessor(BasePreprocessor):
    def __init__(self):
        self.features = []
        self.one_hot_encoder = None
        self.feature_names = []

    def fit(self, X, y=None):
        # 1. Load auxiliary data
        sect = pd.read_csv("data/revenue_distribution_by_sector.csv")
        env = pd.read_csv("data/environmental_activities.csv")
        
        self.one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        
        # Fit OHE on country_code
        # Fill NaNs
        X_temp = X.copy()
        X_temp['country_code'] = X_temp['country_code'].fillna('Unknown')
        self.one_hot_encoder.fit(X_temp[['country_code']])
        
        # Define feature names
        cat_feature_names = self.one_hot_encoder.get_feature_names_out(['country_code']).tolist()
        
        self.feature_names = [
            'log_total_revenue',
            'log_scope1_revenue',
            'log_scope2_revenue',
            'scope1_revenue_present',
            'scope2_revenue_present',
            'env_adjusted_score',
            'social_score',
            'governance_score'
        ] + cat_feature_names
        
        return self

    def transform(self, X):
        # Load necessary data
        sect = pd.read_csv("data/revenue_distribution_by_sector.csv")
        class_df = pd.read_csv("data/sector_emission_scope_classification.csv")
        env = pd.read_csv("data/environmental_activities.csv")
        
        X_temp = X.copy()
        
        # 1. Calculate Scope 1 & 2 Revenue
        sect_merged = sect.merge(class_df, on='nace_level_1_code', how='left')
        
        if 'revenue' not in X_temp.columns:
            X_temp['revenue'] = 0
            
        sect_with_rev = sect_merged.merge(X_temp[['entity_id', 'revenue']], on='entity_id', how='inner')
        
        sect_with_rev['scope_1_revenue_part'] = sect_with_rev['revenue'] * sect_with_rev['revenue_pct'] * sect_with_rev['affects_scope_1'].fillna(0).astype(int)
        sect_with_rev['scope_2_revenue_part'] = sect_with_rev['revenue'] * sect_with_rev['revenue_pct'] * sect_with_rev['affects_scope_2'].fillna(0).astype(int)
        
        # Aggregate
        entity_revenue_split = sect_with_rev.groupby('entity_id')[['scope_1_revenue_part', 'scope_2_revenue_part']].sum().reset_index()
        entity_revenue_split.rename(columns={'scope_1_revenue_part': 'scope_1_revenue', 'scope_2_revenue_part': 'scope_2_revenue'}, inplace=True)
        
        # Merge back to X
        X_temp = X_temp.merge(entity_revenue_split, on='entity_id', how='left')
        X_temp['scope_1_revenue'] = X_temp['scope_1_revenue'].fillna(0)
        X_temp['scope_2_revenue'] = X_temp['scope_2_revenue'].fillna(0)
        
        # 2. Generate Features
        
        # log_total_revenue
        X_temp['log_total_revenue'] = np.log1p(X_temp['revenue'])
        
        # log_scope1_revenue
        X_temp['log_scope1_revenue'] = np.log1p(X_temp['scope_1_revenue'])
        
        # log_scope2_revenue
        X_temp['log_scope2_revenue'] = np.log1p(X_temp['scope_2_revenue'])
        
        # scope1_revenue_present (0/1)
        X_temp['scope1_revenue_present'] = (X_temp['scope_1_revenue'] > 0).astype(int)
        
        # scope2_revenue_present (0/1)
        X_temp['scope2_revenue_present'] = (X_temp['scope_2_revenue'] > 0).astype(int)
        
        # env_adjusted_score
        env_adj_agg = env.groupby('entity_id')['env_score_adjustment'].sum().reset_index()
        X_temp = X_temp.merge(env_adj_agg, on='entity_id', how='left')
        X_temp['env_score_adjustment'] = X_temp['env_score_adjustment'].fillna(0)
        
        if 'environmental_score' in X_temp.columns:
             X_temp['env_adjusted_score'] = X_temp['environmental_score'] + X_temp['env_score_adjustment']
        else:
             X_temp['env_adjusted_score'] = 0
             
        X_temp['social_score'] = X_temp['social_score'].fillna(0)
        X_temp['governance_score'] = X_temp['governance_score'].fillna(0)
        
        # country_code OHE
        X_temp['country_code'] = X_temp['country_code'].fillna('Unknown')
        country_encoded = self.one_hot_encoder.transform(X_temp[['country_code']])
        country_cols = self.one_hot_encoder.get_feature_names_out(['country_code'])
        country_df = pd.DataFrame(country_encoded, columns=country_cols, index=X_temp.index)
        
        X_temp = pd.concat([X_temp, country_df], axis=1)
        
        # Select features
        for col in self.feature_names:
            if col not in X_temp.columns:
                X_temp[col] = 0
                
        return X_temp[self.feature_names].fillna(0)



