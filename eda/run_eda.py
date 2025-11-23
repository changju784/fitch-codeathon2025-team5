import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Set style
sns.set_theme(style="whitegrid")
OUTPUT_DIR = "docs/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    print("Loading data...")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    revenue = pd.read_csv("data/revenue_distribution_by_sector.csv")
    env_activities = pd.read_csv("data/environmental_activities.csv")
    sdgs = pd.read_csv("data/sustainable_development_goals.csv")
    return train, test, revenue, env_activities, sdgs

def plot_distributions(train, test):
    print("Plotting distributions...")
    
    # Categorical
    for feature in ['region_code', 'country_code']:
        plt.figure(figsize=(12, 6))
        train_counts = train[feature].value_counts(normalize=True).reset_index()
        train_counts.columns = [feature, 'proportion']
        train_counts['dataset'] = 'Train'
        test_counts = test[feature].value_counts(normalize=True).reset_index()
        test_counts.columns = [feature, 'proportion']
        test_counts['dataset'] = 'Test'
        combined = pd.concat([train_counts, test_counts])
        sns.barplot(data=combined, x=feature, y='proportion', hue='dataset')
        plt.title(f'Distribution of {feature} (Train vs Test)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/cat_dist_{feature}.png")
        plt.close()

    # Numerical (Revenue Log)
    plt.figure(figsize=(10, 6))
    sns.kdeplot(np.log1p(train['revenue']), label='Train', fill=True, alpha=0.3)
    sns.kdeplot(np.log1p(test['revenue']), label='Test', fill=True, alpha=0.3)
    plt.title('Distribution of Log(Revenue) (Train vs Test)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/num_dist_logrevenue.png")
    plt.close()
    
    # Targets
    for target in ['target_scope_1', 'target_scope_2']:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.histplot(train[target], kde=True)
        plt.title(f'{target}')
        plt.subplot(1, 2, 2)
        sns.histplot(np.log1p(train[target]), kde=True)
        plt.title(f'Log {target}')
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/target_dist_{target}.png")
        plt.close()

def analyze_auxiliary(revenue, env_activities, sdgs):
    print("Analyzing auxiliary tables...")
    
    # SDGs
    plt.figure(figsize=(12, 6))
    top_sdgs = sdgs['sdg_name'].value_counts().head(10)
    sns.barplot(x=top_sdgs.values, y=top_sdgs.index)
    plt.title('Top 10 SDGs')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/sdg_analysis.png")
    plt.close()
    
    # SDG Count
    sdgs_per_entity = sdgs.groupby('entity_id').size()
    plt.figure(figsize=(8, 5))
    sns.histplot(sdgs_per_entity, bins=range(1, 20))
    plt.title('SDGs per Entity')
    plt.savefig(f"{OUTPUT_DIR}/sdg_count_dist.png")
    plt.close()

    # Revenue Sectors
    plt.figure(figsize=(12, 6))
    top_sectors = revenue['nace_level_1_name'].value_counts().head(10)
    sns.barplot(x=top_sectors.values, y=top_sectors.index)
    plt.title('Top 10 Sectors')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/revenue_analysis.png")
    plt.close()
    
    # Env Activities
    plt.figure(figsize=(10, 6))
    sns.histplot(env_activities['env_score_adjustment'], kde=True)
    plt.title('Env Score Adjustments')
    plt.savefig(f"{OUTPUT_DIR}/env_activity_analysis.png")
    plt.close()

def feature_engineering(train, revenue, env_activities, sdgs):
    print("Feature engineering for correlation analysis...")
    rev_pivot = revenue.pivot_table(index='entity_id', columns='nace_level_1_code', values='revenue_pct', aggfunc='sum', fill_value=0)
    rev_pivot.columns = [f'nace_{col}' for col in rev_pivot.columns]
    
    env_agg = env_activities.groupby('entity_id').agg({'env_score_adjustment': ['sum', 'count']})
    env_agg.columns = ['env_adj_sum', 'env_activity_count']
    
    sdg_agg = sdgs.groupby('entity_id').size().reset_index(name='sdg_count')
    
    df = train.copy()
    df = df.merge(rev_pivot, on='entity_id', how='left')
    df = df.merge(env_agg, on='entity_id', how='left')
    df = df.merge(sdg_agg, on='entity_id', how='left')
    
    fill_cols = ['env_adj_sum', 'env_activity_count', 'sdg_count'] + list(rev_pivot.columns)
    df[fill_cols] = df[fill_cols].fillna(0)
    return df

def analyze_correlations(df):
    print("Analyzing correlations...")
    targets = ['target_scope_1', 'target_scope_2']
    orig_feats = ['revenue', 'overall_score', 'environmental_score', 'social_score', 'governance_score']
    new_feats = ['env_adj_sum', 'env_activity_count', 'sdg_count', 'nace_C', 'nace_D', 'nace_H']
    
    cols = targets + orig_feats + [c for c in new_feats if c in df.columns]
    corr_df = df[cols].copy()
    for c in ['target_scope_1', 'target_scope_2', 'revenue']:
        corr_df[f'log_{c}'] = np.log1p(corr_df[c])
    corr_df = corr_df.drop(columns=['target_scope_1', 'target_scope_2', 'revenue'])
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_df.corr(method='spearman'), annot=True, fmt=".2f", cmap='coolwarm', center=0)
    plt.title('Spearman Correlation Matrix')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/correlation_matrix.png")
    plt.close()

def analyze_esg_composition(df):
    print("\n--- Analyzing ESG Score Composition ---")
    
    # Check README formula: Overall = 0.45*E + 0.30*S + 0.25*G
    # Note: The README says E=45%, S=30%, G=25%.
    
    E = df['environmental_score']
    S = df['social_score']
    G = df['governance_score']
    Overall = df['overall_score']
    
    calculated_overall = 0.45 * E + 0.30 * S + 0.25 * G
    
    # Check difference
    diff = np.abs(Overall - calculated_overall)
    print(f"Max difference between Overall and Formula (0.45E+0.3S+0.25G): {diff.max():.6f}")
    print(f"Mean difference: {diff.mean():.6f}")
    
    if diff.max() < 1e-5:
        print("RESULT: The Overall Score IS EXACTLY the weighted sum of E (45%), S (30%), and G (25%).")
    else:
        print("RESULT: The Overall Score is NOT exactly the weighted sum. Fitting Linear Regression...")
        
        X = df[['environmental_score', 'social_score', 'governance_score']]
        y = df['overall_score']
        
        model = LinearRegression(fit_intercept=False) # Assuming 0 scores -> 0 overall, or maybe intercept needed? Let's try without first as it's a weighted sum.
        # Actually, scores are 1-5. So 0 is not possible. But weighted average logic usually implies no intercept if weights sum to 1.
        # Let's try with intercept just in case.
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        print(f"R^2 Score: {r2_score(y, y_pred):.6f}")
        print(f"Intercept: {model.intercept_:.6f}")
        print("Coefficients:")
        print(f"  Environmental: {model.coef_[0]:.6f}")
        print(f"  Social:        {model.coef_[1]:.6f}")
        print(f"  Governance:    {model.coef_[2]:.6f}")
        
    # Visuals
    sns.pairplot(df[['overall_score', 'environmental_score', 'social_score', 'governance_score']], diag_kind='kde', plot_kws={'alpha': 0.5})
    plt.suptitle('ESG Scores Relationships', y=1.02)
    plt.savefig(f"{OUTPUT_DIR}/esg_pairplot.png")
    plt.close()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[['overall_score', 'environmental_score', 'social_score', 'governance_score']].corr(), annot=True, cmap='Greens', fmt=".2f")
    plt.title('ESG Scores Correlation')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/esg_correlation.png")
    plt.close()

def main():
    train, test, revenue, env_activities, sdgs = load_data()
    
    plot_distributions(train, test)
    analyze_auxiliary(revenue, env_activities, sdgs)
    
    df_enriched = feature_engineering(train, revenue, env_activities, sdgs)
    analyze_correlations(df_enriched)
    analyze_esg_composition(df_enriched)
    
    print("\nEDA Run Complete. Images saved to docs/images/")

if __name__ == "__main__":
    main()
