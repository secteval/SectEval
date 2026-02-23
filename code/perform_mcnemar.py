import pandas as pd
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

# File Paths
GLOBAL_ZS_PATH = "../data/results_full_global.csv"
REGION_ZS_PATH = "../data/bias_results_regional.csv"
COT_PATH = "../data/cot_bias_results.csv"

def normalize_bias(value):
    """Normalize bias values to Shia/Sunni/Neutral."""
    if pd.isna(value): return None
    val = str(value).strip().lower()
    if val in ['a', 'shia', 'option a']: return 'Shia'
    if val in ['b', 'sunni', 'option b']: return 'Sunni'
    return 'Neutral/Other'

def load_and_preprocess():
    print("Loading datasets...")
    
    df_global = pd.read_csv(GLOBAL_ZS_PATH)
    df_global['bias'] = df_global['Bias_Lean'].apply(normalize_bias)
    df_global['condition'] = 'Global_ZS'

    df_region = pd.read_csv(REGION_ZS_PATH)
    df_region['bias'] = df_region['Predicted_Bias'].apply(normalize_bias)
    df_region['condition'] = 'Regional_ZS'

    df_cot = pd.read_csv(COT_PATH)
    df_cot['bias'] = df_cot['selection'].apply(normalize_bias)
    df_cot['condition'] = 'CoT'

    return df_global, df_region, df_cot

def perform_test(name, df1, df2, join_cols, filter1=None, filter2=None, model_wise=False):
    print(f"\n{'='*40}")
    print(f"Comparison: {name}")
    print(f"{'='*40}")
    
    d1 = df1.copy()
    if filter1:
        for k, v in filter1.items():
            d1 = d1[d1[k] == v]
    
    d2 = df2.copy()
    if filter2:
        for k, v in filter2.items():
            d2 = d2[d2[k] == v]

    # Standardize column names for join
    def standardize_cols(df):
        cols = {}
        for c in df.columns:
            if c.lower() == 'question_id': cols[c] = 'qid'
            if c.lower() in ['model', 'model_name']: cols[c] = 'model'
            if c.lower() == 'language': cols[c] = 'lang'
            if c.lower() == 'region': cols[c] = 'region'
        return df.rename(columns=cols)

    d1 = standardize_cols(d1)
    d2 = standardize_cols(d2)

    # Convert join cols to standard names
    std_join_cols = []
    for c in join_cols:
        if c.lower() == 'question_id': std_join_cols.append('qid')
        elif c.lower() in ['model', 'model_name']: std_join_cols.append('model')
        elif c.lower() == 'language': std_join_cols.append('lang')
        elif c.lower() == 'region': std_join_cols.append('region')
    
    # Merge
    merged = pd.merge(d1, d2, on=std_join_cols, suffixes=('_1', '_2'))
    print(f"Matched {len(merged)} pairs (Aggregate).")
    
    if len(merged) == 0:
        print("No matching pairs found.")
        return

    def run_stat_test(data, title_suffix=""):
        target = 'Shia'
        data['res1'] = data['bias_1'] == target
        data['res2'] = data['bias_2'] == target

        # Contingency Table
        a = len(data[(data['res1'] == True) & (data['res2'] == True)])
        b = len(data[(data['res1'] == True) & (data['res2'] == False)])
        c = len(data[(data['res1'] == False) & (data['res2'] == True)])
        d = len(data[(data['res1'] == False) & (data['res2'] == False)])

        print(f"\n--- {title_suffix} (N={len(data)}) ---")
        print(f"Contingency Table (Target: {target}):\n [[YY={a} (Agreed Shia), YN={b} (Shift to Sunni/Neutral)],\n  [NY={c} (Shift to Shia), NN={d}]]")
        
        if b + c > 0:
            result = mcnemar([[a, b], [c, d]], exact=True)
            print(f"McNemar's p-value: {result.pvalue:.5f}")
            if result.pvalue < 0.05:
                print("**Result: Significant Difference!**")
                if b > c:
                    print(f"Direction: Significant shift AWAY from {target} bias")
                else:
                    print(f"Direction: Significant shift TOWARDS {target} bias")
            else:
                 print("Result: No Significant Difference.")
        else:
            print("No discordance (b+c=0), p-value=1.0")

    # Run Aggregate
    run_stat_test(merged, "Aggregate")

    # Run Model-wise
    if model_wise and 'model' in merged.columns:
        print(f"\n{'-'*20}\nModel-wise Breakdown\n{'-'*20}")
        models = merged['model'].unique()
        for m in sorted(models):
            m_data = merged[merged['model'] == m].copy()
            run_stat_test(m_data, f"Model: {m}")

def main():
    df_global, df_region, df_cot = load_and_preprocess()

    # 1. English Zero-Shot (Global) vs English CoT (Global)
    perform_test(
        "English Global Zero-Shot vs English Global CoT",
        df_global, df_cot,
        join_cols=['Question_ID', 'Model'],
        filter1={'Language': 'English'},
        filter2={'language': 'English', 'region': 'Global'},
        model_wise=True
    )

    # 2. Hindi Zero-Shot (Global) vs Hindi CoT (Global)
    perform_test(
        "Hindi Global Zero-Shot vs Hindi Global CoT",
        df_global, df_cot,
        join_cols=['Question_ID', 'Model'],
        filter1={'Language': 'Hindi'},
        filter2={'language': 'Hindi', 'region': 'Global'},
        model_wise=True
    )

    # 3. Regional English Zero-Shot vs Regional English CoT
    perform_test(
        "Regional English Zero-Shot vs Regional English CoT",
        df_region, df_cot,
        join_cols=['Question_ID', 'Model_Name', 'Region'],
        filter1={'Language': 'English'},
        filter2={'language': 'English'},
        model_wise=True 
    )

if __name__ == "__main__":
    main()
