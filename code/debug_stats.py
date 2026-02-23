import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

GLOBAL_ZS_PATH = "../data/results_full_global.csv"

def normalize_bias(value):
    if pd.isna(value): return None
    val = str(value).strip().lower()
    if val in ['a', 'shia', 'option a']: return 'Shia'
    if val in ['b', 'sunni', 'option b']: return 'Sunni'
    return 'Neutral/Other'

def verify_model(df, model_name):
    print(f"\n--- Verifying {model_name} ---")
    df_m = df[df['Model'] == model_name].copy()
    
    df_eng = df_m[df_m['Language'] == 'English'][['Question_ID', 'Bias_Lean']].rename(columns={'Bias_Lean': 'Bias_Eng'})
    df_hin = df_m[df_m['Language'] == 'Hindi'][['Question_ID', 'Bias_Lean']].rename(columns={'Bias_Lean': 'Bias_Hin'})
    
    merged = pd.merge(df_eng, df_hin, on='Question_ID')
    
    merged['Bias_Eng'] = merged['Bias_Eng'].apply(normalize_bias)
    merged['Bias_Hin'] = merged['Bias_Hin'].apply(normalize_bias)
    
    # Detailed Transition Counts
    transitions = merged.groupby(['Bias_Eng', 'Bias_Hin']).size().reset_index(name='Count')
    print("Transitions:")
    print(transitions)
    
    # McNemar "Is Shia?"
    is_shia_eng = merged['Bias_Eng'] == 'Shia'
    is_shia_hin = merged['Bias_Hin'] == 'Shia'
    
    YY = ((is_shia_eng) & (is_shia_hin)).sum()
    YN = ((is_shia_eng) & (~is_shia_hin)).sum() # Shia -> Not Shia
    NY = ((~is_shia_eng) & (is_shia_hin)).sum() # Not Shia -> Shia
    NN = ((~is_shia_eng) & (~is_shia_hin)).sum()
    
    print(f"\nMcNemar 'Is Shia' Table:")
    print(f"[[YY={YY}, YN={YN}],")
    print(f" [NY={NY}, NN={NN}]]")
    
    if YN + NY > 0:
        res = mcnemar([[YY, YN], [NY, NN]], exact=True)
        print(f"p-value: {res.pvalue}")
    else:
        print("p-value: 1.0")

def main():
    df = pd.read_csv(GLOBAL_ZS_PATH)
    verify_model(df, 'Llama3.1-8B')
    verify_model(df, 'Claude-3.5-Sonnet')
    verify_model(df, 'GPT-4o')

if __name__ == "__main__":
    main()
