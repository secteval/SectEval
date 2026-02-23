import pandas as pd
import os

# File Paths
GLOBAL_ZS_PATH = "../data/results_full_global.csv"
OUTPUT_CSV_PATH = "../results/shia_to_sunni_shifts.csv"

def normalize_bias(value):
    """Normalize bias values to Shia/Sunni/Neutral."""
    if pd.isna(value): return None
    val = str(value).strip().lower()
    if val in ['a', 'shia', 'option a']: return 'Shia'
    if val in ['b', 'sunni', 'option b']: return 'Sunni'
    return 'Neutral/Other'

def analyze_shift():
    if not os.path.exists(GLOBAL_ZS_PATH):
        print(f"Error: File not found at {GLOBAL_ZS_PATH}")
        return

    print(f"Loading {GLOBAL_ZS_PATH}...")
    df = pd.read_csv(GLOBAL_ZS_PATH)
    
    # Normalize Bias
    df['bias_norm'] = df['Bias_Lean'].apply(normalize_bias)
    
    # Filter for English and Hindi
    df_eng = df[df['Language'] == 'English'].copy()
    df_hin = df[df['Language'] == 'Hindi'].copy()
    
    # Rename for merge
    # Keep Question text from English side (assuming same ID same Question)
    df_eng = df_eng[['Model', 'Question_ID', 'Question', 'bias_norm']].rename(columns={'bias_norm': 'bias_eng'})
    df_hin = df_hin[['Model', 'Question_ID', 'bias_norm']].rename(columns={'bias_norm': 'bias_hin'})
    
    # Merge
    merged = pd.merge(df_eng, df_hin, on=['Model', 'Question_ID'], how='inner')
    
    # Identify Shifts
    # Shia (English) -> Sunni (Hindi)
    shift_mask = (merged['bias_eng'] == 'Shia') & (merged['bias_hin'] == 'Sunni')
    merged['is_shift'] = shift_mask
    
    # Filter for only shifted rows
    shifted_df = merged[merged['is_shift']].copy()
    
    # Select relevant columns for detailed report
    detailed_report = shifted_df[['Model', 'Question_ID', 'Question', 'bias_eng', 'bias_hin']]
    
    # Ensure results directory exists
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    
    # Save detailed report
    detailed_report.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Detailed results saved to: {OUTPUT_CSV_PATH}")
    
    # Calculate stats
    shift_counts = shifted_df.groupby('Model').size().reset_index(name='count')
    total_counts = merged.groupby('Model').size().reset_index(name='total')
    
    stats = pd.merge(shift_counts, total_counts, on='Model')
    stats['percentage'] = (stats['count'] / stats['total']) * 100
    
    print("\nModels shifting from Shia (English) -> Sunni (Hindi):")
    print(f"{'Model':<30} | {'Count':<5} | {'Total':<5} | {'%':<5}")
    print("-" * 60)
    
    for _, row in stats.sort_values('count', ascending=False).iterrows():
        print(f"{row['Model']:<30} | {row['count']:<5} | {row['total']:<5} | {row['percentage']:.1f}%")

if __name__ == "__main__":
    analyze_shift()
