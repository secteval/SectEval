import pandas as pd
import os

files = [
    "/home/vivekp/isib/bias_results.csv",
    "/home/vivekp/isib/cot_bias_results.csv",
    "/home/vivekp/isib/results_full.csv"
]

for f in files:
    if os.path.exists(f):
        print(f"--- Checking {f} ---")
        try:
            df = pd.read_csv(f, on_bad_lines='skip') # Skip bad lines to avoid basic parsing errors
            print("Columns:", df.columns.tolist())
            if 'region' in df.columns:
                print("Unique Regions:", df['region'].unique())
            if 'Identity' in df.columns:
                 print("Unique Identities:", df['Identity'].unique())
            if 'template_id' in df.columns:
                print("Unique Template IDs:", df['template_id'].unique())
            if 'language' in df.columns:
                print("Unique Languages:", df['language'].unique())
            if 'Language' in df.columns:
                print("Unique Languages (Cap):", df['Language'].unique())
            print("Row count:", len(df))
        except Exception as e:
            print(f"Error reading {f}: {e}")
    else:
        print(f"File not found: {f}")
