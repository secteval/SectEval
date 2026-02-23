import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Settings
INPUT_CSV = "../results/shia_to_sunni_flips.csv"
OUTPUT_IMG = "../results/shia_to_sunni_shift_chart.png"

# Professional Color Palette (from generate_paper_plots.py)
COLOR_SHIA = '#4E79A7'    # Blue
COLOR_SUNNI = '#F28E2B'   # Orange
COLOR_NEUTRAL = '#E5E5E5' # Light Grey

# Configuration for Research Paper Quality
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 600, # Requested 600 DPI
})

def plot_chart():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found. Run analysis first.")
        return

    df = pd.read_csv(INPUT_CSV)
    
    # Count flips per model
    counts = df['Model'].value_counts().reset_index()
    counts.columns = ['Model', 'Count']
    
    # Sort for plotting (Highest count first)
    counts = counts.sort_values('Count', ascending=False)
    
    # Plot - Vertical Bars
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid") 
    # Re-apply font params after set_style might overwrite them
    plt.rcParams.update({"font.family": "serif"})

    # Create vertical bar chart
    # using COLOR_SUNNI since the shift is TOWARDS Sunni
    ax = sns.barplot(
        data=counts, 
        x='Model', 
        y='Count', 
        color=COLOR_SUNNI
    )
    
    # Add labels
    plt.title('Frequency of Bias Shift: Shia (English) → Sunni (Hindi)', fontsize=18, pad=20, fontweight='bold')
    plt.ylabel('Number of Questions Flipped', fontsize=14)
    plt.xlabel('Model', fontsize=14)
    
    # Rotate x labels for readability
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of bars
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height)}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', 
                    fontsize=12, color='black', xytext=(0, 5),
                    textcoords='offset points')
        
    sns.despine()
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=600, bbox_inches='tight')
    print(f"Chart saved to {OUTPUT_IMG}")

if __name__ == "__main__":
    plot_chart()
