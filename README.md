# SectEval: Evaluating the Latent Sectarian Preferences of Large Language Models

This repository contains the code and data for the paper "SectEval: Evaluating the Latent Sectarian Preferences of Large Language Models".

## Structure

- **code/**: Contains the Python scripts used for testing and analysis.
    - `run_cot_test.py`: Script to run Chain-of-Thought bias tests.
    - `run_region_test.py`: Script to run regional identity bias tests.
    - `perform_mcnemar.py`: Script to perform McNemar's statistical test on the results.
    - `inspect_data.py`: Utility script to inspect data files.
- **data/**: Contains the datasets and results.
    - `questions_full.csv`: The full set of 88 questions in English and Hindi.
    - `cot_bias_results.csv`: Results from the Chain-of-Thought experiments.
    - `bias_results_regional.csv`: Results from the Regional Identity experiments (Zero-shot).
    - `results_full_global.csv`: Results from the Global Zero-shot experiments.
    - `cot_bias_prompts.json`: Prompts used for CoT testing.
- **results/**: Contains the output of the analysis.
    - `mcnemar_results.txt`: Statistical analysis results comparing bias across conditions.

## Usage

To run the statistical analysis:

```bash
cd code
python3 perform_mcnemar.py
```

This will generate the McNemar's test results, comparing model behavior across English/Hindi and Global/Regional contexts.

## Citation

If you use this dataset or code, please cite our paper:

Maheshwari, A., Gajkeshwar, A., Sharma, K., & Patel, V. (2025). SectEval: Evaluating the Latent Sectarian Preferences of Large Language Models.
