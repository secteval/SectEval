
from statsmodels.stats.contingency_tables import mcnemar
import scipy.stats as stats

# Data from user snippet
# [[YY=2499, YN=793],
#  [NY=1154, NN=3581]]

table = [[2499, 793],
         [1154, 3581]]

print(f"Testing table: {table}")

# 1. Using statsmodels (Exact) which was used in the code
result = mcnemar(table, exact=True)
print(f"Statsmodels Exact p-value: {result.pvalue}")
print(f"formatted: {result.pvalue:.5f}")

# 2. Using statsmodels (Chi-Square) for comparison
result_chi2 = mcnemar(table, exact=False, correction=True)
print(f"Statsmodels Chi2 p-value: {result_chi2.pvalue}")

# 3. Manual Binomial Calculation
# P(X <= 793) for B(1947, 0.5) * 2 (two-sided)
n = 793 + 1154
k = 793
p_value_binom = stats.binom.cdf(k, n, 0.5) * 2
print(f"Manual Binomial p-value: {p_value_binom}")
