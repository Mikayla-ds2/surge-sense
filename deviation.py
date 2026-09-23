"""
goal of this function is to answer either given a patient belongs to feature-group F, is their rate of outcome O
different from what you'd expect if F and O were unrelated or given a patient belongs to outcome-group O,
is the composition makeup of feature F different from what you'd expect if F and o were unrelated?

the contigency table is normalized by feature (all the feature's rows equal 100%); the baseline
is the outcome's overall distribution (the outcome being the same despite the feature points to
independence)
"""

from dataclasses import dataclass

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, norm

@dataclass
class DeviationResult:
    """
    Bundled the output on the function so when called, it returns a named object and not a tuple 
    that easily breaks    
    """
    
    tidy: pd.DataFrame # one row per pair (feature, outcome)
    table: pd.DataFrame # wide crosstab, in %    
    baseline: pd.Series
    chi2_statistic: float # global test: is there any association at all?
    chi2_pvalue: float
    dof: int # degrees of freedom; tells chi2 distribution shape chi2_statistic is compared against
    n_cells: int # how many cells (feature, outcome) were tested
    bonferroni_alpha: float # alpha, corrected for testing n_cells at once 
    z_critical: float # |adjusted residual| beyond this is significant
    
    def summary(self):
        return pd.Series({
            'Chi-squared statistic': self.chi2_statistic,
            'p-value': self.chi2_pvalue,
            'Degrees of Freedom': self.dof,
            'Number of cells': self.n_cells,
            'Bonferroni alpha': self.bonferroni_alpha,
            'Critical z-score': self.z_critical
            }, name='Deviation test')
    
def test_deviation(
    data: pd.DataFrame, # cleaned dataset
    feature: str, # either group I'm comparing or measured composition w/in outcome
    outcome: str, # measuing rates or makeup container
    alpha: float = 0.05, # desired significance level before correction either 5% or 1% 
                        # signifance is wrong when called/ percent of time residual falls out of 
                        # range by pure chance if null hypothesis is actually true
) -> DeviationResult:
    """
    Compute the rate or composition, compare them the
    overall baseline, and flag which deviations are statistically significant (Bonferroni corrected)
    vs. just noise from a small sample.
    
    Remember low_reliability = expected count < 5.
    """
    
    subset = data[[feature, outcome]].copy()
    
    observed = pd.crosstab(subset[feature], subset[outcome]) # raw counts crosstab for chi2
    
    table = pd.crosstab(
        subset[feature], subset[outcome], normalize='index'
    ).mul(100)
    
    baseline = (
        subset[outcome].value_counts(normalize=True).mul(100).
        reindex(table.columns)
    )
    
    diff_pp = table.sub(baseline, axis='columns') # the intuitive, communicable measure
    # however cannot be used bc 3pp means different things for a group of 50 vs 1000
    
    chi2_stat, p_value, dof, expected = chi2_contingency(observed=observed)
    expected_df = pd.DataFrame(
        expected, index=observed.index, columns=observed.columns
    ) # p-value for association at all; expected frequency table for later
    
    grand_total = observed.values.sum()
    row_prop = observed.sum(axis=1) / grand_total # each feature category's share of all rows 
    col_prop = observed.sum(axis=0) / grand_total # each outcome category's share of all rows
    
    variance_term = expected_df.mul(1 - row_prop, axis=0).mul(1 - col_prop, axis=1)
    adjusted_residual = (observed - expected_df) / np.sqrt(variance_term)
    
    n_cells = observed.shape[0] * observed.shape[1]
    bonferroni_alpha = alpha / n_cells
    z_critical = norm.ppf(1 - bonferroni_alpha / 2)
    
    tidy = pd.concat( # long format table
        {
            'observed_count': observed.stack(), # .stack() turns wide table -> Series aligned
            'expected_count': expected_df.stack(),
            'pct': table.stack(),
            'diff_pp': diff_pp.stack(),
            'adjusted_residual': adjusted_residual.stack(),
        },
        axis=1,
    )
    tidy.index = tidy.index.set_names([feature, outcome])
    tidy = tidy.reset_index()
    
    # baseline only varies by outcome category so .map() works fine
    tidy['baseline_pct'] = tidy[outcome].map(baseline)
    
    tidy['low_reliability'] = tidy['expected_count'] < 5
    tidy['significant'] = tidy['adjusted_residual'].abs() > z_critical
    
    # keep full precision in table; round for visualization
    return DeviationResult(
        tidy=tidy,
        table=table,
        baseline=baseline,
        chi2_statistic=chi2_stat,
        chi2_pvalue=p_value,
        dof=dof,
        n_cells=n_cells,
        bonferroni_alpha=bonferroni_alpha,
        z_critical=z_critical
    )