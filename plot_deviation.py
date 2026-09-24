"""
visualization file for deviation:
color encodes direction (above/below baseline)
opacity encodes confidence (is this finding trustworthy?)
    - cell earns full opacity when cell is significant (adjusted residual passes 
    bonferroni threshold; effect probably isn't chance) & not low-reliability (
    expected count is >= 5
    )
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# come back to how exactly the functions get run, tested, and then used in jupyter

def _confidence_alpha(
    adjusted_residual,
    significant,
    low_reliability,
    z_critical,
    min_alpha=0.25, # opacity for anything that fails both tests; muted not hidden
    max_alpha=1.0, # opacity ceiling for most confident findings
    saturate_at=3.0, # keeps scale meaninfgul; prevents extreme residuals from
                    # washing out significant but not extreme findings
):

    if low_reliability or not significant:
        return min_alpha # fails either check -> alpha floor
    
    strength = abs(adjusted_residual) / z_critical
    scaled = np.clip((strength - 1.0) / (saturate_at - 1.0), 0.0, 1.0)
    return min_alpha + scaled * (max_alpha - min_alpha)

def plot_diverging_plot(
    result, # DeviationResult
    feature, 
    outcome,
    ncols=2, # panels per row
    panel_height=4.5,
    above_color='#9D9368',
    below_color='#743015',
):
    """
    facet diverging bar for each feature category, bars = pp gap from outcome's baseline,
    opacity = statistical confidence
    """
    
    tidy = result.tidy
    categories = tidy[feature].unique()
    nrows = int(np.ceil(len(categories) / ncols))
    
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(14, panel_height * nrows), sharex=True,
    )
    axes = np.atleast_1d(axes).flatten()
    
    x_limit = tidy['diff_pp'].abs().max() * 1.15 # shared x-limits so all bar lengths mean
                                                # the same thing
    for ax, category in zip(axes, categories):
        subset = tidy[tidy[feature] == category].sort_values('diff_pp')
        
        colors = np.where(subset['diff_pp'] >= 0, above_color, below_color)
        alphas = [
            _confidence_alpha(
                row.adjusted_residual,
                row.significant,
                row.low_reliability,
                result.z_critical,
            ) for row in subset.itertuples()
        ]                                   
        bars = ax.barh(subset[outcome], subset['diff_pp'], color=colors)
        for bar, alpha in zip(bars, alphas):
            bar.set_alpha(alpha)
        
        ax.axvline(0, color='black', linewidth=1)
        ax.set_xlim(-x_limit, x_limit)
        ax.set_title(str(category), fontsize=11)
        ax.set_xlabel("Difference from baseline (pp)")
        
    for ax in axes[len(categories):]:
        ax.set_visible(False) # hides any unused plots
        
    fig.suptitle(
        f"{feature} vs. {outcome}: deviation from baseline\n"
        "(faded = not significant or not reliable; solid = confident finding)",
        y=1.02,
    )
    plt.tight_layout()
    return fig

def plot_confidence_heatmap(result, feature, outcome, figsize=(13, 6)):
    """
    heatmap of pp deviation w/ non-significant or low-reliability cells muted
    result : RateDeviationResult
    """
    tidy = result.tidy
    diff_wide = tidy.pivot(index=feature, columns=outcome, values='diff_pp') # wide formatted
    # table of deviation values
    
    tidy_w_alpha = tidy.assign(
        _alpha=[
            _confidence_alpha(
                row.adjusted_residual,
                row.significant,
                row.low_reliability,
                result.z_critical, 
            )
            for row in tidy.itertuples()
        ]
    )                                   
    alpha_wide = tidy_w_alpha.pivot(
        index=feature, columns=outcome, values="_alpha"
    ).loc[diff_wide.index, diff_wide.columns]
    
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    palette = ['#EAD3A9','#C4B389','#B7966A',
            '#9D9368','#A05135','#84592B',
            '#733F28','#743015','#462D1B']
    discrete = ListedColormap(palette)
    continuous = LinearSegmentedColormap.from_list(palette)
 
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        diff_wide,
        annot=diff_wide.round(1),
        fmt=".1f",
        cmap=continuous,
        center=0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Difference from baseline (pp)"},
    )      
    
    # adding semi-transparent white rectangle for cells that didn't clear both significance and
    # low-reliability tests
    min_alpha_floor = 0.25
    for i, feature_cat in enumerate(diff_wide.index):
        for j, outcome_cat in enumerate(diff_wide.columns):
            if alpha_wide.loc[feature_cat, outcome_cat] <= min_alpha_floor:
                ax.add_patch(
                    plt.Rectangle(
                        (j, i), 1, 1,
                        fill=True, color="white", alpha=0.55,
                        zorder=3, linewidth=0,
                    )
                )
 
    ax.set_title(
        "Unmuted cells = statistically significant AND reliable (expected count >= 5)"
    )
    plt.tight_layout()
    return fig                       
