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
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
from pywaffle import Waffle
import textwrap

# come back to how exactly the functions get run, test, and then used in jupyter

def _confidence_alpha(
    adjusted_residual,
    significant,
    low_reliability,
    z_critical,
    min_alpha=0.25, # opacity for findings that fail either confidence check
    significant_min_alpha=0.55, # clear floor for significant, reliable findings
    max_alpha=1.0, # opacity ceiling for most confident findings
    saturate_at=3.0, # keeps scale meaninfgul; prevents extreme residuals from
                    # washing out significant but not extreme findings
):

    if low_reliability or not significant:
        return min_alpha # fails either check -> alpha floor
    
    strength = abs(adjusted_residual) / z_critical
    scaled = np.clip((strength - 1.0) / (saturate_at - 1.0), 0.0, 1.0)
    return significant_min_alpha + scaled * (max_alpha - significant_min_alpha)

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
    continuous = LinearSegmentedColormap.from_list('earth_sequential', palette)
    
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
    # Keep the heatmap's mute threshold aligned with _confidence_alpha's
    # opacity for cells that fail either confidence check.
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
                       
def plot_composition_waffle(
    result,
    feature,
    outcome,
    color_map=None,
    icons="circle-user",
    icon_size=16,
    rows=10,
    figsize=(7, 6),
):
    """
    One waffle chart per feature category (e.g. one per department), each
    summing to ~100%, squares colored by outcome category (e.g. day of
    week). Color here encodes CATEGORY IDENTITY, not direction/confidence
    -- this is the plain "what does the makeup actually look like" view,
    meant to be read alongside plot_composition_treemap's significance-
    encoded version of the same composition.
 
    result.table is expected to have `feature` as rows (each summing to
    ~100%) and `outcome` as columns -- i.e. result should come from a
    deviation() call with THIS `feature` as the function's feature arg.
 
    Returns
    -------
    list of (category, fig) tuples -- one entry per feature category.
    Unlike plot_diverging_plot / plot_confidence_heatmap, this does NOT
    return a single figure, since each category gets its own waffle.
    """
    plot_data = result.table  # feature as rows (index), outcome as columns
    categories = plot_data.index.tolist()
    outcome_categories = plot_data.columns.tolist()
 
    if color_map is None:
        palette = ['#EAD3A9', '#C4B389', '#B7966A',
                   '#9D9368', '#A05135', '#84592B',
                   '#733F28', '#743015', '#462D1B']
        color_map = {
            cat: palette[i % len(palette)]
            for i, cat in enumerate(outcome_categories)
        }
 
    figs = []
    for category in categories:
        series = plot_data.loc[category].sort_values(ascending=False)
        sorted_outcome_cats = series.index.tolist()
 
        fig = plt.figure(
            FigureClass=Waffle,
            values=series,
            colors=[color_map[c] for c in sorted_outcome_cats],
            labels=[f"{c} ({pct:.1f}%)" for c, pct in series.items()],
            icons=icons,
            icon_size=icon_size,
            rows=rows,
            rounding_rule="ceil",
            legend={
                "loc": "lower center",
                "bbox_to_anchor": (0.5, -0.3),
                "ncol": 3,
                "framealpha": 0,
            },
            figsize=figsize,
        )
        fig.suptitle(
            f"{outcome.replace('_', ' ').title()} Makeup \u2014 {category}",
            fontweight="bold",
            x=0.3,
            y=0.95,
            ha="center",
        )
        figs.append((category, fig))
 
    return figs
 
 
def plot_composition_treemap(
    result,
    feature,
    outcome,
    above_color='#9D9368',
    below_color='#743015',
    figsize=(10, 6),
):
    """
    One treemap per feature category, rectangles sized by composition
    percentage WITHIN that category, colored by direction (above/below
    baseline) and confidence -- same _confidence_alpha logic as the
    diverging bar and heatmap, just rendered as rectangle area instead
    of bar length or grid position.
 
    Returns
    -------
    list of (category, fig) tuples -- one entry per feature category,
    same shape as plot_composition_waffle's return.
    """
    tidy = result.tidy
    categories = tidy[feature].unique()
 
    figs = []
    for category in categories:
        subset = tidy[tidy[feature] == category]
 
        colors = []
        for row in subset.itertuples():
            base_color = above_color if row.diff_pp >= 0 else below_color
            alpha = _confidence_alpha(
                row.adjusted_residual,
                row.significant,
                row.low_reliability,
                result.z_critical,
            )
            colors.append(matplotlib.colors.to_rgba(base_color, alpha=alpha))


        fig, ax = plt.subplots(figsize=figsize)
        squarify.plot(
            sizes=subset['pct'],
            label=subset[outcome],
            color=colors,
            bar_kwargs={"linewidth": 1.1, "edgecolor": 'black'},
            text_kwargs={'fontsize': 9},
            ax=ax,
        )
        ax.axis('off')
        ax.set_title(
            f"{outcome.replace('_', ' ').title()} Makeup \u2014 {category}",
            fontweight='bold',
        )
        figs.append((category, fig))
 
    return figs
