import warnings

import fastcluster
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from adjustText import adjust_text
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .consts import COLORS_INDEL83, COLORS_SBS96, INDEL_TYPES_83, SBS_TYPES_96
from .utils import match_to_catalog, value_checker


def set_salamander_style():
    sns.set_context("notebook")
    sns.set_style("ticks")
    params = {
        "axes.edgecolor": "black",
        "axes.labelsize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 16,
        "errorbar.capsize": 3,
        "font.family": "DejaVu Sans",
        "legend.fontsize": 12,
        "lines.markersize": 8,
        "pdf.fonttype": 42,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }
    mpl.rcParams.update(params)


def history_plot(values, conv_test_freq, min_iteration=0, ax=None, **kwargs):
    n_values = len(values)
    ns_iteration = np.arange(
        conv_test_freq, n_values * conv_test_freq + 1, conv_test_freq
    )

    if min_iteration > ns_iteration[-1]:
        raise ValueError(
            "The smallest iteration number shown in the history plot "
            "cannot be larger than the total number of iterations."
        )
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
        ax.set(xlabel="n_iteration", ylabel="objective function value")

    min_index = next(
        idx
        for idx, n_iteration in enumerate(ns_iteration)
        if n_iteration >= min_iteration
    )
    ax.plot(ns_iteration[min_index:], values[min_index:], **kwargs)
    return ax


def _annotate_plot(
    ax,
    data,
    annotations,
    ha="left",
    fontsize="medium",
    color="black",
    adjust_annotations=True,
    adjust_kwargs=None,
    **kwargs,
):
    for data_point, annotation in zip(data, annotations):
        ax.text(
            data_point[0],
            data_point[1],
            annotation,
            ha=ha,
            fontsize=fontsize,
            color=color,
            **kwargs,
        )
    if adjust_annotations:
        if adjust_kwargs is None:
            adjust_kwargs = {} if adjust_kwargs is None else adjust_kwargs.copy()

        annotations = [
            child for child in ax.get_children() if isinstance(child, mpl.text.Text)
        ]
        annotations = [
            annotation for annotation in annotations if annotation.get_text()
        ]
        adjust_text(annotations, **adjust_kwargs)


def _scatter_1d(data: np.ndarray, ax=None, **kwargs):
    if data.ndim != 1:
        raise ValueError(f"The datapoints of {data} (rows) have to be one-dimensional.")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 1))

    ax.spines[["left", "bottom"]].set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.axhline(y=0, color="black", zorder=1)
    data_2d = np.vstack([data, np.zeros_like(data)]).T
    sns.scatterplot(x=data_2d[:, 0], y=data_2d[:, 1], ax=ax, zorder=2, **kwargs)
    return data_2d, ax


def _scatter_2d(data, ax=None, **kwargs):
    """
    The rows (!) of 'data' are assumed to be the data points.
    """
    if data.shape[1] != 2:
        raise ValueError(f"The datapoints of {data} (rows) have to be two-dimensional.")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    ax.set(xlabel="x", ylabel="y")
    sns.scatterplot(x=data[:, 0], y=data[:, 1], ax=ax, **kwargs)
    return data, ax


def _pca_2d(data, ax=None, **kwargs):
    """
    The rows (!) of 'data' are assumed to be the data points.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    data_2d = PCA(n_components=2).fit_transform(data)
    ax.set(xlabel="PC1", ylabel="PC2")
    sns.scatterplot(x=data_2d[:, 0], y=data_2d[:, 1], ax=ax, **kwargs)
    return data_2d, ax


def _tsne_2d(data, perplexity=30, ax=None, **kwargs):
    """
    The rows (!) of 'data' are assumed to be the single data points.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perplexity = min(perplexity, len(data) - 1)
        data_2d = TSNE(perplexity=perplexity).fit_transform(data)

    ax.set(xlabel="t-SNE1", xticks=[], ylabel="t-SNE2", yticks=[])
    sns.scatterplot(x=data_2d[:, 0], y=data_2d[:, 1], ax=ax, **kwargs)
    return data_2d, ax


def _umap_2d(data, n_neighbors=15, min_dist=0.1, ax=None, **kwargs):
    """
    The rows (!) of 'data' are assumed to be the single data points.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    n_neighbors = min(n_neighbors, len(data) - 1)
    data_2d = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist).fit_transform(data)

    ax.set(xlabel="UMAP1", xticks=[], ylabel="UMAP2", yticks=[])
    sns.scatterplot(x=data_2d[:, 0], y=data_2d[:, 1], ax=ax, **kwargs)
    return data_2d, ax


def embeddings_plot(
    data: np.ndarray,
    method="umap",
    normalize=False,
    annotations=None,
    annotation_kwargs=None,
    adjust_annotations=True,
    adjust_kwargs=None,
    ax=None,
    **kwargs,
):
    """
    Plot a dimensionality reduction of high-dimensional data.
    If the dimension of the data is one or two, it is plotted
    directly, ignoring the chosen method.

    Parameters
    ----------
    data : np.ndarray
        The rows (!) are assumed to be the data points.

    method : str, default='umap'
        Either 'tsne', 'pca' or 'umap'. The respective dimensionality reduction
        is applied to plot the data in 2D. If the dimensionality of the data is
        less or equal to two, the points are plotted directly.

    normalize : bool, default=False
        If True, the data points are normalized to have a Euclidean norm of
        one before applying the dimensionality reduction.

    annotations : list[str], default=None
        Annotations of the projected data points. The length of the list has to match
        the number of data points.

    annotation_kwargs : dict, default=None
        keyword arguments to pass to matplotlibs plt.txt()

    adjust_annotations : bool, default=True
        If True, the annotations are adjusted to prevent them from overlapping.

    adjust_kwargs : dict, default=None
        keyword arguments to pass to adjustText's adjust_text.
        This is for example useful when arrows connecting the projected
        data points and the adjusted positions of the annotations
        should be added.

    ax : matplotlib.axes.Axes, default=None
        Pre-existing axes for the plot. Otherwise, an axes is created.

    **kwargs :
        keyword arguments to pass to seaborn's scatterplot

    Returns
    -------
    ax : matplotlib.axes.Axes
        The matplotlib axes containing the plot.
    """
    value_checker("method", method, ["pca", "tsne", "umap"])

    if normalize:
        data /= np.sqrt(np.sum(data**2, axis=1))[:, np.newaxis]

    n_dimensions = data.shape[1]

    if n_dimensions in [1, 2]:
        warnings.warn(
            f"The dimension of the data points is {n_dimensions}. "
            f"The method argument '{method}' will be ignored "
            "and the embeddings are plotted directly.",
            UserWarning,
        )

    if n_dimensions == 1:
        data_2d, ax = _scatter_1d(data[:, 0], ax=ax, **kwargs)
    elif n_dimensions == 2:
        data_2d, ax = _scatter_2d(data, ax=ax, **kwargs)
    elif method == "tsne":
        data_2d, ax = _tsne_2d(data, ax=ax, **kwargs)
    elif method == "pca":
        data_2d, ax = _pca_2d(data, ax=ax, **kwargs)
    else:
        data_2d, ax = _umap_2d(data, ax=ax, **kwargs)

    if annotations is not None:
        annotation_kwargs = (
            {} if annotation_kwargs is None else annotation_kwargs.copy()
        )
        _annotate_plot(
            ax,
            data_2d,
            annotations,
            adjust_annotations=adjust_annotations,
            adjust_kwargs=adjust_kwargs,
            **annotation_kwargs,
        )

    return ax


def corr_plot(
    corr: pd.DataFrame, figsize=(6, 6), cmap="vlag", annot=True, fmt=".2f", **kwargs
):
    linkage = hierarchy.linkage(corr)
    clustergrid = sns.clustermap(
        corr,
        row_linkage=linkage,
        figsize=figsize,
        vmin=-1,
        vmax=1,
        cmap=cmap,
        annot=annot,
        fmt=fmt,
        **kwargs,
    )

    return clustergrid


def _get_colors_signature_plot(mutation_types, colors=None):
    """
    Given the mutation types and the colors argument of sigplot_bar, return the
    final colors used in the signature bar chart.
    """
    n_features = len(mutation_types)

    if colors == "SBS96" or (
        n_features == 96 and all(mutation_types == SBS_TYPES_96) and colors is None
    ):
        if n_features != 96:
            raise ValueError(
                "The standard SBS colors can only be used "
                "when the signatures have 96 features."
            )
        colors = COLORS_SBS96

    elif colors == "Indel83" or (
        n_features == 83 and all(mutation_types == INDEL_TYPES_83) and colors is None
    ):
        if n_features != 83:
            raise ValueError(
                "The standard Indel colors can only be used "
                "when the signatures have 83 features."
            )
        colors = COLORS_INDEL83

    elif type(colors) in [str, tuple]:
        colors = n_features * [colors]

    elif type(colors) is list:
        if len(colors) != n_features:
            raise ValueError(
                f"The list of colors must be of length n_features={n_features}."
            )

    else:
        colors = n_features * ["gray"]

    return colors


def _signature_plot(
    signature, colors=None, annotate_mutation_types=False, ax=None, **kwargs
):
    """
    Inputs:
    -------
    signature: pd.Signature
        Signature with mutation types and name.

    colors: str, tuple or list
        Can be set to 'SBS96' or 'Indel83' to use the standard bar colors
        for these mutation types.
        Otherwise, when a single string or tuple is provided,
        all bars will have the same color. Alternatively,
        a list can be used to specifiy the color of each bar individually.

    ax:
        A single matplotlib Axes in which to draw the plot.

    kwargs: dict
        Any keyword arguments to be passed to matplotlibs ax.bar
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 1))

    signature_normalized = signature / signature.sum(axis=0)
    mutation_types = signature.index
    colors = _get_colors_signature_plot(mutation_types, colors)

    ax.set_title(signature_normalized.columns[0])
    ax.spines["left"].set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_xlim((-1, len(mutation_types)))

    ax.bar(
        mutation_types,
        signature_normalized.iloc[:, 0],
        linewidth=0,
        color=colors,
        **kwargs,
    )

    if annotate_mutation_types:
        ax.set_xticks(mutation_types)
        ax.set_xticklabels(
            mutation_types, family="monospace", fontsize=4, ha="center", rotation=90
        )

    else:
        ax.set_xticks([])

    return ax


def signature_plot(
    signature,
    catalog=None,
    colors=None,
    annotate_mutation_types=False,
    ax=None,
    **kwargs,
):
    """
    Inputs:
    -------
    signature: pd.Signature
        Signature with mutation types and name.

    catalog: pd.DataFrame
        If a catalog is provided, the single best matching catalog signature
        will also be plotted.

    colors: str, tuple or list
        Can be set to 'SBS96' or 'Indel83' to use the standard bar colors
        for these mutation types.
        Otherwise, when a single string or tuple is provided,
        all bars will have the same color. Alternatively,
        a list can be used to specifiy the color of each bar individually.

    ax:
        Axes in which to draw the plot. A single Axes if catalog is None;
        two Axes if a catalog is given.

    kwargs: dict
        Any keyword arguments to be passed to matplotlibs ax.bar
    """
    if catalog is None:
        if ax is None:
            _, ax = plt.subplots(figsize=(4, 1))

        signatures = [signature]
        axes = [ax]

    else:
        if ax is None:
            _, ax = plt.subplots(1, 2, figsize=(8, 1))

        matched_signature = match_to_catalog(signature, catalog, metric="cosine")
        signatures = [signature, matched_signature]
        axes = ax

    for sig, axis in zip(signatures, axes):
        _signature_plot(
            sig,
            colors=colors,
            annotate_mutation_types=annotate_mutation_types,
            ax=axis,
            **kwargs,
        )

    if catalog is None:
        return axes[0]

    return axes


def signatures_plot(
    signatures,
    catalog=None,
    colors=None,
    annotate_mutation_types=False,
    axes=None,
    **kwargs,
):
    """
    Inputs:
    -------
    signatures : pd.DataFrame
        Named signatures of shape (n_features, n_signatures)

    catalog: pd.DataFrame
        If a catalog is provided, the best matching catalog signatures
        will also be plotted.

    axes : list
        Axes in which to draw the plot. Multiple Axes if more than one signature
        is provided or a catalog is given. Otherwise a single axis.
        When a catalog is provided, axes is expected to be of shape (n_signatures, 2).
    """
    n_signatures = signatures.shape[1]

    if n_signatures == 1:
        ax = signature_plot(
            signatures,
            catalog=catalog,
            colors=colors,
            annotate_mutation_types=annotate_mutation_types,
            ax=axes,
            **kwargs,
        )
        return ax

    if axes is None:
        if catalog is None:
            _, axes = plt.subplots(n_signatures, 1, figsize=(4, n_signatures))
        else:
            _, axes = plt.subplots(n_signatures, 2, figsize=(8, n_signatures))

    for ax, signature in zip(axes, signatures):
        signature_plot(
            signatures[[signature]],
            catalog=catalog,
            colors=colors,
            annotate_mutation_types=annotate_mutation_types,
            ax=ax,
            **kwargs,
        )
    plt.tight_layout()

    return axes


def _get_sample_order(exposures: pd.DataFrame, normalize=True):
    """
    Compute the aesthetically most pleasing order of the samples
    for a stacked bar chart of the exposures.

    Parameters
    ----------
    exposures : pd.DataFrame of shape (n_signatures, n_samples)
        The named exposure matrix

    normalize : bool, default=True
        If True, the exposures are normalized before computing the
        hierarchical clustering.

    Returns
    -------
    sample_order : np.ndarray
        The ordered sample names
    """
    if normalize:
        # not in-place
        exposures = exposures / exposures.sum(axis=0)

    d = pdist(exposures.T)
    linkage = fastcluster.linkage(d)
    # get the optimal sample order that is consistent
    # with the hierarchical clustering linkage
    sample_order = hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(linkage, d))
    sample_order = exposures.columns[sample_order].to_numpy()
    return sample_order


def _reorder_exposures(
    exposures: pd.DataFrame, sample_order=None, reorder_signatures=True
):
    """
    Reorder the samples with hierarchical clustering and
    reorder the signatures by their total relative exposure.

    Parameters
    ----------
    exposures : pd.DataFrame of shape (n_signatures, n_samples)
        The named exposure matrix

    sample_order : np.ndarray, default=None
        A predefined order of the samples as a list of sample names.
        If None, hierarchical clustering is used to compute the
        aesthetically most pleasing order.

    reorder_signatures : bool, default=True
        If True, the signatures will be reordered such that the
        total relative exposures of the signatures decrease from the bottom
        to the top signature in the stacked bar chart.

    Returns
    -------
    exposures_reordered : pd.DataFrame of shape (n_signatures, n_samples)
        The reorderd named exposure matrix
    """
    if sample_order is None:
        sample_order = _get_sample_order(exposures)

    exposures_reordered = exposures[sample_order]

    # order the signatures by their total relative exposure
    if reorder_signatures:
        exposures_normalized = exposures_reordered / exposures_reordered.sum(axis=0)
        signature_order = (
            exposures_normalized.sum(axis=1).sort_values(ascending=False).index
        )
        exposures_reordered = exposures_reordered.reindex(signature_order)

    return exposures_reordered


def exposures_plot(
    exposures: pd.DataFrame,
    sample_order=None,
    reorder_signatures=True,
    annotate_samples=True,
    colors=None,
    ncol_legend=1,
    ax=None,
    **kwargs,
):
    """
    Visualize the exposures with a stacked bar chart.

    Parameter
    ---------
    exposures : pd.DataFrame of shape (n_signatures, n_samples)
        The named exposure matrix.

    sample_order : np.ndarray, default=None
        A predefined order of the samples as a list of sample names.
        If None, hierarchical clustering is used to compute the
        aesthetically most pleasing order.

    reorder_signatures : bool, default=True
        If True, the signatures will be reordered such that the
        total relative exposures of the signatures decrease from the bottom
        to the top signature in the stacked bar chart.

    annotate_samples : bool, default=True
        If True, the x-axis is annotated with the sample names.

    colors : list of length n_signatures, default=None
        Colors to pass to matplotlibs ax.bar, one per signature.

    n_col_legend : int, default=1
        The number of columns of the legend.

    ax : matplotlib.axes.Axes, default=None
        Pre-existing axes for the plot. Otherwise, create an axis internally.

    kwargs : dict
        Any keyword arguments to be passed to matplotlibs ax.bar.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The matplotlib axes containing the plot.
    """
    n_signatures, n_samples = exposures.shape
    # not in-place
    exposures = exposures / exposures.sum(axis=0)
    exposures = _reorder_exposures(
        exposures, sample_order=sample_order, reorder_signatures=reorder_signatures
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(0.3 * n_samples, 4))

    if colors is None:
        colors = list(sns.color_palette("deep")) * (1 + n_signatures // 10)

    bottom = np.zeros(n_samples)

    for signature, color in zip(exposures.T, colors):
        signature_exposures = exposures.T[signature].to_numpy()
        ax.bar(
            np.arange(n_samples),
            signature_exposures,
            color=color,
            width=1,
            label=signature,
            linewidth=0,
            bottom=bottom,
            **kwargs,
        )
        bottom += signature_exposures

    if annotate_samples:
        ax.set_xticks(np.arange(n_samples))
        ax.set_xticklabels(exposures.columns, rotation=90, ha="center", fontsize=10)

    else:
        ax.get_xaxis().set_visible(False)

    ax.set_title("Sample exposures")
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(0.975, 0.5), ncol=ncol_legend)

    return ax
