from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

import fastcluster
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from matplotlib.axes import Axes
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

from .consts import COLORS_INDEL83, COLORS_SBS96, INDEL_TYPES_83, SBS_TYPES_96
from .utils import _concat_light, _get_basis_obsm, _get_basis_obsp, match_to_catalog

if TYPE_CHECKING:
    from anndata import AnnData
    from matplotlib.colors import Colormap
    from matplotlib.typing import ColorType
    from mudata import MuData
    from seaborn.matrix import ClusterGrid


def set_salamander_style():
    sns.set_context("notebook")
    sns.set_style("ticks")
    params = {
        "axes.edgecolor": "black",
        "axes.labelsize": "medium",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": "large",
        "font.family": "DejaVu Sans",
        "legend.fontsize": "medium",
        "pdf.fonttype": 42,
        "xtick.labelsize": "small",
        "ytick.labelsize": "small",
    }
    mpl.rcParams.update(params)


def history(
    values: np.ndarray,
    conv_test_freq: int,
    min_iteration: int = 0,
    ax: Axes | None = None,
    **kwargs,
) -> Axes:
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
    ax: Axes,
    data: np.ndarray,
    annotations: Iterable[str],
    fontsize: float | str = "small",
    color: ColorType = "black",
    adjust_annotations: bool = True,
    adjust_kwargs: dict[str, Any] | None = None,
    **kwargs,
) -> None:
    for data_point, annotation in zip(data, annotations):
        ax.text(
            data_point[0],
            data_point[1],
            annotation,
            fontsize=fontsize,
            color=color,
            **kwargs,
        )
    if adjust_annotations:
        adjust_kwargs = {} if adjust_kwargs is None else adjust_kwargs.copy()
        texts = [
            child for child in ax.get_children() if isinstance(child, mpl.text.Text)
        ]
        texts_nonempty = [annotation for annotation in texts if annotation.get_text()]
        adjust_text(texts_nonempty, **adjust_kwargs)


def _scatter_1d(
    data: np.ndarray,
    xlabel: str | None = None,
    color: list[ColorType] | None = None,
    zorder: list[int] | None = None,
    ax: Axes | None = None,
    **kwargs,
) -> Axes:
    if data.ndim != 1:
        raise ValueError("The data have to be one-dimensional.")

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 1))

    if zorder is None:
        zorder = len(data) * [1]

    ax.spines[["left", "bottom"]].set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.axhline(y=0, color="black", zorder=np.min(zorder) - 1)

    for zord in np.unique(zorder):
        subgroup = np.where(zorder == zord)[0]
        subdata = data[subgroup]
        subgroup_color = [color[d] for d in subgroup] if color is not None else None
        sns.scatterplot(
            x=subdata,
            y=np.zeros_like(subdata),
            color=subgroup_color,
            zorder=zord,
            ax=ax,
            **kwargs,
        )

    if xlabel:
        ax.set_xlabel(xlabel)

    return ax


def _scatter_2d(
    data: np.ndarray,
    xlabel: str | None = None,
    ylabel: str | None = None,
    ticks: bool = True,
    color: list[ColorType] | None = None,
    zorder: list[int] | None = None,
    ax: Axes | None = None,
    **kwargs,
) -> Axes:
    """
    The rows (!) of 'data' are assumed to be the data points.
    """
    if data.shape[1] != 2:
        raise ValueError("The datapoints (rows) have to be two-dimensional.")

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    if zorder is None:
        zorder = len(data) * [1]

    for zord in np.unique(zorder):
        subgroup = np.where(zorder == zord)[0]
        subdata = data[subgroup]
        subgroup_color = [color[d] for d in subgroup] if color is not None else None
        sns.scatterplot(
            x=subdata[:, 0],
            y=subdata[:, 1],
            color=subgroup_color,
            zorder=zord,
            ax=ax,
            **kwargs,
        )

    if xlabel:
        ax.set_xlabel(xlabel)

    if ylabel:
        ax.set_ylabel(ylabel)

    if not ticks:
        ax.set(xticks=[], yticks=[])

    return ax


def scatter_numpy(
    data: np.ndarray,
    xlabel: str | None = None,
    ylabel: str | None = None,
    ticks: bool = True,
    color: list[ColorType] | None = None,
    zorder: list[int] | None = None,
    annotations: Iterable[str] | None = None,
    annotation_kwargs: dict[str, Any] | None = None,
    adjust_annotations: bool = True,
    adjust_kwargs: dict[str, Any] | None = None,
    ax: Axes | None = None,
    **kwargs,
) -> Axes:
    if data.ndim == 1 or data.shape[1] == 1:
        ax = _scatter_1d(data, xlabel, color, zorder, ax, **kwargs)
        data_2d = np.vstack([data.flatten(), np.zeros_like(data.flatten())]).T
    elif data.ndim == 2 and data.shape[1] == 2:
        ax = _scatter_2d(data, xlabel, ylabel, ticks, color, zorder, ax, **kwargs)
        data_2d = data
    else:
        raise ValueError(
            "Scatterplots are only supported for one- or two-dimensional data."
        )

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


def scatter(
    adata: AnnData | MuData,
    x: str,
    y: str | None = None,
    ticks: bool = True,
    color: str | None = None,
    zorder: str | None = None,
    **kwargs,
) -> Axes:
    if y is None:
        data = adata.obs[x].to_numpy()
    else:
        data = adata.obs[[x, y]].to_numpy()

    col = list(adata.obs[color]) if color is not None else None
    zord = list(adata.obs[zorder]) if zorder is not None else None
    ax = scatter_numpy(
        data, xlabel=x, ylabel=y, ticks=ticks, color=col, zorder=zord, **kwargs
    )
    return ax


def scatter_multiple(
    adatas: Iterable[AnnData | MuData],
    x: str,
    y: str | None = None,
    ticks: bool = True,
    color: str | None = None,
    zorder: str | None = None,
    **kwargs,
) -> Axes:
    obs_keys = [x, y, color, zorder]
    obs_keys = [key for key in obs_keys if key is not None]
    combined = _concat_light(adatas, obs_keys=obs_keys)
    ax = scatter(
        adata=combined, x=x, y=y, ticks=ticks, color=color, zorder=zorder, **kwargs
    )
    return ax


def embedding_numpy(
    data: np.ndarray,
    dimensions: tuple[int, int] = (0, 1),
    xlabel: str | None = None,
    ylabel: str | None = None,
    ticks: bool = True,
    color: list[ColorType] | None = None,
    zorder: list[int] | None = None,
    **kwargs,
) -> Axes:
    if data.ndim == 2 and data.shape[1] > 2:
        data = data[:, dimensions]

    ax = scatter_numpy(data, xlabel, ylabel, ticks, color, zorder, **kwargs)
    return ax


# fmt: off
def _basisobsm2name(basis: str) -> str:
    name = (
        "PC" if basis == "pca"
        else "tSNE" if basis == "tsne"
        else "UMAP" if basis == "umap"
        else basis
    )
    return name
# fmt: on


def embedding(
    adata: AnnData | MuData,
    basis: str,
    dimensions: tuple[int, int] = (0, 1),
    xlabel: str | None = None,
    ylabel: str | None = None,
    ticks: bool | None = None,
    color: str | None = None,
    zorder: str | None = None,
    **kwargs,
) -> Axes:
    data = _get_basis_obsm(adata, basis)
    name = _basisobsm2name(basis)
    labels = [f"{name}{d+1}" for d in dimensions]

    if xlabel is None:
        xlabel = labels[0]

    if ylabel is None:
        ylabel = labels[1]

    if ticks is None:
        ticks = False if basis in ["tsne", "umap"] else True

    col = list(adata.obs[color]) if color is not None else None
    zord = list(adata.obs[zorder]) if zorder is not None else None
    ax = embedding_numpy(
        data,
        dimensions=dimensions,
        xlabel=xlabel,
        ylabel=ylabel,
        ticks=ticks,
        color=col,
        zorder=zord,
        **kwargs,
    )
    return ax


def embedding_multiple(
    adatas: Iterable[AnnData | MuData],
    basis: str,
    dimensions: tuple[int, int] = (0, 1),
    xlabel: str | None = None,
    ylabel: str | None = None,
    ticks: bool | None = None,
    color: str | None = None,
    zorder: str | None = None,
    **kwargs,
) -> Axes:
    obs_keys = [color, zorder]
    obs_keys = [key for key in obs_keys if key is not None]
    combined = _concat_light(adatas, obs_keys=obs_keys, obsm_keys=[basis])
    ax = embedding(
        adata=combined,
        basis=basis,
        dimensions=dimensions,
        xlabel=xlabel,
        ylabel=ylabel,
        ticks=ticks,
        color=color,
        zorder=zorder,
        **kwargs,
    )
    return ax


def pca(adata: AnnData, **kwargs) -> Axes:
    return embedding(adata, basis="pca", **kwargs)


def pca_multiple(adatas: Iterable[AnnData | MuData], **kwargs) -> Axes:
    return embedding_multiple(adatas, basis="pca", **kwargs)


def tsne(adata: AnnData, **kwargs) -> Axes:
    return embedding(adata, basis="tsne", **kwargs)


def tsne_multiple(adatas: Iterable[AnnData | MuData], **kwargs) -> Axes:
    return embedding_multiple(adatas, basis="tsne", **kwargs)


def umap(adata: AnnData, **kwargs) -> Axes:
    return embedding(adata, basis="umap", **kwargs)


def umap_multiple(adatas: Iterable[AnnData | MuData], **kwargs) -> Axes:
    return embedding_multiple(adatas, basis="umap", **kwargs)


def correlation_pandas(
    corr: pd.DataFrame,
    figsize: tuple[float, float] = (4.0, 4.0),
    cmap: Colormap | str | None = "vlag",
    fmt: str = ".2f",
    **kwargs,
) -> ClusterGrid:
    linkage = hierarchy.linkage(corr)
    clustergrid = sns.clustermap(
        corr,
        row_linkage=linkage,
        figsize=figsize,
        vmin=-1,
        vmax=1,
        cmap=cmap,
        fmt=fmt,
        **kwargs,
    )
    return clustergrid


def correlation(adata: AnnData, **kwargs) -> ClusterGrid:
    corr = pd.DataFrame(
        _get_basis_obsp(adata, "correlation"),
        index=adata.obs_names,
        columns=adata.obs_names,
    )
    return correlation_pandas(corr, **kwargs)


def _get_colors_barplot(var_names, colors=None):
    """
    Given the variable names / features and the colors argument of barplot,
    return the final colors used in the bar chart.
    """
    n_vars = len(var_names)

    if colors == "SBS96" or (
        n_vars == 96 and all(var_names == SBS_TYPES_96) and colors is None
    ):
        if n_vars != 96:
            raise ValueError(
                "The standard SBS colors can only be used "
                "when the signatures have 96 features."
            )
        colors = COLORS_SBS96

    elif colors == "Indel83" or (
        n_vars == 83 and all(var_names == INDEL_TYPES_83) and colors is None
    ):
        if n_vars != 83:
            raise ValueError(
                "The standard Indel colors can only be used "
                "when the signatures have 83 features."
            )
        colors = COLORS_INDEL83

    elif type(colors) in [str, tuple]:
        colors = n_vars * [colors]

    elif type(colors) is list:
        if len(colors) != n_vars:
            raise ValueError(f"The list of colors must be of length n_vars={n_vars}.")

    else:
        colors = n_vars * ["gray"]

    return colors


def _barplot_single(
    data: pd.DataFrame,
    colors: ColorType | list[ColorType] | None = None,
    annotate_vars: bool = False,
    ax: Axes | None = None,
    **kwargs,
) -> Axes:
    """
    Plot the relative values of a non-negative dataframe
    with a single row.

    Inputs
    ------
    data: pd.DataFrame
        A dataframe with only one row, typically a single signature
        or the feature counts of a single sample.
        The columns of data are expected to be the names of the features.

    colors:
        Can be set to 'SBS96' or 'Indel83' to use the standard bar colors
        for these features.
        Otherwise, when a single color is provided, all bars will have
        the same color. Alternatively, a list can be used to specifiy
        the color of each bar individually.

    annotate_vars: bool, default=False
        If True, the x-axis has ticks and annotations.

    ax:
        Axes object to draw the plot onto. Otherwise, create an Axes internally.

    kwargs: dict
        Any keyword arguments to be passed to matplotlibs ax.bar
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 1))

    data_normalized = data.div(data.sum(axis=1), axis=0)
    var_names = data.columns
    colors = _get_colors_barplot(var_names, colors)

    ax.set_title(data.index[0])
    ax.spines["left"].set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_xlim((-1, len(var_names)))

    ax.bar(
        var_names,
        data_normalized.iloc[0, :],
        linewidth=0,
        color=colors,
        **kwargs,
    )

    if annotate_vars:
        ax.set_xticks(var_names)
        ax.set_xticklabels(
            var_names, family="monospace", fontsize="x-small", ha="center", rotation=90
        )
    else:
        ax.set_xticks([])

    return ax


def _barplot_matched(
    data: pd.DataFrame,
    catalog: pd.DataFrame | None = None,
    colors: ColorType | list[ColorType] | None = None,
    annotate_vars: bool = False,
    ax: Axes | Iterable[Axes] | None = None,
    **kwargs,
) -> Axes | Iterable[Axes]:
    """
    Plot the relative values of a non-negative dataframe
    with a single row.
    The closest matching row from a 'catalog' can also be plotted.

    Inputs
    ------
    data: pd.DataFrame
        A dataframe with only one row, typically a single mutational
        signature.

    catalog: pd.DataFrame
        If a catalog with matching features is provided, the single best
        matching row will also be plotted.

    colors: str, tuple or list
        Can be set to 'SBS96' or 'Indel83' to use the standard bar colors
        for these features.
        Otherwise, when a single color is provided, all bars will have
        the same color. Alternatively, a list can be used to specifiy
        the color of each bar individually.

    annotate_vars: bool, default=False
        If True, the x-axis has ticks and annotations.

    ax:
        Axes object(s) to draw the plot onto. A single Axes if catalog is None;
        two Axes if a catalog is given.

    kwargs: dict
        Any keyword arguments to be passed to matplotlibs ax.bar
    """
    if catalog is None:
        assert isinstance(ax, Axes) or ax is None
        return _barplot_single(
            data, colors=colors, annotate_vars=annotate_vars, ax=ax, **kwargs
        )

    if ax is None:
        _, axes = plt.subplots(1, 2, figsize=(8, 1))
    else:
        axes = ax

    matched_data = match_to_catalog(data, catalog, metric="cosine")
    data_all = [data, matched_data]

    for d, axis in zip(data_all, axes):
        _barplot_single(
            d,
            colors=colors,
            annotate_vars=annotate_vars,
            ax=axis,
            **kwargs,
        )

    return axes


def barplot_pandas(
    data: pd.DataFrame,
    catalog: pd.DataFrame | None = None,
    colors: ColorType | list[ColorType] | None = None,
    annotate_vars: bool = False,
    axes: Axes | Iterable[Axes] | None = None,
    **kwargs,
) -> Axes | Iterable[Axes]:
    """
    Plot the relative values of the rows of a non-negative dataframe.
    The closest matching rows from a 'catalog' can also be plotted.

    Inputs
    ------
    data : pd.DataFrame
        Annotated dataframe of shape (n_obs, n_vars), typically
        a collection of mutational signatures.

    catalog: pd.DataFrame
        If a catalog with matching features is provided, the best matching
        rows of the catalog are also plotted.

    colors: str, tuple or list
        Can be set to 'SBS96' or 'Indel83' to use the standard bar colors
        for these features.
        Otherwise, when a single color is provided, all bars will have
        the same color. Alternatively, a list can be used to specifiy
        the color of each bar individually.

    annotate_vars: bool, default=False
        If True, the x-axis has ticks and annotations.

    axes : Axes | list[Axes]
        Axes object(s) to draw the plot onto. Multiple Axes if 'data' has more than
        one column or a catalog is given. Otherwise a single Axes.
        When a catalog is provided, axes is expected to be of shape (n_obs, 2).
    """
    n_obs = data.shape[0]

    if n_obs == 1:
        return _barplot_matched(
            data,
            catalog=catalog,
            colors=colors,
            annotate_vars=annotate_vars,
            ax=axes,
            **kwargs,
        )

    if axes is None:
        if catalog is None:
            _, axes = plt.subplots(n_obs, 1, figsize=(4, n_obs))
        else:
            _, axes = plt.subplots(n_obs, 2, figsize=(8, n_obs))

    assert isinstance(
        axes, Iterable
    ), "Adding multiple barplots to custom 'axes' requires 'axes' to be iterable."

    for ax, row in zip(axes, data.T):
        _barplot_matched(
            data.loc[[row], :],
            catalog=catalog,
            colors=colors,
            annotate_vars=annotate_vars,
            ax=ax,
            **kwargs,
        )

    plt.tight_layout()
    return axes


def barplot(adata: AnnData, **kwargs) -> Axes | Iterable[Axes]:
    return barplot_pandas(adata.to_df(), **kwargs)


def get_obs_order(data: pd.DataFrame, normalize: bool = True) -> np.ndarray:
    """
    Compute the aesthetically most pleasing order of the observations
    of a non-negative data array of shape (n_obs, n_dimensions) for a
    stacked barchart using hierarchical clustering.

    Inputs
    ------
    data : pd.DataFrame of shape (n_obs, n_dimensions)
        An annotated non-negative data matrix, typically the signature
        exposures.

    normalize : bool, default=True
        If True, the data is row-normalized before computing the
        optimal order.

    Returns
    -------
    order : np.ndarray
        The ordered observations.
    """
    if normalize:
        # no in-place
        data = data.div(data.sum(axis=1), axis=0)

    d = pdist(data)
    linkage = fastcluster.linkage(d)
    # get the optimal order that is consistent
    # with the hierarchical clustering linkage
    obs_order = hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(linkage, d))
    obs_order = data.index[obs_order].to_numpy()
    return obs_order


def _reorder_data(
    data: pd.DataFrame,
    obs_order: np.ndarray | None = None,
    normalize: bool = True,
    reorder_dimensions: bool = True,
) -> pd.DataFrame:
    """
    Reorder non-negative data using hierarchical clustering and optionally
    reorder the dimensions by their total relative sums.

    Inputs
    ------
    data : pd.DataFrame of shape (n_obs, n_dimensions)
        An annotated non-negative data matrix, typically the signature
        exposures.

    obs_order : np.ndarray, default=None
        An optional predefined order of the observations.

    normalize : bool, default=True
        If True, the data is row-normalized before computing the
        optimal order. Only used if 'data_order' is not given.

    reorder_dimensions : bool, default=True
        If True, the dimensions/columns will be reordered such that their
        total relative sums decrease from the left to the right.

    Returns
    -------
    data_reordered : pd.DataFrame of shape (n_obs, n_dimensions)
        The reorderd annotated data.
    """
    if obs_order is None:
        obs_order = get_obs_order(data, normalize=normalize)

    data_reordered = data.loc[obs_order, :]

    # order the columns by their total relative contribution
    if reorder_dimensions:
        data_normalized = data.div(data.sum(axis=1), axis=0)
        dim_ordered = data_normalized.sum(axis=0).sort_values(ascending=False).index
        data_reordered = data_reordered[dim_ordered]

    return data_reordered


def stacked_barplot(
    data: pd.DataFrame,
    obs_order: np.ndarray | None = None,
    reorder_dimensions: bool = True,
    annotate_obs: bool = True,
    colors: Iterable[ColorType] | None = None,
    title: str | None = None,
    ncol_legend: int = 1,
    ax: Axes | None = None,
    **kwargs,
) -> Axes:
    """
    Visualize non-negative data with a stacked bar chart, typically
    the signature exposures.

    Inputs
    ------
    data : pd.DataFrame of shape (n_obs, n_dimensions)
        An annotated non-negative data matrix, typically the signature
        exposures.

    obs_order : np.ndarray, default=None
        An optional predefined order of the observations.
        If None, hierarchical clustering is used to compute the
        aesthetically most pleasing order.

    reorder_dimensions : bool, default=True
        If True, the columns of 'data' will be reordered such that their
        total relative contributions in the stacked bar chart is increasing.

    annotate_obs : bool, default=True
        If True, the x-axis is annotated with the observation names.

    colors : iterable of length n_dimensions, default=None
        Colors to pass to matplotlibs ax.bar, one per dimension.

    n_col_legend : int, default=1
        The number of columns of the legend.

    ax : matplotlib.axes.Axes, default=None
        Axes object to draw the plot onto. Otherwise, create an Axes internally.

    kwargs : dict
        Any keyword arguments to be passed to matplotlibs ax.bar.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The matplotlib axes containing the plot.
    """
    n_obs, n_dimensions = data.shape
    data = data.div(data.sum(axis=1), axis=0)
    data = _reorder_data(
        data, obs_order=obs_order, reorder_dimensions=reorder_dimensions
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(0.3 * n_obs, 4))

    if colors is None:
        colors = sns.color_palette("deep") * (1 + n_dimensions // 10)

    bottom = np.zeros(n_obs)

    for dimension, color in zip(data, colors):
        values = data[dimension].to_numpy()
        ax.bar(
            np.arange(n_obs),
            values,
            color=color,
            width=1,
            label=dimension,
            linewidth=0,
            bottom=bottom,
            **kwargs,
        )
        bottom += values

    if annotate_obs:
        ax.set_xticks(np.arange(n_obs))
        ax.set_xticklabels(data.index, rotation=90, ha="center", fontsize="x-small")
    else:
        ax.get_xaxis().set_visible(False)

    if title:
        ax.set_title(title)

    ax.spines[["left", "bottom"]].set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(0.975, 0.5), ncol=ncol_legend)

    return ax
