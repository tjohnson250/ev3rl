# Code from: https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy.ma as ma

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Arguments:
        im         : The AxesImage to be labeled.
    Optional arguments:
        data       : Data used to annotate. If None, the image's data is used.
        valfmt     : The format of the annotations inside the heatmap.
                     This should either use the string format method, e.g.
                     "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
        textcolors : A list or array of two color specifications. The first is
                     used for values below a threshold, the second for those
                     above.
        threshold  : Value in data units according to which the colors from
                     textcolors are applied. If None (the default) uses the
                     middle of the colormap as separation.

    Further arguments are passed on to the created text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

# From https://github.com/matplotlib/matplotlib/pull/5054/files
class DivergingNorm(Normalize):
    """
    A subclass of matplotlib.colors.Normalize.
        Normalizes data into the ``[0.0, 1.0]`` interval.
    """

    def __init__(self, vmin=None, vcenter=None, vmax=None):
        """Normalize data with an offset midpoint
            Useful when mapping data unequally centered around a conceptual
        center, e.g., data that range from -2 to 4, with 0 as the midpoint.
            Parameters
        ----------
        vmin : float, optional
            The data value that defines ``0.0`` in the normalized data.
            Defaults to the min value of the dataset.
            vcenter : float, optional
            The data value that defines ``0.5`` in the normalized data.
            Defaults to halfway between *vmin* and *vmax*.
            vmax : float, optional
            The data value that defines ``1.0`` in the normalized data.
            Defaults to the the max value of the dataset.
            Examples
        --------
        >>> import matplotlib.colors as mcolors
        >>> offset = mcolors.DivergingNorm(vmin=-2., vcenter=0., vmax=4.)
        >>> data = [-2., -1., 0., 1., 2., 3., 4.]
        >>> offset(data)
        array([0., 0.25, 0.5, 0.625, 0.75, 0.875, 1.0])
        """
        self.vmin = vmin
        self.vcenter = vcenter
        self.vmax = vmax

    def __call__(self, value, clip=None):
        """Map value to the interval [0, 1]. The clip argument is unused."""
        result, is_scalar = self.process_value(value)
        self.autoscale_None(result)
        vmin, vcenter, vmax = self.vmin, self.vcenter, self.vmax
        if vmin == vmax == vcenter:
            result.fill(0)
        elif not vmin <= vcenter <= vmax:
            raise ValueError("minvalue must be less than or equal to "
                                "centervalue which must be less than or "
                                "equal to maxvalue")
        else:
            vmin = float(vmin)
            vcenter = float(vcenter)
            vmax = float(vmax)
            # in degenerate cases, prefer the center value to the extremes
            degen = (result == vcenter) if vcenter == vmax else None
            x, y = [vmin, vcenter, vmax], [0, 0.5, 1]
            result = ma.masked_array(np.interp(result, x, y),
                                        mask=ma.getmask(result))
            if degen is not None:
                result[degen] = 0.5
        if is_scalar:
                result = np.atleast_1d(result)[0]
        return result

    def autoscale_None(self, A):
        ' autoscale only None-valued vmin or vmax'
        if self.vmin is None and np.size(A) > 0:
            self.vmin = ma.min(A)
        if self.vmax is None and np.size(A) > 0:
            self.vmax = ma.max(A)
        if self.vcenter is None:
            self.vcenter = (self.vmax + self.vmin) * 0.5
