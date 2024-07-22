import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import LimitOption
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import medfilt, savgol_filter
from typing_extensions import List, Sequence, Tuple, Union


def export_numpy_array_with_header(
    export: bool,
    np_array: np.ndarray,
    fmt: Union[str, Sequence[str]],
    column_names: List[str],
    units: List[str],
    comments: List[str],
    index: int,
    sample_name: str,
    maccor_file_name: str,
    analysis_type: str,
    export_dir: Union[str, Path],
):
    """Save a numpy array to a text file. Prepend up to three lines for column
    names, units and comments.

    Parameters
    ----------
    export : boolean input.
        To be specified as '0' or '1'
    np_array : 1D or 2D array_like
        Data to be saved to a text file.
    fmt : str or sequence of strs
        A single format (%10.5f), a sequence of formats, or a
        multi-format string, e.g. 'Iteration %d -- %10.5f', in which
        case `delimiter` is ignored. For complex `X`, the legal options
        for `fmt` are:
            a) a single specifier, `fmt='%.4e'`, resulting in numbers formatted
               like `' (%s+%sj)' % (fmt, fmt)`
            b) a full string specifying every real and imaginary part, e.g.
               `' %.4e %+.4ej %.4e %+.4ej %.4e %+.4ej'` for 3 columns
            c) a list of specifiers, one per column - in this case, the real
               and imaginary part must have separate specifiers,
               e.g. `['%.3e + %.3ej', '(%.15e%+.15ej)']` for 2 columns
    column_names : list / array of str
    units : list / array of str
    comments : list / array of str
    index : int
    sample_name : str
    maccor_file_name : str
    analysis_type : str
    export_dir : str

    Returns
    -------
    out : TXT file in specified directory
        Tab delimited data with up to three header lines.

    See Also
    --------
    numpy.savetxt : Save an array to a text file.

    Notes
    -----

    References
    ----------
    .. [1] `Format Specification Mini-Language
           <http://docs.python.org/library/string.html#
           format-specification-mini-language>`_, Python Documentation.

    Examples
    --------

    """

    # import numpy as np
    if export == 1:
        # construct figure name from sample name and analysis type
        sample_name = sample_name.replace(".", "")
        fileName = (
            export_dir + "\\" + str(index) + "_" + sample_name + "_" + analysis_type
        )
        if comments is None:  # may be reasonable to always include a line
            column_names = "\t".join(column_names)
            units = "\t".join(units)
            header = (
                "Sample: "
                + sample_name
                + "\n"
                + "Parenting data file: "
                + maccor_file_name
                + "\n"
                + column_names
                + "\n"
                + units
                + "\n"
            )
        else:
            column_names = "\t".join(column_names)
            units = "\t".join(units)
            comments = "\t".join(comments)
            header = (
                "Sample: "
                + sample_name
                + "\n"
                + "Parenting data file: "
                + maccor_file_name
                + "\n"
                + column_names
                + "\n"
                + units
                + "\n"
                + comments
            )

        if os.path.exists(fileName + ".txt"):
            i = 1
            while os.path.exists(fileName + "_{}".format(i) + ".txt"):
                i += 1
            np.savetxt(
                fname=fileName + "_{}".format(i) + ".txt",
                X=np_array,
                fmt=fmt,
                delimiter="\t",
                newline="\n",
                header=header,
                footer="",
                comments="",
                encoding=None,
            )
        else:
            np.savetxt(
                fname=fileName + ".txt",
                X=np_array,
                fmt=fmt,
                delimiter="\t",
                newline="\n",
                header=header,
                footer="",
                comments="",
                encoding=None,
            )


def export_graph(
    export: bool,
    figure: plt.Figure,
    legend: plt.Legend,
    index: int,
    sample_name: str,
    analysis_type: str,
    export_dir,
):
    """Save a matplotlib.pyplot figue to a PNG file.

    Parameters
    ----------
    export : boolean input.
        To be specified as '0' or '1'
    figure : pyplot figure object

    legend : pyplot object
        Constructed from legend handles. Determines extra space on plot layer
    index : int
    sample_name : str
    analysis_type : str
    export_dir : str

    Returns
    -------
    out : PNG file in specified directory

    See Also
    --------
    matplotlib.pyplot.savefig : Save the current figure

    Notes
    -----
    Resolution is set to 300 dpi in the function definition. If a higher
    resolution is needed, e.g. for publication, a higher resolution can be
    specified.

    References
    ----------
    https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html

    """

    if export == 1:
        # construct figure name from sample name and analysis type
        sample_name = sample_name.replace(".", "")
        fileName = (
            export_dir + "\\" + str(index) + "_" + sample_name + "_" + analysis_type
        )
        if os.path.exists(fileName + ".png"):
            i = 1
            while os.path.exists(fileName + "_{}".format(i) + ".png"):
                i += 1
            figure.savefig(
                fname=fileName + "_{}".format(i) + ".png",
                dpi=300,
                bbox_extra_artists=(legend,),
                bbox_inches="tight",
            )
        else:
            figure.savefig(
                fname=fileName,
                dpi=300,
                bbox_extra_artists=(legend,),
                bbox_inches="tight",
            )
        # Note that the bbox_extra_artists must be an iterable


def raise_window(fig_name: str = None):
    """Raise the plot window for Figure figname to the foreground.  If no argument
    is given, raise the current figure.

    This function will only work with a Qt graphics backend.  It assumes you
    have already executed the command 'import matplotlib.pyplot as plt'.
    """
    if fig_name:
        plt.figure(fig_name)
        cfm = plt.get_current_fig_manager()
        cfm.window.activateWindow()
        cfm.window.raise_()


def smooth_methods(
    selection: int,
    input_array: np.ndarray,
    filter_window: int,
    sigma: float,
    poly_order: int,
) -> np.ndarray:
    """
    Insert description here

    Parameters
    ----------
    selection : int
        [0; 4] specifies operation to be carried out
    input_array : ndarray
        The data to be filtered
    filter_window : int
        The length of the filter window, must be a positive odd integer.
    sigma : scalar
        Gaussian filter only: Standard deviation for Gaussian kernel
    poly_order : int
        Savitzky Golay only: The order of the polynomial used to fit the
        samples. polyorder must be less than filter_window.

    Returns
    -------
    out : ndarray
        Smoothed data
        0 : unprocesses input data
        1 : ndarray, same shape as inputArray.
        2 : ndarray
        3 : An array the same size as input containing the median filtered
            result.
        4 : pandas.Series, same shape as inputArray.

    See Also
    --------
    scipy.signal.savgol_filter
    scipy.signal.medfilt
    scipy.ndimage.filters.gaussian_filter1d
    pandas.Series.rolling.mean : Calculate the rolling mean of the values.

    """
    if selection == 0:  # no smoothing
        return input_array
    elif selection == 1:  # Savitzky Golay
        return savgol_filter(
            input_array,
            window_length=filter_window,
            polyorder=poly_order,
            mode="nearest",
        )
    elif selection == 2:  # Gaussian filter
        return gaussian_filter1d(input_array, sigma=sigma, mode="reflect")
    elif selection == 3:  # Median filter
        return medfilt(input_array, kernel_size=filter_window)
    elif selection == 4:  # Adjacent averaging filter
        input_series = pd.Series(input_array)
        return input_series.rolling(window=filter_window, center=True).mean()
    else:
        raise ValueError("Invalid selection for smoothing method")


def plot_cycle_wise_single_scale(
    figure_name: str,
    colormap: str,
    cycles_to_plot: str,
    cycle_list: List[int],
    array: np.ndarray,
    n_col: int,
    x1_col: int,
    y1_col: int,
    x2_col: int,
    y2_col: int,
    comments: List[str],
    x_title: str,
    y_title: str,
    x_limits: Union[LimitOption, Tuple[float, float]],
    y_limits: Union[LimitOption, Tuple[float, float]],
    cap_unit: str,
    active_material_mass_unit: str,
):
    """
    Plot data from array to figure with one layer 'axis'. From a pair of plots,
    only the first one receives a legend. Axis ticks and labels are determined
    from specified units. Titles have to be passed as arguments.

    Parameters
    ----------
    figure_name : str
        Unique (!) name of to create figure object / window
    colormap : str
        Chosen from list of coloramps on
        https://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
    cycles_to_plot : list of integers
        List of cycles to be plotted
    cycle_list : list of integers
        List of all cycles in the data set
    array : ndarray
        Structure containing data to be plotted
    n_col : int
        Number of columns per cycle
    x1_col : int
        column index within 'Array' for x-axis of forward cycle
    y1_col : int
        column index within 'Array' for y-axis of forward cycle
    x2_col : int
        column index within 'Array' for x-axis of backward cycle
    y2_col : int
        column index within 'Array' for y-axis of backward cycle
    comments : list of strings
        List of legend entries - usually cycle count 'cycle #'
    x_title : str
        Specifies measured variable and unit
    y_title : str
        Specifies measured variable and unit
    x_limits : tuple
        [xmin, xmax] or 'auto'
    y_limits : tuple
        [ymin, ymax] or 'auto'
    cap_unit : str
        Several cases - used to determine axis scaling / ticks
    active_material_mass_unit : str
        Several cases - used to determine axis scaling / ticks

    Returns
    -------
    out : matplotlib.pyplot figure object

    """
    if plt.fignum_exists(figure_name):
        plt.close(figure_name)

    fig = plt.figure(figure_name)
    ax1 = fig.add_subplot(111)

    for cycle in cycles_to_plot:
        i_cyc = np.where(cycle_list == cycle)[0][0]
        i_ctp = np.where(cycles_to_plot == cycle)[0][0]
        # plot of charge
        ax1.plot(
            array[:, i_cyc * n_col + x1_col],
            array[:, i_cyc * n_col + y1_col],
            color=colormap[i_ctp],
            label=comments[i_cyc * n_col + y1_col],
        )
        # plot of discharge
        ax1.plot(
            array[:, i_cyc * n_col + x2_col],
            array[:, i_cyc * n_col + y2_col],
            color=colormap[i_ctp],
            label="_nolegend_",
        )

    ax1.set_xlabel(x_title, fontdict=None, labelpad=None)
    ax1.set_ylabel(y_title, fontdict=None, labelpad=None)
    # In case user wants to set axis limits  manually, this parameter is used
    # if not set to 'auto'
    if x_limits != "auto":
        ax1.set_xlim(x_limits)  # xScale needs to be specified as [xmin,xmax]

    if y_limits != "auto":
        ax1.set_ylim(y_limits)  # yScale needs to be specified as [ymin,ymax]

    # To catch the mAh/kg exception scientific notation is used
    if (
        (cap_unit == "mAh")
        & (active_material_mass_unit == "kg")
        & (x_title[0:7] == "Capacity")
    ):
        ax1.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    h1, l1 = ax1.get_legend_handles_labels()
    lgd = ax1.legend(
        h1,
        l1,
        bbox_to_anchor=(0, 1.05, 1, 0.105),
        loc="lower left",
        ncol=4,
        mode="expand",
        borderaxespad=0,
    )
    # Get number of legend entries to estimate nedded  rectangle size
    num_leg = len(l1)
    extra = 0.5 * round(num_leg / 4) * 4 / 50  # 4 equals ncol
    fig.tight_layout(rect=(0, 0, 1, 1 + extra))

    return fig, lgd


def plot_cycle_wise_dual_scale(
    figure_name: str,
    colormap: str,
    cycles_to_plot: str,
    cycle_list: List[int],
    array: np.ndarray,
    n_col: int,
    x1_col: int,
    y1_col: int,
    y2_col: int,
    comments: List[str],
    x_title: str,
    y1_title: str,
    y2_title: str,
    x_limits: Union[LimitOption, Tuple[float, float]],
    y1_limits: Union[LimitOption, Tuple[float, float]],
    y2_limits: Union[LimitOption, Tuple[float, float]],
    cap_unit: str,
    active_material_mass: str,
):
    """
    Plot data from array to figure with two layer 'axes'. From a pair of plots,
    only the first one receives a legend. Axis ticks and labels are determined
    from specified units. Titles have to be passed as arguments.

    Parameters
    ----------
    figure_name : str
        Unique (!) name of to create figure object / window
    colormap : str
        Chosen from list of coloramps on
        https://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
    cycles_to_plot : list of integers
        List of cycles to be plotted
    array : ndarray
        Structure containing data to be plotted
    n_col : int
        Number of columns per cycle
    x1_col : int
        column index within 'Array' for x-axis
    y1_col : int
        column index within 'Array' for left y-axis
    y2_col : int
        column index within 'Array' for right y-axis
    comments : list of strings
        List of legend entries - usually cycle count 'cycle #'
    x_title : str
        Specifies measured variable and unit
    y1_title : str
        Specifies measured variable and unit
    y2_title : str
        Specifies measured variable and unit
    x_limits : tuple
        [xmin, xmax] or 'auto'
    y1_limits : tuple
        [ymin, ymax] or 'auto'
    y2_limits : tuple
        [ymin, ymax] or 'auto'
    cap_unit : str
        Several cases - used to determine axis scaling / ticks
    active_material_mass : str
        Several cases - used to determine axis scaling / ticks

    Returns
    -------
    fig : matplotlib.pyplot figure object
    leg : matplotlib.pyplot legend object
    xlim : [xmin, xmax] - boundaries of the x-axis

    """
    if plt.fignum_exists(figure_name):
        plt.close(figure_name)

    fig = plt.figure(figure_name)
    # Anode
    col1 = "tab:blue"
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(x_title, fontdict=None, labelpad=None)
    ax1.set_ylabel(y1_title, fontdict=None, labelpad=None, color=col1)
    ax1.tick_params(axis="y", labelcolor=col1)
    # Cathode
    col2 = "tab:red"
    ax2 = ax1.twinx()
    ax2.set_ylabel(y2_title, fontdict=None, labelpad=None, color=col2)
    ax2.tick_params(axis="y", labelcolor=col2)

    for cycle in cycles_to_plot:
        i_cyc = np.where(cycle_list == cycle)[0][0]
        i_ctp = np.where(cycles_to_plot == cycle)[0][0]
        # plot of anode
        ax1.plot(
            array[:, i_cyc * n_col + x1_col],
            array[:, i_cyc * n_col + y1_col],
            color=colormap[i_ctp],
            label=comments[i_cyc * n_col + y1_col],
        )
        # plot of cathode
        ax2.plot(
            array[:, i_cyc * n_col + x1_col],
            array[:, i_cyc * n_col + y2_col],
            color=colormap[i_ctp],
            label="_nolegend_",
        )

    # In case user wants to set axis limits  manually, this parameter is used
    # if not set to 'auto'
    if x_limits != "auto":
        ax1.set_xlim(x_limits)  # xScale needs to be specified as [xmin,xmax]
    if y1_limits != "auto":
        ax1.set_ylim(y1_limits)  # y-range needs to be specified as [ymin,ymax]
    if y2_limits != "auto":
        ax2.set_ylim(y2_limits)  # y-range needs to be specified as [ymin,ymax]

    # To make charge and discharge cyle same x-scale
    xlim = ax1.get_xlim()  # will be passed to second call of this function

    # To catch the mAh/kg exception scientific notation is used
    if (
        (cap_unit == "mAh")
        & (active_material_mass == "kg")
        & (x_title[0:7] == "Capacity")
    ):
        ax1.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    # Plot an annotation arrow indicating which plots belong to which axes
    # Left y-axis
    line1 = ax1.lines[0]  # first plot in axis 1
    i = 0
    while np.isnan(line1.get_xdata()[i]):  # look for first non NaN entry
        if i < len(line1.get_xdata()) - 1:
            i += 1
        else:
            break
    y1foot_x = line1.get_xdata()[i]  # first entry of first cycle in ctp
    y1foot_y = line1.get_ydata()[i]  # first entry of first cycle in ctp
    ax1.annotate(
        "",
        xy=(xlim[0], y1foot_y),
        xytext=(y1foot_x, y1foot_y),
        arrowprops=dict(arrowstyle="->"),
    )
    # Right y-axis
    line2 = ax2.lines[0]  # first plot in axis 2
    i = -1
    while np.isnan(line2.get_xdata()[i]):  # look for last non NaN entry
        if i > -(len(line1.get_xdata()) - 1):
            i -= 1
        else:
            break
    y2foot_x = line2.get_xdata()[i]  # last entry of first cycle in ctp
    y2foot_y = line2.get_ydata()[i]  # last entry of first cycle in ctp
    ax2.annotate(
        "",
        xy=(xlim[1], y2foot_y),
        xytext=(y2foot_x, y2foot_y),
        arrowprops=dict(arrowstyle="->"),
    )

    h1, l1 = ax1.get_legend_handles_labels()
    lgd = ax1.legend(
        h1,
        l1,
        bbox_to_anchor=(0, 1.05, 1, 0.105),
        loc="lower left",
        ncol=4,
        mode="expand",
        borderaxespad=0,
    )
    # Get number of legend entries to estimate nedded  rectangle size
    num_leg = len(l1)
    extra = 0.5 * round(num_leg / 4) * 4 / 50  # 4 equals ncol
    fig.tight_layout(rect=(0, 0, 1, 1 + extra))

    return fig, lgd, xlim
