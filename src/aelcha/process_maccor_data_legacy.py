import os
import re
from pathlib import Path

import digatron_utility
import maccor_utility
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bs4 import dammit
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import medfilt, savgol_filter
from typing_extensions import List, Sequence, Tuple, Union

from aelcha.common import LimitOption


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
    [1] `Format Specification Mini-Language
           <http://docs.python.org/library/string.html#
           format-specification-mini-language>`_, Python Documentation.

    Examples
    --------

    """

    # import numpy as np
    if export == 1:
        # construct figure name from sample name and analysis type
        sample_name = sample_name.replace(".", "")
        file_name = (
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

        if os.path.exists(file_name + ".txt"):
            i = 1
            while os.path.exists(file_name + "_{}".format(i) + ".txt"):
                i += 1
            np.savetxt(
                fname=file_name + "_{}".format(i) + ".txt",
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
                fname=file_name + ".txt",
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
    legend: plt.legend,
    index: int,
    sample_name: str,
    analysis_type: str,
    export_dir,
):
    """Save a matplotlib.pyplot figure to a PNG file.

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
        file_name = (
            export_dir + "\\" + str(index) + "_" + sample_name + "_" + analysis_type
        )
        if os.path.exists(file_name + ".png"):
            i = 1
            while os.path.exists(file_name + "_{}".format(i) + ".png"):
                i += 1
            figure.savefig(
                fname=file_name + "_{}".format(i) + ".png",
                dpi=300,
                bbox_extra_artists=(legend,),
                bbox_inches="tight",
            )
        else:
            figure.savefig(
                fname=file_name,
                dpi=300,
                bbox_extra_artists=(legend,),
                bbox_inches="tight",
            )
        # Note that the bbox_extra_artists must be an iterable


def raise_window(fig_name: str = None):
    """Raise the plot window for Figure fig_name to the foreground.  If no argument
    is given, raise the current figure.

    This function will only work with a Qt graphics backend.  It assumes you
    have already executed the command 'import matplotlib.pyplot as plt'.
    """
    if fig_name:
        plt.figure(fig_name)
        cfm = plt.get_current_fig_manager()
        cfm.window.activateWindow()
        cfm.window.raise_()


def convert_c2p(c2p, cycle_list):
    """
    Function to read c2p variable as passed by function processMaccorData and
    to decide whether to plot all cycles, a series of cycles specified by
    'm-n' with first cycle m and last cycle n to plot, or a list of cycles
    that are separated by ',' (ideally ', ')

    Parameters
    ----------
    c2p : str
        List of cycles to be plotted.
    cycle_list : ndarray
        List of unique elements in cyclization data's column 'cycle'. Elements
        of c2p are compared with cycleList to exclude invalid arguments and
        thereby avoid errors

    Returns
    -------
    out : ndarray
        Converted list of cycles to plot

    See Also
    --------
    numpy.unique : Find the unique elements of an array.

    """
    # import numpy.unique
    # import re.sub
    c2p = c2p.replace(" ", "")  # get rid of eventual spaces

    if c2p == "all":
        ctp = cycle_list
    elif "," in c2p:  # numbers separated by ',' in Excel sheet. This case
        # applies if numbers are either space with trailing or tailing spaces
        c2p_list = c2p.split(sep=",")  # get separated entries as string
        if "" in c2p_list:
            # remove empty entries '' as resulting from trailing ','
            c2p_list.remove("")
        c2p_int = list(map(int, c2p_list))  # map the list to integers
        c2p_unique = np.unique(c2p_int)  # allow unique values only
        c2p_unique.sort()  # sort them
        # exclude values not included in cycleList
        c2p_smaller = [i for i in c2p_unique if i <= max(cycle_list)]
        ctp = [i for i in c2p_smaller if i >= min(cycle_list)]
        ctp = np.array(ctp, dtype="int")
    elif "-" in c2p:  # series of cycles specified by 'm-n'
        c2p_list = c2p.split(sep="-")  # get separated entries as string
        c2p_int = list(map(int, c2p_list))  # map the list to integers
        c2p_int.sort()  # sort them
        m_n = c2p_int
        # exclude values not included in cycleList
        if m_n[1] > max(cycle_list):
            m_n[1] = max(cycle_list)
        if m_n[0] < min(cycle_list):
            m_n[0] = min(cycle_list)
        ctp = np.array(list(range(m_n[0], m_n[1] + 1)), dtype="int")
    elif "." in c2p:  # numbers separated by ',' in Excel sheet. Import option
        # decimal=',' read list of numbers as 1.2.3.4.5.
        c2p_list = c2p.split(sep=".")  # get separated entries as string
        if "" in c2p_list:
            # remove empty entries '' as resulting from trailing ','
            c2p_list.remove("")
        c2p_int = list(map(int, c2p_list))  # map the list to integers
        c2p_unique = np.unique(c2p_int)  # allow unique values only
        c2p_unique.sort()  # sort them
        # exclude values not included in cycleList
        c2p_smaller = [i for i in c2p_unique if i <= max(cycle_list)]
        ctp = [i for i in c2p_smaller if i >= min(cycle_list)]
        ctp = np.array(ctp, dtype="int")
    elif (
        (c2p != "all")
        & ("," not in c2p)
        & ("." not in c2p)
        & ("-" not in c2p)
        & (len(c2p) < 4)
    ):
        # the string c2p does not suit any above case and is a single integer
        # and consists of only 3 signs
        ctp = np.array(int(c2p), dtype="int")
    else:
        ctp = np.array([])
        print(
            "Your cycle selection did not match the suggested format! "
            + "Consider separating integer values by comma and space ', '!"
        )
    return ctp


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
    active_material_mass_unit: str,
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
    cycle_list : list of integers
        List of all cycles in the data set
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
    active_material_mass_unit : str
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
        & (active_material_mass_unit == "kg")
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


def process_maccor_data(
    index: int,
    input_dir: Union[str, Path],
    file_name: str,
    input_source_type: int,
    decimal_separator: str,
    maccor_dll_path: Path,
    sample_name: str,
    active_material_mass: float,
    three_electrode_cell,
    cycles_to_plot: str,
    draw_graphs_on_screen: bool,
    export_graphs: bool,
    export_analysis_data: bool,
    plot_cap_ce_vs_cn: bool,
    plot_volt_vs_cap: bool,
    plot_dq_dv: bool,
    dv_min_increment: float,
    dq_dv_filter_selection: int,
    dq_dv_filter_window_size: int,
    dq_dv_filter_sigma_or_polynomial_order: int,
    export_dir: Union[str, Path],
    colormap: str,
    cm_start: float,
    cm_end: float,
    v_c_x_limits: Union[list[float], tuple[float, float]],
    v_c_y_limits: Union[list[float], tuple[float, float]],
    v_c_y1_limits: Union[list[float], tuple[float, float]],
    v_c_y2_limits: Union[list[float], tuple[float, float]],
    dq_dv_x_limits: Union[list[float], tuple[float, float]],
    dq_dv_y_limits: Union[list[float], tuple[float, float]],
    dq_dv_x1_limits: Union[list[float], tuple[float, float]],
    dq_dv_x2_limits: Union[list[float], tuple[float, float]],
    use_dammit: bool = False,
):
    """
    Insert description here
    """
    # How many figure windows exist before evaluation --> only raise new ones
    # later on
    # figNumsIni = plt.get_fignums()
    if "." not in file_name:
        file_name += ".txt"
    elif ".csv" in file_name:
        # It is a Digatron csv file
        pass
    elif re.match(pattern=r".*(\.[0]{1}\d{2})$", string=file_name) is not None:
        # It is a Maccor raw file
        pass
    elif file_name.split(".")[-1].lower() != "txt":
        file_name += ".txt"
    else:
        pass

    file_path = Path(input_dir) / file_name
    print(f"Processing file {str(file_path)}")

    encoding = "utf-8"

    if use_dammit:
        # Read the file content
        with open(file_path, "rb") as file:
            raw_data = file.read()

        # Use UnicodeDammit to detect the encoding
        ud = dammit.UnicodeDammit(raw_data)
        encoding = ud.original_encoding

        print(f"Detected encoding: {encoding}")

    if input_source_type == -11:  # Maccor raw file
        if maccor_dll_path is None:
            raise ValueError("Maccor DLL path must be provided for raw files.")
        if not maccor_dll_path.exists():
            raise FileNotFoundError(f"Maccor DLL not found at {maccor_dll_path}")
        maccor_data_obj = maccor_utility.read.read_maccor_data_file(
            file_path=file_path,
            frmt=maccor_utility.read.MaccorDataFormat.raw,
            dll_path=maccor_dll_path,
        )
        maccor_df: pd.DataFrame = maccor_data_obj.data.as_dataframe
        maccor_df.rename(
            columns={
                "Capacity": "Cap_Ah",
                "Voltage": "Voltage_V",
                "CycleNumProc": "Cycle_P",
                "MainMode": "Md",
                "Aux1": "AUX1_V",
            },
            inplace=True,
        )
        maccor_data = maccor_df

    elif input_source_type == 10:  # or input_source_type == 13:
        # the file to import was exported from MIMS Client v1
        # enter number (indexing from 1) of the header containing column names
        # here
        n_header = 14  # later: read from Excel file or specify in a dictionary

        # determine the number of columns of the file. This is necessary since
        # maccor data file rows close with '\t' (tab) instead of '\n' (new
        # line) which causes import to expect one more col
        with open(file_path, "r", encoding=encoding) as maccor_file:
            for x in range(1, n_header + 1):  # range(m, n) = [m, ..., n-1]
                # In the last interation, this is the header line containing column
                # names
                header_line = maccor_file.readline()

        n_col = len(header_line.split("\t"))

        # option: read column names from header line. But: variables later on
        # are called according to the names that are defined here --> static
        # naming is preferable
        # probably the safer way to get/set column names:
        md_dtype = {
            "Rec": int,
            "Cycle_P": int,
            "Cycle_C": int,
            "Step": int,
            "TestTime": str,
            "StepTime": str,
            "Cap_Ah": float,
            "Ener_Wh": float,
            "Current_A": float,
            "Voltage_V": float,
            "Md": str,
            "ES": int,
            "DPT_Time": str,
        }
        md_col_names = [
            "Rec",
            "Cycle_P",
            "Cycle_C",
            "Step",
            "TestTime",
            "StepTime",
            "Cap_Ah",
            "Ener_Wh",
            "Current_A",
            "Voltage_V",
            "Md",
            "ES",
            "DPT_Time",
        ]
        if n_col > 13:
            md_dtype["AUX1_V"] = float
            md_col_names.append("AUX1_V")
        if n_col > 14:
            i = 2
            while (n_col - i) >= 13:
                entry = "Aux" + str(i) + "_V"
                md_col_names.append(entry)  # write constructed column name
                md_dtype.update({entry: float})
                i += 1
        # print(md_dtype, md_col_names, n_col)
        # read file to a panda dataFrame
        maccor_data = pd.read_table(
            filepath_or_buffer=file_path,
            sep="\t",
            encoding=encoding,
            header=None,
            dtype=md_dtype,
            names=md_col_names,
            usecols=range(0, n_col),
            skiprows=15,
            decimal=decimal_separator,
        )

        #        # read file to a panda dataFrame
        #        maccor_data = pd.read_table(
        #           filepath_or_buffer=mFile, sep='\t',
        #           header=n_header-2, dtype=None,
        #           usecols=range(0, n_col), decimal=maccorDecimalSep)
        #
        #        # Remove characters from columns name to apply rules of python
        #        # variable naming. Remove double '__' for better readability
        #        for letter in '[]()':
        #            maccor_data.columns = maccor_data.columns.str.replace(letter, '')
        #        for letter in '-._ ':
        #            maccor_data.columns = maccor_data.columns.str.replace(letter, '_')
        #        maccor_data.columns = maccor_data.columns.str.replace('__', '_')

    elif input_source_type == 11:
        # the file to import was exported from MaccorExport.exe v1
        n_header = 3
        with open(file_path, "r", encoding=encoding) as maccor_file:
            for x in range(1, n_header + 1):  # range(m, n) = [m, ..., n-1]
                # In the last interation, this is the header line containing column
                # names
                header_line = maccor_file.readline()

        n_col = len(header_line.split("\t"))

        if n_col == 14:
            md_dtype = {
                "Rec": int,
                "Cycle_P": int,
                "Step": int,
                "TestTime": str,
                "StepTime": str,
                "Cap_Ah": float,
                "Ener_Wh": float,
                "Current_A": float,
                "Voltage_V": float,
                "Md": str,
                "ES": int,
                "DPT_Time": str,
                "AUX1_V": float,
            }
            md_col_names = [
                "Rec",
                "Cycle_P",
                "Step",
                "TestTime",
                "StepTime",
                "Cap_Ah",
                "Ener_Wh",
                "Current_A",
                "Voltage_V",
                "Md",
                "ES",
                "DPT_Time",
                "AUX1_V",
            ]
        elif n_col == 13:
            md_dtype = {
                "Rec": int,
                "Cycle_P": int,
                "Step": int,
                "TestTime": str,
                "StepTime": str,
                "Cap_Ah": float,
                "Ener_Wh": float,
                "Current_A": float,
                "Voltage_V": float,
                "Md": str,
                "ES": int,
                "DPT_Time": str,
            }
            md_col_names = [
                "Rec",
                "Cycle_P",
                "Step",
                "TestTime",
                "StepTime",
                "Cap_Ah",
                "Ener_Wh",
                "Current_A",
                "Voltage_V",
                "Md",
                "ES",
                "DPT_Time",
            ]
        elif n_col >= 15:
            md_dtype = {
                "Rec": int,
                "Cycle_P": int,
                "Step": int,
                "TestTime": str,
                "StepTime": str,
                "Cap_Ah": float,
                "Ener_Wh": float,
                "Current_A": float,
                "Voltage_V": float,
                "Md": str,
                "ES": int,
                "DPT_Time": str,
                "AUX1_V": float,
            }
            md_col_names = [
                "Rec",
                "Cycle_P",
                "Step",
                "TestTime",
                "StepTime",
                "Cap_Ah",
                "Ener_Wh",
                "Current_A",
                "Voltage_V",
                "Md",
                "ES",
                "DPT_Time",
                "AUX1_V",
            ]
            i = 2
            while (n_col - i) >= 13:
                entry = "Aux" + str(i) + "_V"
                md_col_names.append(entry)  # write constructed column name
                md_dtype.update({entry: float})
                i += 1

        # print(md_dtype, md_col_names, n_col)
        maccor_data = pd.read_table(
            filepath_or_buffer=file_path,
            sep="\t",
            encoding=encoding,
            header=None,
            dtype=md_dtype,
            names=md_col_names,
            usecols=range(0, n_col - 1),
            skiprows=3,
            decimal=decimal_separator,
        )

    elif input_source_type == 20:
        digatron_data_obj = digatron_utility.read.read_digatron_data_file(
            file_path,
            # remove_nan_rows=False
        )
        digatron_units = digatron_data_obj.meta["Units"]
        digatron_df: pd.DataFrame = digatron_data_obj.data.as_dataframe  # Tabular data
        new_col_names = {
            "Spannung": "Voltage_V",
            "Zyklus": "Cycle_P",
            "Status": "Md",
        }
        if "Z0" in digatron_df.columns:
            new_col_names.update({"Zyklus": "Zyklus", "Z0": "Cycle_P"})
        digatron_df.rename(columns=new_col_names, inplace=True)
        digatron_df.replace(
            {
                "Md": {
                    "PAU": "R",
                    "LAD": "C",
                    "ELA": "D",
                }
            },
            inplace=True,
        )
        factor = 1
        if digatron_units["AhAkku"] == "mAh":
            factor = 1e-3
        cap_ah = []
        for row in digatron_df.index:
            if digatron_df.at[row, "Md"] == "C":
                cap_ah.append(digatron_df.at[row, "AhLad"] * factor)
            elif digatron_df.at[row, "Md"] == "D":
                cap_ah.append(digatron_df.at[row, "AhEla"] * factor)
            else:
                cap_ah.append(0)
            # if digatron_df.at[row, "AhLad"] != 0:
            #     cap_ah.append(digatron_df.at[row, "AhLad"])
            # elif digatron_df.at[row, "AhEla"] != 0:
            #     cap_ah.append(digatron_df.at[row, "AhEla"])
            # else:
            #     cap_ah.append(0)
        digatron_df.insert(6, "Cap_Ah", cap_ah)
        maccor_data = digatron_df

    else:
        raise ValueError("Unknown input type / format")

    m_length = len(maccor_data)  # this is a check-up
    # List of unique entries in column 'Cycle_P'
    cycle_list = maccor_data.Cycle_P.unique()

    # Check whether Aux voltage is anode or cathode vs. reference
    if "AUX1_V" in maccor_data.columns:
        mean_aux = maccor_data.AUX1_V.mean()
        mean_cell_volt = maccor_data.Voltage_V.mean()
        if mean_aux >= mean_cell_volt:
            # Aux is connected to cathode
            aux_con = "cathode"
        else:
            # Aux is connected to anode
            aux_con = "anode"
    else:
        three_electrode_cell = 0
        print("No auxilary voltage 'AUX1 [V]' was found in selected file!")

    # Change axis title and export to reduce leading 0s if largest charge
    # or discharge capacity is bigger or equal 1 Ah
    cap_unit = "Ah"
    if max(maccor_data.Cap_Ah) >= 1:
        cap_unit = "Ah"
    elif max(maccor_data.Cap_Ah) < 1:
        cap_unit = "mAh"
    # Same for weight
    w_unit = 0
    if active_material_mass != 0:
        if active_material_mass >= 1000:
            w_unit = "kg"
        elif active_material_mass < 1000:
            w_unit = "g"
        cap_unit_g = r"{}$\cdot${}".format(cap_unit, w_unit) + r"$^{-1}$"
        cap_unit_d = "{} {}-1".format(cap_unit, w_unit)
    elif active_material_mass == 0:
        cap_unit_g = cap_unit
        cap_unit_d = cap_unit

    # Setting smoothing parameter
    sigma = int(dq_dv_filter_sigma_or_polynomial_order)
    p_order = int(dq_dv_filter_sigma_or_polynomial_order)
    dq_dv_filter_window_size = int(dq_dv_filter_window_size)

    # Make sure export folder does exist
    if (export_analysis_data == 1) or (export_graphs == 1):
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)

    # -------------------------------------------------------------------------
    # %% First chapter: charge & discharge capacity, Coulomb efficiency vs.
    # cycle num
    # -------------------------------------------------------------------------

    # Estimate number of cycles
    num_cycle = len(cycle_list)

    # Performed even if cap_eff == 0
    # Create array or table to hold cycle count, charge & discharge
    # capacity and Coulomb efficiency. To determine its length check
    # highest cycle count in file

    # columns: 'Cycle', 'Charge Capacity / Ah', 'Discharge Capacity / Ah',
    # 'Coulombic Efficiency / %'
    cb_names = [
        "Cycle",
        "Charge capacity",
        "Discharge capacity",
        "Coulombic efficiency",
    ]  # Will later on be used as header lines in export file
    cb_units = ["#", cap_unit_d, cap_unit_d, "%"]  # Same here
    # This is the safe way to initiate this array even if first cycle is
    # counted 0
    cycle_based = np.zeros((num_cycle, 4))

    # Determine charge and discharge capacity

    # Loop over all rows starting with the second (index=1) one since we
    #  are always comparing the actual row with its predecessor
    for i in range(1, m_length):  # the last row has index m_length-1
        cycle = maccor_data.at[i - 1, "Cycle_P"]  # Cycle count
        i_cyc = np.where(cycle_list == cycle)[0][0]
        if maccor_data.at[i, "Md"] != "C" and maccor_data.at[i - 1, "Md"] == "C":
            cycle_based[i_cyc, 0] = cycle  # set cycle count
            # set charge capacity
            cycle_based[i_cyc, 1] = maccor_data.at[i - 1, "Cap_Ah"]
        elif i == m_length - 1 and maccor_data.at[i, "Md"] == "C":
            cycle_based[i_cyc, 0] = cycle  # set cycle count
            # set charge capacity
            cycle_based[i_cyc, 1] = maccor_data.at[i, "Cap_Ah"]
        # elif = else if
        elif maccor_data.at[i, "Md"] != "D" and maccor_data.at[i - 1, "Md"] == "D":
            # set discharge capacity
            cycle_based[i_cyc, 2] = maccor_data.at[i - 1, "Cap_Ah"]
        elif i == m_length - 1 and maccor_data.at[i, "Md"] == "D":
            # set discharge capacity
            cycle_based[i_cyc, 2] = maccor_data.at[i, "Cap_Ah"]

    # Coulomb efficiency = DischCap/ChCap - catch exception of zero
    #  division by where argument
    cycle_based[:, 3] = 100 * np.divide(
        cycle_based[:, 2], cycle_based[:, 1], where=cycle_based[:, 1] != 0
    )
    # Every row has to be considered, but when a row gets deleted the next
    #  row takes its index
    deletes = 0
    for row in range(0, len(cycle_based)):
        if (cycle_based[row - deletes, 1] == 0) and (
            cycle_based[row - deletes, 2] == 0
        ):
            # Charge and discharge capacity of cycle 0 are 0 --> remove row
            cycle_based = np.delete(cycle_based, row - deletes, 0)
            deletes += 1

    # Update cycle_list according to entries with a non-zero capacity in
    #  cycle_based (which got deleted already):
    cycle_list = np.unique(cycle_based[:, 0])
    ctp = convert_c2p(cycles_to_plot, cycle_list)

    len_ctp = len(ctp)
    colormap_array = mpl.colormaps.get_cmap(colormap)
    colormap = [colormap_array(i) for i in np.linspace(cm_start, cm_end, len_ctp)]

    if plot_cap_ce_vs_cn == 1:
        # Reduce leading 0s by multiplying with factor if largest charge
        # or discharge capacity is bigger or equal 1 Ah
        if cap_unit == "mAh":
            cycle_based[:, 1] = 1000 * cycle_based[:, 1]
            cycle_based[:, 2] = 1000 * cycle_based[:, 2]
        # Recalculation if weight is specified
        if active_material_mass != 0:
            cycle_based[:, 1] = cycle_based[:, 1] / active_material_mass
            cycle_based[:, 2] = cycle_based[:, 2] / active_material_mass
            if w_unit == "kg":
                cycle_based[:, 1] = cycle_based[:, 1] / 1000
                cycle_based[:, 2] = cycle_based[:, 2] / 1000

        # Export data
        export_numpy_array_with_header(
            export=export_analysis_data,
            np_array=cycle_based,
            fmt="%.1i %.8f %.8f %.3f",
            column_names=cb_names,
            units=cb_units,
            comments=None,
            index=index,
            sample_name=sample_name,
            maccor_file_name=file_name,
            analysis_type="cap_and_eff_vs_cycle",
            export_dir=export_dir,
        )

        # Plot capacity and Coulomb efficiency vs. cycle
        figure_name = str(index) + "_" + sample_name + "_" + "cap_and_eff_vs_cycle"

        if plt.fignum_exists(figure_name):
            plt.close(figure_name)

        fig = plt.figure(figure_name)
        ax1 = fig.add_subplot(111)

        color = "k"
        ax1.set_xlabel("Cycle")
        ax1.set_ylabel("Capacity / " + cap_unit_g, color=color)
        ax1.scatter(
            cycle_based[:, 0],
            cycle_based[:, 1],
            marker="^",
            color="r",
            label="Charge capacity",
        )
        ax1.scatter(
            cycle_based[:, 0],
            cycle_based[:, 2],
            marker="v",
            color="b",
            label="Discharge capacity",
        )
        ax1.tick_params(axis="y", labelcolor=color)
        # ax1.margins(y=0.1)
        # Default margin is 0.05 but in this case not suitable since data points
        # overlap
        # To catch the mAh/kg exception scales are set by hand including a
        # margin of 5 % and scientific notation is used
        if (cap_unit == "mAh") & (w_unit == "kg"):
            ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            # Join capacity lists to make search easier
            joined_cap = cycle_based[:, 1]
            joined_cap = np.append(joined_cap, cycle_based[:, 2], axis=0)
            ax1.set_ylim = (
                min(joined_cap) - 0.05 * max(joined_cap),
                1.05 * max(joined_cap),
            )
            # this command seems to have no effect

        # instantiate a second axes that shares the same x-axis
        ax2 = ax1.twinx()

        color = "g"
        ax2.set_ylabel("Coulombic efficiency / %", color=color)
        # we already handled the x-label with ax1
        ax2.scatter(
            cycle_based[:, 0],
            cycle_based[:, 3],
            marker="o",
            color=color,
            label="Coulombic efficiency",
        )
        ax2.tick_params(axis="y", labelcolor=color)
        # ax1.margins(y=0.1)
        # Default margin is 0.05 but in this case not suitable since data points
        # overlap

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        lgd = ax1.legend(
            h1 + h2,
            l1 + l2,
            bbox_to_anchor=(0, 1.05, 1, 0.105),
            loc="lower left",
            ncol=2,
            mode="expand",
            borderaxespad=0,
        )

        # Get number of legend entries to estimate needed  rectangle size
        num_leg = len(l1) + len(l2)
        extra = 0.5 * round(num_leg / 2) * 2 / 50  # 2 equals n_col
        fig.tight_layout(rect=(0, 0, 1, 1 + extra))
        # creating box around axes and legend
        # otherwise the right y-label is slightly clipped

        # Export graph
        export_graph(
            export=export_graphs,
            figure=fig,
            legend=lgd,
            index=index,
            sample_name=sample_name,
            analysis_type="cap_and_eff_vs_cycle",
            export_dir=export_dir,
        )
        if mpl.is_interactive() is True:
            if draw_graphs_on_screen == 1:
                fig.show()
            elif draw_graphs_on_screen == 0:
                plt.close(fig)  # has to be called after export

    # -------------------------------------------------------------------------
    # %% Second chapter: voltage during charge and discharge vs. capacity
    # -------------------------------------------------------------------------
    # First part until if (volt_cap == 1) has to be performed any ways because
    # dQ_dV is relying on it as well

    # Will be performed no matter if three electrode cell or not
    # filter imported Maccor data according to entry 'C' or 'D' in Col 'Md'
    cf_md = maccor_data.loc[maccor_data.Md == "C"]  # C-FilteredMaccorData
    df_md = maccor_data.loc[maccor_data.Md == "D"]  # D-FilteredMaccorData

    # estimate longest C/D- cycle to estimated required array size to hold
    # cycle data
    longest_ch = 0
    longest_dch = 0
    for cycle in cycle_list:
        len_c = len(cf_md[cf_md.Cycle_P == cycle])
        len_d = len(df_md[df_md.Cycle_P == cycle])
        if len_c >= longest_ch:
            longest_ch = len_c
        if len_d >= longest_dch:
            longest_dch = len_d
    if longest_ch > longest_dch:
        cb_size = longest_ch
    else:  # longest_ch <= longest_dch
        cb_size = longest_dch

    if (plot_volt_vs_cap == 1) or (plot_dq_dv == 1):
        # The following has to be performed also if volt_cap = 0 but dQ_dV = 1
        # create list of columns, mostly for the later generation of header
        # lines
        cv_columns = []  # Empty initialization
        cv_units = []
        cv_comments = []
        for cycle in cycle_list:
            cv_columns.append("chCap%i" % cycle)
            cv_columns.append("chVolt%i" % cycle)
            cv_columns.append("disCap%i" % cycle)
            cv_columns.append("disVolt%i" % cycle)
            for i in range(0, 2):  # Run two times
                cv_units.append(cap_unit_d)
                cv_units.append("V")
            for i in range(0, 4):  # Run four times
                cv_comments.append("Cycle %i" % cycle)

        # Columns: 'Capacity / Ah', 'Voltage' for each cycle, charge and
        # discharge separated

        cap_based = np.empty((cb_size, 4 * len(cycle_list)), dtype=float)
        cap_based[:] = np.nan

        # Distributing column entries filtered by Md and Cycle_P to separate
        # columns of array/table cap_based for later export and plotting
        for cycle in cycle_list:
            i_cyc = np.where(cycle_list == cycle)[0][0]
            #            print(i_cyc)
            len_c = len(cf_md[cf_md.Cycle_P == cycle])
            len_d = len(df_md[df_md.Cycle_P == cycle])
            cap_based[0:len_c, i_cyc * 4 + 0] = cf_md.Cap_Ah[cf_md.Cycle_P == cycle]
            cap_based[0:len_c, i_cyc * 4 + 1] = cf_md.Voltage_V[cf_md.Cycle_P == cycle]
            cap_based[0:len_d, i_cyc * 4 + 2] = df_md.Cap_Ah[df_md.Cycle_P == cycle]
            cap_based[0:len_d, i_cyc * 4 + 3] = df_md.Voltage_V[df_md.Cycle_P == cycle]

            if cap_unit == "mAh":  # recalculate for unit conformity
                cap_based[0:len_c, i_cyc * 4 + 0] = (
                    1e3 * cap_based[0:len_c, i_cyc * 4 + 0]
                )
                cap_based[0:len_d, i_cyc * 4 + 2] = (
                    1e3 * cap_based[0:len_d, i_cyc * 4 + 2]
                )

            if active_material_mass != 0:
                cap_based[0:len_c, i_cyc * 4 + 0] = (
                    cap_based[0:len_c, i_cyc * 4 + 0] / active_material_mass
                )
                cap_based[0:len_d, i_cyc * 4 + 2] = (
                    cap_based[0:len_d, i_cyc * 4 + 2] / active_material_mass
                )
                if w_unit == "kg":
                    cap_based[0:len_c, i_cyc * 4 + 0] = (
                        1e-3 * cap_based[0:len_c, i_cyc * 4 + 0]
                    )
                    cap_based[0:len_d, i_cyc * 4 + 2] = (
                        1e-3 * cap_based[0:len_d, i_cyc * 4 + 2]
                    )

    if ((plot_volt_vs_cap == 1) or (plot_dq_dv == 1)) and (three_electrode_cell == 1):
        # The following has to be performed also if volt_cap = 0 but dQ_dV = 1
        # create list of columns, mostly for the later generation of header
        # lines
        cv_columns3 = []  # Empty initialization
        cv_units3 = []
        cv_comments3 = []
        for cycle in cycle_list:
            cv_columns3.append("chCap%i" % cycle)
            cv_columns3.append("chVoltA%i" % cycle)
            cv_columns3.append("chVoltC%i" % cycle)
            cv_columns3.append("disCap%i" % cycle)
            cv_columns3.append("disVoltA%i" % cycle)
            cv_columns3.append("disVoltC%i" % cycle)
            for i in range(0, 2):  # Run two times
                cv_units3.append(cap_unit_d)
                cv_units3.append("V")
                cv_units3.append("V")
            for i in range(0, 6):  # Run four times
                cv_comments3.append("Cycle %i" % cycle)

        # Columns: 'Capacity / Ah', 'Voltage' for each cycle, charge and
        # discharge separated

        cap_based3 = np.empty((cb_size, 6 * len(cycle_list)), dtype=float)
        cap_based3[:] = np.nan

        # Distributing column entries filtered by Md and Cycle_P to separate
        # columns of array/table cap_based for later export and plotting
        for cycle in cycle_list:
            i_cyc = np.where(cycle_list == cycle)[0][0]
            len_c = len(cf_md[cf_md.Cycle_P == cycle])
            len_d = len(df_md[df_md.Cycle_P == cycle])
            # Charge
            cap_based3[0:len_c, i_cyc * 6 + 0] = cf_md.Cap_Ah[cf_md.Cycle_P == cycle]
            # Anode voltage
            if aux_con == "anode":
                cap_based3[0:len_c, i_cyc * 6 + 1] = (cf_md.AUX1_V)[
                    cf_md.Cycle_P == cycle
                ]
            else:  # aux_con == 'cathode'
                cap_based3[0:len_c, i_cyc * 6 + 1] = (
                    cf_md.AUX1_V[cf_md.Cycle_P == cycle]
                    - cf_md.Voltage_V[cf_md.Cycle_P == cycle]
                )
            # Cathode voltage
            if aux_con == "anode":
                cap_based3[0:len_c, i_cyc * 6 + 2] = (
                    cf_md.AUX1_V[cf_md.Cycle_P == cycle]
                    + cf_md.Voltage_V[cf_md.Cycle_P == cycle]
                )
            else:  # aux_con == 'cathode'
                cap_based3[0:len_c, i_cyc * 6 + 2] = cf_md.AUX1_V[
                    cf_md.Cycle_P == cycle
                ]

            # Discharge
            cap_based3[0:len_d, i_cyc * 6 + 3] = df_md.Cap_Ah[df_md.Cycle_P == cycle]
            # Anode voltage
            if aux_con == "anode":
                cap_based3[0:len_d, i_cyc * 6 + 4] = df_md.AUX1_V[
                    df_md.Cycle_P == cycle
                ]
            else:  # aux_con == 'cathode'
                cap_based3[0:len_d, i_cyc * 6 + 4] = (
                    df_md.AUX1_V[df_md.Cycle_P == cycle]
                    - df_md.Voltage_V[df_md.Cycle_P == cycle]
                )
            # Cathode voltage
            if aux_con == "anode":
                cap_based3[0:len_d, i_cyc * 6 + 5] = (
                    df_md.AUX1_V[df_md.Cycle_P == cycle]
                    + df_md.Voltage_V[df_md.Cycle_P == cycle]
                )
            else:  # aux_con == 'cathode'
                cap_based3[0:len_d, i_cyc * 6 + 5] = df_md.AUX1_V[
                    df_md.Cycle_P == cycle
                ]

            if cap_unit == "mAh":  # recalculate for unit conformity
                cap_based3[0:len_c, i_cyc * 6 + 0] = (
                    1e3 * cap_based3[0:len_c, i_cyc * 6 + 0]
                )
                cap_based3[0:len_d, i_cyc * 6 + 3] = (
                    1e3 * cap_based3[0:len_d, i_cyc * 6 + 3]
                )

            if active_material_mass != 0:
                cap_based3[0:len_c, i_cyc * 6 + 0] = (
                    cap_based3[0:len_c, i_cyc * 6 + 0] / active_material_mass
                )
                cap_based3[0:len_d, i_cyc * 6 + 3] = (
                    cap_based3[0:len_d, i_cyc * 6 + 3] / active_material_mass
                )
                if w_unit == "kg":
                    cap_based3[0:len_c, i_cyc * 6 + 0] = (
                        1e-3 * cap_based3[0:len_c, i_cyc * 6 + 0]
                    )
                    cap_based3[0:len_d, i_cyc * 6 + 3] = (
                        1e-3 * cap_based3[0:len_d, i_cyc * 6 + 3]
                    )

    if plot_volt_vs_cap == 1:
        # Export data
        export_numpy_array_with_header(
            export=export_analysis_data,
            np_array=cap_based,
            fmt="%.8f",
            column_names=cv_columns,
            units=cv_units,
            comments=cv_comments,
            index=index,
            sample_name=sample_name,
            maccor_file_name=file_name,
            analysis_type="volt_vs_cap",
            export_dir=export_dir,
        )

        # Plot Voltage vs. Capacity
        figure_name = str(index) + "_" + sample_name + "_" + "volt_vs_cap"
        # colormaps can be chosen from:
        # https://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
        # colormap = plt.cm.gist_rainbow

        fig, lgd = plot_cycle_wise_single_scale(
            figure_name,
            colormap,
            ctp,
            cycle_list,
            array=cap_based,
            n_col=4,
            x1_col=0,
            y1_col=1,
            x2_col=2,
            y2_col=3,
            comments=cv_comments,
            x_title="Capacity / " + cap_unit_g,
            y_title="Voltage / V",
            x_limits=v_c_x_limits,
            y_limits=v_c_y_limits,
            cap_unit=cap_unit,
            active_material_mass_unit=w_unit,
        )

        # Export graph
        export_graph(
            export=export_graphs,
            figure=fig,
            legend=lgd,
            index=index,
            sample_name=sample_name,
            analysis_type="volt_vs_cap",
            export_dir=export_dir,
        )

        if mpl.is_interactive() is True:
            if draw_graphs_on_screen == 1:
                fig.show()
            elif draw_graphs_on_screen == 0:
                plt.close(fig)  # has to be called after export

    if (plot_volt_vs_cap == 1) and (three_electrode_cell == 1):
        # Export data
        export_numpy_array_with_header(
            export=export_analysis_data,
            np_array=cap_based3,
            fmt="%.8f",
            column_names=cv_columns3,
            units=cv_units3,
            comments=cv_comments3,
            index=index,
            sample_name=sample_name,
            maccor_file_name=file_name,
            analysis_type="volt_vs_cap_3el",
            export_dir=export_dir,
        )

        # Plot Voltage vs. Capacity

        # colormaps can be chosen from:
        # https://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
        # colormap = plt.cm.gist_rainbow

        # Estimate y_lim for the two y-axes from the data

        # Anode
        if v_c_y1_limits == "auto":
            # Anode voltage
            if aux_con == "anode":
                anode_ch_cycles = cf_md.AUX1_V
                anode_dis_cycles = df_md.AUX1_V
            else:  # aux_con == 'cathode'
                anode_ch_cycles = cf_md.AUX1_V - cf_md.Voltage_V
                anode_dis_cycles = df_md.AUX1_V - df_md.Voltage_V
            anode_max = max([max(anode_ch_cycles), max(anode_dis_cycles)])
            anode_min = min([min(anode_ch_cycles), min(anode_dis_cycles)])
            # Do not forget to include margin
            anode_min -= 0.05 * (anode_max - anode_min)
            anode_max += 0.05 * (anode_max - anode_min)
            vc_y1_limits_here = [anode_min, anode_max]
        else:
            vc_y1_limits_here = v_c_y1_limits

        # Cathode
        if v_c_y2_limits == "auto":
            # Cathode voltage
            if aux_con == "anode":
                cathode_ch_cycles = cf_md.AUX1_V + cf_md.Voltage_V
                cathode_dis_cycles = df_md.AUX1_V + df_md.Voltage_V
            else:  # aux_con == 'cathode'
                cathode_ch_cycles = cf_md.AUX1_V
                cathode_dis_cycles = df_md.AUX1_V
            cathode_max = max([max(cathode_ch_cycles), max(cathode_dis_cycles)])
            cathode_min = min([min(cathode_ch_cycles), min(cathode_dis_cycles)])
            # Do not forget to include margin
            cathode_min -= 0.05 * (cathode_max - cathode_min)
            cathode_max += 0.05 * (cathode_max - cathode_min)
            vc_y2_limits_here = [cathode_min, cathode_max]
        else:
            vc_y2_limits_here = v_c_y2_limits

        #        vc_y1_limits_here = 'auto'
        #        vc_y2_limits_here = 'auto'
        # Charge cycle
        figure_name = str(index) + "_" + sample_name + "_" + "volt_vs_cap_3el_ch"

        fig, lgd, x_lim = plot_cycle_wise_dual_scale(
            figure_name,
            colormap,
            ctp,
            cycle_list,
            array=cap_based3,
            n_col=6,
            x1_col=0,
            y1_col=1,
            y2_col=2,
            comments=cv_comments3,
            x_title="Charge capacity / " + cap_unit_g,
            y1_title="Anode vs. ref. / V ",
            y2_title="Cathode vs. ref. / V",
            x_limits=v_c_x_limits,
            y1_limits=vc_y1_limits_here,
            y2_limits=vc_y2_limits_here,
            cap_unit=cap_unit,
            active_material_mass_unit=w_unit,
        )

        # Export graph
        export_graph(
            export=export_graphs,
            figure=fig,
            legend=lgd,
            index=index,
            sample_name=sample_name,
            analysis_type="volt_vs_cap_3el_ch",
            export_dir=export_dir,
        )

        if mpl.is_interactive() is True:
            if draw_graphs_on_screen == 1:
                fig.show()
            elif draw_graphs_on_screen == 0:
                plt.close(fig)  # has to be called after export

        # Discharge cycle
        figure_name = str(index) + "_" + sample_name + "_" + "volt_vs_cap_3el_dis"

        if v_c_x_limits == "auto":
            # set discharge graphs x-scale to x-scale of charge cycle
            v_c_x_limits_here = x_lim
        else:
            v_c_x_limits_here = v_c_x_limits

        fig, lgd, x_lim = plot_cycle_wise_dual_scale(
            figure_name,
            colormap,
            ctp,
            cycle_list,
            array=cap_based3,
            n_col=6,
            x1_col=3,
            y1_col=4,
            y2_col=5,
            comments=cv_comments3,
            x_title="Discharge capacity / " + cap_unit_g,
            y1_title="Anode vs. ref. / V ",
            y2_title="Cathode vs. ref. / V",
            x_limits=v_c_x_limits_here,
            y1_limits=vc_y1_limits_here,
            y2_limits=vc_y2_limits_here,
            cap_unit=cap_unit,
            active_material_mass_unit=w_unit,
        )

        # Export graph
        export_graph(
            export=export_graphs,
            figure=fig,
            legend=lgd,
            index=index,
            sample_name=sample_name,
            analysis_type="volt_vs_cap_3el_dis",
            export_dir=export_dir,
        )

        if mpl.is_interactive() is True:
            if draw_graphs_on_screen == 1:
                fig.show()
            elif draw_graphs_on_screen == 0:
                plt.close(fig)  # has to be called after export

    # -------------------------------------------------------------------------
    # %% Third chapter: differential voltage and capacity analysis
    # -------------------------------------------------------------------------
    dv_min_increment *= 0.001  # minimum voltage increment for dQ/dV calculation
    # passed as mV, converted to V !!!
    if plot_dq_dv == 1:
        vb_size = cb_size

        # create list of columns, mostly for the later generation of header
        # lines
        vb_columns = []  # empty initialization
        vb_units = []
        vb_comments = []

        for cycle in cycle_list:
            vb_columns.append("chVolt%i" % cycle)
            vb_columns.append("chCap%i" % cycle)
            vb_columns.append("chdQdV%i" % cycle)
            vb_columns.append("disVolt%i" % cycle)
            vb_columns.append("disCap%i" % cycle)
            vb_columns.append("disdQdV%i" % cycle)
            for i in range(0, 2):  # run two times
                vb_units.append("V")
                vb_units.append(cap_unit_d)
                vb_units.append(cap_unit_d + " V-1")
            for i in range(0, 6):  # run 6 times
                vb_comments.append("Cycle %i" % cycle)

        #        v_based = np.empty((vb_size, 6*num_cycle), dtype=float)
        #        v_based[:] = np.nan

        v_based_f = np.empty((vb_size, 6 * num_cycle), dtype=float)
        v_based_f[:] = np.nan

        for cycle in cycle_list:
            # Index of element in cycle_list == cycle
            i_cyc = np.where(cycle_list == cycle)[0][0]
            len_c = len(cf_md[cf_md.Cycle_P == cycle])
            len_d = len(df_md[df_md.Cycle_P == cycle])

            # Charge cycle
            v_based_f[0, i_cyc * 6 + 0] = cap_based[0, i_cyc * 4 + 1]
            v_based_f[0, i_cyc * 6 + 1] = cap_based[0, i_cyc * 4 + 0]

            v_row_ch = 1
            for row in range(1, len_c):
                dv_ch = abs(
                    cap_based[row, i_cyc * 4 + 1]
                    - v_based_f[v_row_ch - 1, i_cyc * 6 + 0]
                )
                if dv_ch >= dv_min_increment:
                    # Voltage
                    v_based_f[v_row_ch, i_cyc * 6 + 0] = cap_based[row, i_cyc * 4 + 1]
                    # Capacity
                    v_based_f[v_row_ch, i_cyc * 6 + 1] = cap_based[row, i_cyc * 4 + 0]
                    v_row_ch += 1  # increment v_row_ch by 1

            # Discharge cycle
            v_based_f[0, i_cyc * 6 + 3] = cap_based[0, i_cyc * 4 + 3]
            v_based_f[0, i_cyc * 6 + 4] = cap_based[0, i_cyc * 4 + 2]

            v_row_dis = 1
            for row in range(1, len_d):
                dv_dis = abs(
                    cap_based[row, i_cyc * 4 + 3]
                    - v_based_f[v_row_dis - 1, i_cyc * 6 + 3]
                )
                if dv_dis >= dv_min_increment:
                    # Voltage
                    v_based_f[v_row_dis, i_cyc * 6 + 3] = cap_based[row, i_cyc * 4 + 3]
                    # Capacity
                    v_based_f[v_row_dis, i_cyc * 6 + 4] = cap_based[row, i_cyc * 4 + 2]
                    v_row_dis += 1  # increment v_row_ch by 1

            v_based = v_based_f  # lazy to rename all instance, so reassign

            # without smoothing
            diff_vc = np.diff(v_based[0:len_c, i_cyc * 6 + 0])
            diff_cc = np.diff(v_based[0:len_c, i_cyc * 6 + 1])
            diff_vd = np.diff(v_based[0:len_d, i_cyc * 6 + 3])
            diff_cd = np.diff(v_based[0:len_d, i_cyc * 6 + 4])

            # dQ/dV of charge cycle
            if len_c != 0:
                v_based[0 : len_c - 1, i_cyc * 6 + 2] = np.divide(
                    diff_cc, diff_vc, out=np.zeros((len_c - 1,)), where=diff_vc != 0
                )
            # dQ/dV of discharge cycle
            if len_d != 0:
                v_based[0 : len_d - 1, i_cyc * 6 + 5] = np.divide(
                    diff_cd, diff_vd, out=np.zeros((len_d - 1,)), where=diff_vd != 0
                )

            # Smooth curve for better visual experience
            # Charge cycle
            v_based[0 : len_c - 1, i_cyc * 6 + 2] = smooth_methods(
                selection=dq_dv_filter_selection,
                input_array=v_based[0 : len_c - 1, i_cyc * 6 + 2],
                filter_window=dq_dv_filter_window_size,
                sigma=sigma,
                poly_order=p_order,
            )
            # Discharge cycle
            v_based[0 : len_d - 1, i_cyc * 6 + 5] = smooth_methods(
                selection=dq_dv_filter_selection,
                input_array=v_based[0 : len_d - 1, i_cyc * 6 + 5],
                filter_window=dq_dv_filter_window_size,
                sigma=sigma,
                poly_order=p_order,
            )

        # Export data
        export_numpy_array_with_header(
            export=export_analysis_data,
            np_array=v_based,
            fmt="%.8f",
            column_names=vb_columns,
            units=vb_units,
            comments=vb_comments,
            index=index,
            sample_name=sample_name,
            maccor_file_name=file_name,
            analysis_type="dQ_dV",
            export_dir=export_dir,
        )

        # Plot dQ/dV
        figure_name = str(index) + "_" + sample_name + "_" + "dQ_dV"
        # colormaps can be chosen from:
        # https://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
        # colormap = plt.cm.gist_rainbow

        fig, lgd = plot_cycle_wise_single_scale(
            figure_name,
            colormap,
            ctp,
            cycle_list,
            array=v_based,
            n_col=6,
            x1_col=0,
            y1_col=2,
            x2_col=3,
            y2_col=5,
            comments=vb_comments,
            x_title="Voltage / V",
            y_title=("dQ/dV / " + cap_unit_g + r"$\cdot$V$^{-1}$"),
            x_limits=dq_dv_x_limits,
            y_limits=dq_dv_y_limits,
            cap_unit=cap_unit,
            active_material_mass_unit=w_unit,
        )

        # Export graph
        export_graph(
            export=export_graphs,
            figure=fig,
            legend=lgd,
            index=index,
            sample_name=sample_name,
            analysis_type="dQ_dV",
            export_dir=export_dir,
        )

        if mpl.is_interactive() is True:
            if draw_graphs_on_screen == 1:
                fig.show()
            elif draw_graphs_on_screen == 0:
                plt.close(fig)  # has to be called after export

    if (plot_dq_dv == 1) and (three_electrode_cell == 1):
        vb_size = cb_size

        # create list of columns, mostly for the later generation of header
        # lines
        vb_columns3 = []  # empty initialization
        vb_units3 = []
        vb_comments3 = []

        v_based_f3 = np.empty((vb_size, 12 * num_cycle), dtype=float)
        v_based_f3[:] = np.nan

        # Split dQ/dV array for separate export of anode and cathode
        # Anode
        vb_columns3_a = []  # empty initialization, half the size than above
        vb_units3_a = []
        vb_comments3_a = []
        # Cathode
        vb_columns3_c = []  # empty initialization, half the size than above
        vb_units3_c = []
        vb_comments3_c = []

        # Anode
        v_based_f3_a = np.empty((vb_size, 6 * num_cycle), dtype=float)
        v_based_f3_a[:] = np.nan
        # Cathode
        v_based_f3_c = np.empty((vb_size, 6 * num_cycle), dtype=float)
        v_based_f3_c[:] = np.nan

        # Fill lists

        for cycle in cycle_list:
            # Three electrode all
            vb_columns3.append("AchVolt%i" % cycle)
            vb_columns3.append("AchCap%i" % cycle)
            vb_columns3.append("AchdQdV%i" % cycle)
            vb_columns3.append("CchVolt%i" % cycle)
            vb_columns3.append("CchCap%i" % cycle)
            vb_columns3.append("CchdQdV%i" % cycle)
            vb_columns3.append("AdisVolt%i" % cycle)
            vb_columns3.append("AdisCap%i" % cycle)
            vb_columns3.append("AdisdQdV%i" % cycle)
            vb_columns3.append("CdisVolt%i" % cycle)
            vb_columns3.append("CdisCap%i" % cycle)
            vb_columns3.append("CdisdQdV%i" % cycle)
            # Anode
            vb_columns3_a.append("AchVolt%i" % cycle)
            vb_columns3_a.append("AchCap%i" % cycle)
            vb_columns3_a.append("AchdQdV%i" % cycle)
            vb_columns3_a.append("AdisVolt%i" % cycle)
            vb_columns3_a.append("AdisCap%i" % cycle)
            vb_columns3_a.append("AdisdQdV%i" % cycle)
            # Cathode
            vb_columns3_c.append("CchVolt%i" % cycle)
            vb_columns3_c.append("CchCap%i" % cycle)
            vb_columns3_c.append("CchdQdV%i" % cycle)
            vb_columns3_c.append("CdisVolt%i" % cycle)
            vb_columns3_c.append("CdisCap%i" % cycle)
            vb_columns3_c.append("CdisdQdV%i" % cycle)
            for i in range(0, 2):  # run two times
                # Three electrode all
                vb_units3.append("V")
                vb_units3.append(cap_unit_d)
                vb_units3.append(cap_unit_d + " V-1")
                vb_units3.append("V")
                vb_units3.append(cap_unit_d)
                vb_units3.append(cap_unit_d + " V-1")
                # Anode
                vb_units3_a.append("V")
                vb_units3_a.append(cap_unit_d)
                vb_units3_a.append(cap_unit_d + " V-1")
                # Cathode
                vb_units3_c.append("V")
                vb_units3_c.append(cap_unit_d)
                vb_units3_c.append(cap_unit_d + " V-1")
            for i in range(0, 12):  # run 12 times
                # Three electrode all
                vb_comments3.append("Cycle %i" % cycle)
            for i in range(0, 6):  # run 6 times
                # Anode
                vb_comments3_a.append("Cycle %i" % cycle)
                # Cathode
                vb_comments3_c.append("Cycle %i" % cycle)

        for cycle in cycle_list:
            # Index of element in cycle_list == cycle
            i_cyc = np.where(cycle_list == cycle)[0][0]
            len_c = len(cf_md[cf_md.Cycle_P == cycle])
            len_d = len(df_md[df_md.Cycle_P == cycle])

            # Charge cycle
            v_based_f3[0, i_cyc * 12 + 0] = cap_based3[
                0, i_cyc * 6 + 1
            ]  # Anode Voltage
            v_based_f3[0, i_cyc * 12 + 1] = cap_based3[
                0, i_cyc * 6 + 0
            ]  # Capacity for A
            v_based_f3[0, i_cyc * 12 + 3] = cap_based3[
                0, i_cyc * 6 + 2
            ]  # Cathode Voltage
            v_based_f3[0, i_cyc * 12 + 4] = cap_based3[
                0, i_cyc * 6 + 0
            ]  # Capacity for C

            a_v_row_ch = 1
            c_v_row_ch = 1
            for row in range(1, len_c):
                d_a_v_ch = abs(
                    cap_based3[row, i_cyc * 6 + 1]  # Anode Voltage
                    - v_based_f3[a_v_row_ch - 1, i_cyc * 12 + 0]
                )
                d_c_v_ch = abs(
                    cap_based3[row, i_cyc * 6 + 2]  # Cathode Voltage
                    - v_based_f3[c_v_row_ch - 1, i_cyc * 12 + 3]
                )
                if d_a_v_ch >= dv_min_increment:
                    # Anode Voltage
                    v_based_f3[a_v_row_ch, i_cyc * 12 + 0] = cap_based3[
                        row, i_cyc * 6 + 1
                    ]
                    # Capacity for Anode
                    v_based_f3[a_v_row_ch, i_cyc * 12 + 1] = cap_based3[
                        row, i_cyc * 6 + 0
                    ]
                    a_v_row_ch += 1  # increment by 1
                if d_c_v_ch >= dv_min_increment:
                    # Cathode Voltage
                    v_based_f3[c_v_row_ch, i_cyc * 12 + 3] = cap_based3[
                        row, i_cyc * 6 + 2
                    ]
                    # Capacity for Cathode
                    v_based_f3[c_v_row_ch, i_cyc * 12 + 4] = cap_based3[
                        row, i_cyc * 6 + 0
                    ]
                    c_v_row_ch += 1  # increment by 1

            # Discharge cycle
            v_based_f3[0, i_cyc * 12 + 6] = cap_based3[
                0, i_cyc * 6 + 4
            ]  # Anode Voltage
            v_based_f3[0, i_cyc * 12 + 7] = cap_based3[
                0, i_cyc * 6 + 3
            ]  # Capacity for A
            v_based_f3[0, i_cyc * 12 + 9] = cap_based3[
                0, i_cyc * 6 + 5
            ]  # Cathode Voltage
            v_based_f3[0, i_cyc * 12 + 10] = cap_based3[
                0, i_cyc * 6 + 3
            ]  # Capacity for C

            a_v_row_dch = 1
            c_v_row_dch = 1
            for row in range(1, len_d):
                d_a_v_dch = abs(
                    cap_based3[row, i_cyc * 6 + 4]  # Anode Voltage
                    - v_based_f3[a_v_row_dch - 1, i_cyc * 12 + 6]
                )
                d_c_v_dch = abs(
                    cap_based3[row, i_cyc * 6 + 5]  # Cathode Voltage
                    - v_based_f3[c_v_row_dch - 1, i_cyc * 12 + 9]
                )
                if d_a_v_dch >= dv_min_increment:
                    # Anode Voltage
                    v_based_f3[a_v_row_dch, i_cyc * 12 + 6] = cap_based3[
                        row, i_cyc * 6 + 4
                    ]
                    # Capacity for Anode
                    v_based_f3[a_v_row_dch, i_cyc * 12 + 7] = cap_based3[
                        row, i_cyc * 6 + 3
                    ]
                    a_v_row_dch += 1  # increment by 1
                if d_c_v_dch >= dv_min_increment:
                    # Cathode Voltage
                    v_based_f3[c_v_row_dch, i_cyc * 12 + 9] = cap_based3[
                        row, i_cyc * 6 + 5
                    ]
                    # Capacity for Cathode
                    v_based_f3[c_v_row_dch, i_cyc * 12 + 10] = cap_based3[
                        row, i_cyc * 6 + 3
                    ]
                    c_v_row_dch += 1  # increment by 1

            v_based3 = v_based_f3  # lazy to rename all instance, so reassign

            # without smoothing
            diff_a_v_ch = np.diff(v_based3[0:len_c, i_cyc * 12 + 0])
            diff_a_c_ch = np.diff(v_based3[0:len_c, i_cyc * 12 + 1])
            diff_c_v_ch = np.diff(v_based3[0:len_c, i_cyc * 12 + 3])
            diff_c_c_ch = np.diff(v_based3[0:len_c, i_cyc * 12 + 4])
            diff_a_v_dcg = np.diff(v_based3[0:len_d, i_cyc * 12 + 6])
            diff_a_c_dch = np.diff(v_based3[0:len_d, i_cyc * 12 + 7])
            diff_c_v_dch = np.diff(v_based3[0:len_d, i_cyc * 12 + 9])
            diff_c_c_dch = np.diff(v_based3[0:len_d, i_cyc * 12 + 10])

            # dQ/dV of charge cycle
            if len_c != 0:
                # Anode
                v_based3[0 : len_c - 1, i_cyc * 12 + 2] = np.divide(
                    diff_a_c_ch,
                    diff_a_v_ch,
                    out=np.zeros((len_c - 1,)),
                    where=diff_a_v_ch != 0,
                )
                # Cathode
                v_based3[0 : len_c - 1, i_cyc * 12 + 5] = np.divide(
                    diff_c_c_ch,
                    diff_c_v_ch,
                    out=np.zeros((len_c - 1,)),
                    where=diff_c_v_ch != 0,
                )

            # dQ/dV of discharge cycle
            if len_d != 0:
                # Anode
                v_based3[0 : len_d - 1, i_cyc * 12 + 8] = np.divide(
                    diff_a_c_dch,
                    diff_a_v_dcg,
                    out=np.zeros((len_d - 1,)),
                    where=diff_a_v_dcg != 0,
                )
                # Cathode
                v_based3[0 : len_d - 1, i_cyc * 12 + 11] = np.divide(
                    diff_c_c_dch,
                    diff_c_v_dch,
                    out=np.zeros((len_d - 1,)),
                    where=diff_c_v_dch != 0,
                )

            # Smooth curve for better visual experience
            # Charge cycle
            # Anode
            v_based3[0 : len_c - 1, i_cyc * 12 + 2] = smooth_methods(
                selection=dq_dv_filter_selection,
                input_array=v_based3[0 : len_c - 1, i_cyc * 12 + 2],
                filter_window=dq_dv_filter_window_size,
                sigma=sigma,
                poly_order=p_order,
            )
            # Cathode
            v_based3[0 : len_c - 1, i_cyc * 12 + 5] = smooth_methods(
                selection=dq_dv_filter_selection,
                input_array=v_based3[0 : len_c - 1, i_cyc * 12 + 5],
                filter_window=dq_dv_filter_window_size,
                sigma=sigma,
                poly_order=p_order,
            )
            # Discharge cycle
            # Anode
            v_based3[0 : len_d - 1, i_cyc * 12 + 8] = smooth_methods(
                selection=dq_dv_filter_selection,
                input_array=v_based3[0 : len_d - 1, i_cyc * 12 + 8],
                filter_window=dq_dv_filter_window_size,
                sigma=sigma,
                poly_order=p_order,
            )
            # Cathode
            v_based3[0 : len_d - 1, i_cyc * 12 + 11] = smooth_methods(
                selection=dq_dv_filter_selection,
                input_array=v_based3[0 : len_d - 1, i_cyc * 12 + 11],
                filter_window=dq_dv_filter_window_size,
                sigma=sigma,
                poly_order=p_order,
            )

            # Split dQ/dV array for separate export of anode and cathode
            # Anode
            v_based_f3_a[0:len_c, i_cyc * 6 + 0] = v_based3[0:len_c, i_cyc * 12 + 0]
            v_based_f3_a[0:len_c, i_cyc * 6 + 2] = v_based3[0:len_c, i_cyc * 12 + 1]
            v_based_f3_a[0 : len_c - 1, i_cyc * 6 + 2] = v_based3[
                0 : len_c - 1, i_cyc * 12 + 2
            ]
            v_based_f3_a[0:len_d, i_cyc * 6 + 3] = v_based_f3[0:len_d, i_cyc * 12 + 6]
            v_based_f3_a[0:len_d, i_cyc * 6 + 4] = v_based_f3[0:len_d, i_cyc * 12 + 7]
            v_based_f3_a[0 : len_d - 1, i_cyc * 6 + 5] = v_based3[
                0 : len_d - 1, i_cyc * 12 + 8
            ]
            # Cathode
            v_based_f3_c[0:len_c, i_cyc * 6 + 0] = v_based3[0:len_c, i_cyc * 12 + 3]
            v_based_f3_c[0:len_c, i_cyc * 6 + 1] = v_based3[0:len_c, i_cyc * 12 + 4]
            v_based_f3_c[0 : len_c - 1, i_cyc * 6 + 2] = v_based3[
                0 : len_c - 1, i_cyc * 12 + 5
            ]
            v_based_f3_c[0:len_d, i_cyc * 6 + 3] = v_based_f3[0:len_d, i_cyc * 12 + 9]
            v_based_f3_c[0:len_d, i_cyc * 6 + 4] = v_based_f3[0:len_d, i_cyc * 12 + 10]
            v_based_f3_c[0 : len_d - 1, i_cyc * 6 + 5] = v_based3[
                0 : len_d - 1, i_cyc * 12 + 11
            ]

        # Export data
        export_numpy_array_with_header(
            export=export_analysis_data,
            np_array=v_based3,
            fmt="%.8f",
            column_names=vb_columns3,
            units=vb_units3,
            comments=vb_comments3,
            index=index,
            sample_name=sample_name,
            maccor_file_name=file_name,
            analysis_type="dQ_dV_3el",
            export_dir=export_dir,
        )

        export_numpy_array_with_header(
            export=export_analysis_data,
            np_array=v_based_f3_a,
            fmt="%.8f",
            column_names=vb_columns3_a,
            units=vb_units3_a,
            comments=vb_comments3_a,
            index=index,
            sample_name=sample_name,
            maccor_file_name=file_name,
            analysis_type="dQ_dV_3el_anode",
            export_dir=export_dir,
        )

        export_numpy_array_with_header(
            export=export_analysis_data,
            np_array=v_based_f3_c,
            fmt="%.8f",
            column_names=vb_columns3_c,
            units=vb_units3_c,
            comments=vb_comments3_c,
            index=index,
            sample_name=sample_name,
            maccor_file_name=file_name,
            analysis_type="dQ_dV_3el_cathode",
            export_dir=export_dir,
        )

        plot_dq_dv_single_electrodes = 1
        if plot_dq_dv_single_electrodes == 1:
            # Plot dQ/dV Anode
            figure_name = str(index) + "_" + sample_name + "_" + "dQ_dV_3el_anode"

            fig, lgd = plot_cycle_wise_single_scale(
                figure_name,
                colormap,
                ctp,
                cycle_list,
                array=v_based_f3_a,
                n_col=6,
                x1_col=0,
                y1_col=2,
                x2_col=3,
                y2_col=5,
                comments=vb_comments,
                x_title="Voltage vs. ref. / V",
                y_title=("dQ/dV / " + cap_unit_g + r"$\cdot$V$^{-1}$"),
                x_limits=dq_dv_x_limits,
                y_limits=dq_dv_y_limits,
                cap_unit=cap_unit,
                active_material_mass_unit=w_unit,
            )
            # Export graph
            export_graph(
                export=export_graphs,
                figure=fig,
                legend=lgd,
                index=index,
                sample_name=sample_name,
                analysis_type="dQ_dV_3el_anode",
                export_dir=export_dir,
            )

            if mpl.is_interactive() is True:
                if draw_graphs_on_screen == 1:
                    fig.show()
                elif draw_graphs_on_screen == 0:
                    plt.close(fig)  # has to be called after export

            # Plot dQ/dV Cathode
            figure_name = str(index) + "_" + sample_name + "_" + "dQ_dV_3el_cathode"

            fig, lgd = plot_cycle_wise_single_scale(
                figure_name,
                colormap,
                ctp,
                cycle_list,
                array=v_based_f3_c,
                n_col=6,
                x1_col=0,
                y1_col=2,
                x2_col=3,
                y2_col=5,
                comments=vb_comments,
                x_title="Voltage vs. ref. / V",
                y_title=("dQ/dV / " + cap_unit_g + r"$\cdot$V$^{-1}$"),
                x_limits=dq_dv_x_limits,
                y_limits=dq_dv_y_limits,
                cap_unit=cap_unit,
                active_material_mass_unit=w_unit,
            )

            # Export graph
            export_graph(
                export=export_graphs,
                figure=fig,
                legend=lgd,
                index=index,
                sample_name=sample_name,
                analysis_type="dQ_dV_3el_cathode",
                export_dir=export_dir,
            )
            # Ende of if plot_dq_dv_single_electrodes == 1...

            if mpl.is_interactive() is True:
                if draw_graphs_on_screen == 1:
                    fig.show()
                elif draw_graphs_on_screen == 0:
                    plt.close(fig)  # has to be called after export

        # Plot dQ/dV
        figure_name = str(index) + "_" + sample_name + "_" + "dQ_dV_3el"

        if plt.fignum_exists(figure_name):
            plt.close(figure_name)

        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, num=figure_name)

        colors = colormap

        for cycle in ctp:
            i_cyc = np.where(cycle_list == cycle)[0][0]
            i_ctp = np.where(ctp == cycle)[0][0]
            # plot of charge
            # Anode
            ax1.plot(
                v_based_f3[:, i_cyc * 12 + 0],
                v_based_f3[:, i_cyc * 12 + 2],
                color=colors[i_ctp],
                label=vb_comments3[i_cyc * 12 + 2],
            )
            # Cathode
            ax2.plot(
                v_based_f3[:, i_cyc * 12 + 3],
                v_based_f3[:, i_cyc * 12 + 5],
                color=colors[i_ctp],
                label="_nolegend_",
            )
            # plot of discharge
            # Anode
            ax1.plot(
                v_based_f3[:, i_cyc * 12 + 6],
                v_based_f3[:, i_cyc * 12 + 8],
                color=colors[i_ctp],
                label="_nolegend_",
            )
            # Cathode
            ax2.plot(
                v_based_f3[:, i_cyc * 12 + 9],
                v_based_f3[:, i_cyc * 12 + 11],
                color=colors[i_ctp],
                label="_nolegend_",
            )

        # hide the spines between ax and ax2
        ax1.spines["right"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax1.yaxis.tick_left()
        ax1.tick_params(labeltop=False)  # don't put tick labels at the top
        ax2.yaxis.tick_right()
        # Make the spacing between the two axes a bit smaller
        plt.subplots_adjust(wspace=0.15)
        d = 0.015  # how big to make the diagonal lines in axes coordinates
        # arguments to pass plot, just so we don't keep repeating them
        kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False, lw=0.75)
        ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-left diagonal
        ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal

        kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
        ax2.plot((-d, d), (-d, +d), **kwargs)  # top-right diagonal
        ax2.plot((-d, d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

        x_title = "Voltage vs. ref. / V"
        y_title = "dQ/dV / " + cap_unit_g + r"$\cdot$V$^{-1}$"

        ax1.set_xlabel(x_title, fontdict=None, labelpad=None)
        ax2.set_xlabel(x_title, fontdict=None, labelpad=None)
        # instead: use a text
        # fig.text(0.5, 0.0, x_title, ha='center')

        ax1.set_ylabel(y_title, fontdict=None, labelpad=None)
        # In case user wants to set axis limits  manually, this parameter is
        # used if not set to 'auto'
        # x-scale needs to be specified as [x_min,x_max]
        if dq_dv_x1_limits != "auto":
            ax1.set_xlim(dq_dv_x1_limits)
        if dq_dv_x2_limits != "auto":
            ax2.set_xlim(dq_dv_x2_limits)
        # y-scale needs to be specified as [y_min,y_max]
        if dq_dv_y_limits != "auto":
            ax1.set_ylim(dq_dv_y_limits)
            ax2.set_ylim(dq_dv_y_limits)

        # To catch the mAh/kg exception scientific notation is used
        if (cap_unit == "mAh") & (w_unit == "kg") & (x_title[0:7] == "Capacity"):
            ax1.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            ax2.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

        h1, l1 = ax1.get_legend_handles_labels()
        lgd = ax1.legend(
            h1,
            l1,
            bbox_to_anchor=(0, 1.05, 2.085, 0.105),
            loc="lower left",
            ncol=4,
            mode="expand",
            borderaxespad=0,
        )
        # Get number of legend entries to estimate needed  rectangle size
        num_leg = len(l1)
        extra = 0.5 * round(num_leg / 4) * 4 / 50  # 4 equals n_col
        fig.tight_layout(rect=(0, 0, 1, 1 + extra))

        # Export graph
        export_graph(
            export=export_graphs,
            figure=fig,
            legend=lgd,
            index=index,
            sample_name=sample_name,
            analysis_type="dQ_dV_3el",
            export_dir=export_dir,
        )

        if mpl.is_interactive() is True:
            if draw_graphs_on_screen == 1:
                fig.show()
            elif draw_graphs_on_screen == 0:
                plt.close(fig)  # has to be called after export


#    # Finally raise all figure interactive windows to the front
#    figNumsFin = plt.get_fig_nums()
#    if mpl.is_interactive() == True:
#        for figNum in range(figNumsIni[-1]+1,figNumsFin[-1]+1):
#            raise_window(figNum)

#    return maccorData, cycleBased, cycleList
#    return capBased3, vBasedF3  # variable_to_inspect
# Output has to be assigned to a variable.
# So put variable_to_inspect = function_call() in the calling python file
