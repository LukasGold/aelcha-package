# -*- coding: utf-8 -*-
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, PrivateAttr, field_validator
from typing_extensions import Dict, List, Optional, Union

from aelcha.common import (
    Capacity,
    CapacityLimit,
    ChargeIncrement,
    ChargePerVoltageLimit,
    CyclesToPlotOption,
    DigatronPreprocessingOption,
    FilterSelectionOption,
    LimitOption,
    MaccorPreprocessingOption,
    Mass,
    Sample,
    SeparatorOption,
    VoltageIncrement,
    VoltageLimit,
    get_model_field_default_value,
    parse_cycles_to_plot,
)


class PandasVersion:
    major = int(pd.__version__.split(".")[0])
    minor = int(pd.__version__.split(".")[1])
    micro = int(pd.__version__.split(".")[2])


class AnalysisParam(BaseModel):
    cycles_to_plot: Union[CyclesToPlotOption, List[int], str]
    """List of cycle numbers to be plotted. Default: all cycles."""
    export_graphs: bool
    """Whether to export the graphs."""
    export_converted_raw_data: bool
    """Whether to export the converted raw data."""
    export_analysis_data: bool
    """Whether to export the analysis data."""
    plot_cap_ce_vs_cn: bool
    """Whether to plot the (charge and discharge) capacity & coulombic efficiency vs.
    cycle number graph."""
    plot_volt_vs_cap: bool
    """Whether to plot the voltage vs. capacity graph."""
    plot_dq_dv: bool
    """Whether to plot the dQ/dV graph."""
    plot_dv_dq: bool
    """Whether to plot the dV/dQ graph."""
    dv_min_increment: Union[float, VoltageIncrement]
    """dVmin / mV - minimum step size between two considered voltage samples. Higher
    value increases SNR but decreases resolution. Default: 2, max: 10"""
    dq_dv_filter_selection: FilterSelectionOption
    """Choose filter to be used for smoothing of dQ/dV result:
        0 = none, 1 = Savitzky Golay, 2 = Gaussian,
        3 = median filter, 4 = adjacent average"""
    dq_dv_filter_window_size: int
    """Filter window size. Enter uneven integer. Does not apply for Gaussian filter.
        Default: 5 points """
    dq_dv_filter_sigma_or_polynomial_order: int
    """For Gaussian filter enter sigma, for Savitzky Golay enter polynomial order."""

    def __init__(self, **data):
        super().__init__(**data)
        if isinstance(self.dv_min_increment, float):
            self.dv_min_increment = VoltageIncrement(
                numerical_value=self.dv_min_increment
            )
        if isinstance(self.cycles_to_plot, str):
            self.cycles_to_plot = parse_cycles_to_plot(self.cycles_to_plot)


class Configuration(BaseModel):
    input_dir_default: Optional[Union[str, Path]] = None
    """Default input directory."""
    export_dir_default: Optional[Union[str, Path]] = None
    """Default export directory."""
    input_source_type: Optional[
        Union[MaccorPreprocessingOption, DigatronPreprocessingOption]
    ] = MaccorPreprocessingOption.mims_client1
    """Input source type."""
    decimal_separator: SeparatorOption = SeparatorOption.comma
    """Decimal separator in the input files."""
    thousands_separator: SeparatorOption = SeparatorOption.dot
    """Thousands separator in the input files."""
    maccor_dll_path: Optional[Union[str, Path]] = None
    """Path to the Maccor DLL file."""
    colormap: str = "gist_rainbow"
    """Colormap for plotting. Must be a valid matplotlib colormap. List of available
    colormaps: https://matplotlib.org/stable/gallery/color/colormap_reference.html"""
    cm_start: float = Field(default=0, ge=0, le=1)
    """Colormap start value."""
    cm_end: float = Field(default=1, ge=0, le=1)
    """Colormap end value."""
    draw_graphs_on_screen: bool = True
    """Whether to draw the graphs on the screen, in Python - uses a lot of memory!
    Changes in this option require a kernel or python restart to take effect!."""
    export_graphs_default: bool = False
    """Default value for exporting the graphs."""
    export_converted_raw_data_default: bool = False
    """Default value for exporting the converted raw data."""
    export_analysis_data_default: bool = False
    """Default value for exporting the analysis data."""
    capacity_default: Union[float, Capacity] = None
    """Default value for rated or theoretical capacity of the sample. Any float value
    will be considered as the capacity in mAh."""
    active_material_mass_default: Union[float, Mass] = None
    """Default value for active material mass of the sample, which will be used for
    normalization. Any float value will be considered as the mass in g. None means no
    normalization."""
    three_electrode_cell_default: bool = False
    """Default value for three-electrode cell."""
    cycles_to_plot_default: Union[CyclesToPlotOption, List[int]] = (
        CyclesToPlotOption.all
    )
    """Default value for list of cycle numbers to be plotted. Default: all cycles."""
    plot_cap_ce_vs_cn_default: bool = False
    """Default value for plotting the (charge and discharge) capacity & Coulombic
    efficiency vs. cycle number graph."""
    plot_volt_vs_cap_default: bool = False
    """Default value for plotting the voltage vs. capacity graph."""
    plot_dq_dv_default: bool = False
    """Default value for plotting the dQ/dV graph."""
    plot_dv_dq_default: bool = False
    """Default value for plotting the dV/dQ graph."""
    dv_min_increment_default: Union[float, VoltageIncrement] = VoltageIncrement(
        numerical_value=2
    )
    """Default value for dVmin / mV - minimum step size between two considered voltage
    samples. Higher value increases SNR but decreases resolution. Default: 2, max: 10"""
    dq_min_increment_default: ChargeIncrement = None
    """Default value for dQmin / mAh - minimum step size between two considered charge
    samples."""
    dq_dv_filter_selection_default: FilterSelectionOption = FilterSelectionOption.none
    """Default value for choosing filter to be used for smoothing of dQ/dV result:
        0 = none, 1 = Savitzky Golay, 2 = Gaussian,
        3 = median filter, 4 = adjacent average"""
    dq_dv_filter_window_size_default: int = 5
    """Default value for filter window size. Enter uneven integer. Does not apply for
    Gaussian filter. Default: 5 points """
    dq_dv_filter_sigma_or_polynomial_order_default: int = Field(default=1, ge=1, le=10)
    """Default value for Gaussian filter enter sigma. For Savitzky Golay, enter
    polynomial order."""
    # Voltage vs. capacity graph
    # Two electrode cells
    # X-axis is capacity
    v_vs_cap_2el_x_low_lim: Union[float, LimitOption, CapacityLimit] = LimitOption.auto
    """Lower limit of the x-axis in the voltage vs. capacity graph for two-electrode
    cells. Any float value will be considered as the capacity limit in mAh."""
    v_vs_cap_2el_x_up_lim: Union[float, LimitOption, CapacityLimit] = LimitOption.auto
    """Upper limit of the x-axis in the voltage vs. capacity graph for two-electrode
    cells. Any float value will be considered as the capacity limit in mAh."""
    # Y-axis is voltage
    v_vs_cap_2el_y_low_lim: Union[float, LimitOption, VoltageLimit] = LimitOption.auto
    """Lower limit of the y-axis in the voltage vs. capacity graph for two-electrode
    cells. Any float value will be considered as the voltage limit in V."""
    v_vs_cap_2el_y_up_lim: Union[float, LimitOption, VoltageLimit] = LimitOption.auto
    """Upper limit of the y-axis in the voltage vs. capacity graph for two-electrode
    cells. Any float value will be considered as the voltage limit in V."""
    # Three electrode cells
    # X-axis is capacity
    # v_vs_cap_3el_x_low_lim: Union[float, LimitOptions, CapacityLimit] = (
    #     LimitOptions.auto
    # )
    # """Lower limit of the x-axis in the voltage vs. capacity graph for three-electrode
    # cells. Any float value will be considered as the capacity limit in mAh."""
    # v_vs_cap_3el_x_up_lim: Union[float, LimitOptions, CapacityLimit] =
    #   LimitOptions.auto
    # """Upper limit of the x-axis in the voltage vs. capacity graph for three-electrode
    # cells. Any float value will be considered as the capacity limit in mAh."""
    # Y-axis is voltage
    v_vs_cap_3el_left_y_low_lim: Union[float, LimitOption, VoltageLimit] = (
        LimitOption.auto
    )
    """Lower limit of the left y-axis in the voltage vs. capacity graph for
    three-electrode cells. Any float value will be considered as the voltage limit in V.
    """
    v_vs_cap_3el_left_y_up_lim: Union[float, LimitOption, VoltageLimit] = (
        LimitOption.auto
    )
    """Upper limit of the left y-axis in the voltage vs. capacity graph for
    three-electrode cells. Any float value will be considered as the voltage limit in V.
    """
    v_vs_cap_3el_right_y_low_lim: Union[float, LimitOption, VoltageLimit] = (
        LimitOption.auto
    )
    """Lower limit of the right y-axis in the voltage vs. capacity graph for
    three-electrode cells. Any float value will be considered as the voltage limit in V.
    """
    v_vs_cap_3el_right_y_up_lim: Union[float, LimitOption, VoltageLimit] = (
        LimitOption.auto
    )
    """Upper limit of the right y-axis in the voltage vs. capacity graph for
    three-electrode cells. Any float value will be considered as the voltage limit in V.
    """
    # dQ/dV vs. voltage graph
    # Two electrode cells
    # X-axis is voltage
    dq_dv_vs_v_2el_x_low_lim: Union[float, LimitOption, VoltageLimit] = LimitOption.auto
    """Lower limit of the x-axis in the dQ/dV vs. voltage graph for two-electrode cells.
    Any float value will be considered as the voltage limit in V."""
    dq_dv_vs_v_2el_x_up_lim: Union[float, LimitOption, VoltageLimit] = LimitOption.auto
    """Upper limit of the x-axis in the dQ/dV vs. voltage graph for two-electrode cells.
    Any float value will be considered as the voltage limit in V."""
    # Y-axis is dQ/dV
    dq_dv_vs_v_2el_y_low_lim: Union[float, LimitOption, ChargePerVoltageLimit] = (
        LimitOption.auto
    )
    """Lower limit of the y-axis in the dQ/dV vs. voltage graph for two-electrode cells.
    Any float value will be considered as the charge per voltage limit in mAh/V."""
    dq_dv_vs_v_2el_y_up_lim: Union[float, LimitOption, ChargePerVoltageLimit] = (
        LimitOption.auto
    )
    """Upper limit of the y-axis in the dQ/dV vs. voltage graph for two-electrode cells.
    Any float value will be considered as the charge per voltage limit in mAh/V."""
    # Three electrode cells
    # X-axis is voltage
    dq_dv_vs_v_3el_left_x_low_lim: Union[float, LimitOption, VoltageLimit] = (
        LimitOption.auto
    )
    """Lower limit of the left x-axis in the dQ/dV vs. voltage graph for three-electrode
    cells. Any float value will be considered as the voltage limit in V."""
    dq_dv_vs_v_3el_left_x_up_lim: Union[float, LimitOption, VoltageLimit] = (
        LimitOption.auto
    )
    """Upper limit of the left x-axis in the dQ/dV vs. voltage graph for three-electrode
    cells. Any float value will be considered as the voltage limit in V."""
    dq_dv_vs_v_3el_right_x_low_lim: Union[float, LimitOption, VoltageLimit] = (
        LimitOption.auto
    )
    """Lower limit of the right x-axis in the dQ/dV vs. voltage graph for
    three-electrode cells. Any float value will be considered as the voltage limit in V.
    """
    dq_dv_vs_v_3el_right_x_up_lim: Union[float, LimitOption, VoltageLimit] = (
        LimitOption.auto
    )
    """Upper limit of the right x-axis in the dQ/dV vs. voltage graph for
    three-electrode cells. Any float value will be considered as the voltage limit in V.
    """

    # Y-axis is dQ/dV
    # dq_dv_vs_v_3el_y_low_lim: Union[float, LimitOptions, ChargePerVoltageLimit] = (
    #     LimitOptions.auto
    # )
    # """Lower limit of the y-axis in the dQ/dV vs. voltage graph for three-electrode
    # cells. Any float value will be considered as the charge per voltage limit
    # in mAh/V."""
    # dq_dv_vs_v_3el_y_up_lim: Union[float, LimitOptions, ChargePerVoltageLimit] = (
    #     LimitOptions.auto
    # )
    # """Upper limit of the y-axis in the dQ/dV vs. voltage graph for three-electrode
    # cells. Any float value will be considered as the charge per voltage limit
    # in mAh/V."""

    @field_validator("colormap")
    def validate_colormap(cls, colormap):
        if colormap not in plt.colormaps:
            raise ValueError(f"Invalid colormap: {colormap}")
        return colormap

    def __init__(self, **data):
        super().__init__(**data)
        # Ensure correct types
        if isinstance(self.maccor_dll_path, str):
            self.maccor_dll_path = Path(self.maccor_dll_path)
        if isinstance(self.capacity_default, float):
            self.capacity = Capacity(numerical_value=self.capacity_default)
        if isinstance(self.active_material_mass_default, float):
            self.active_material_mass = Mass(
                numerical_value=self.active_material_mass_default
            )
        if isinstance(self.cycles_to_plot_default, str):
            self.cycles_to_plot_default = parse_cycles_to_plot(
                self.cycles_to_plot_default
            )
        # Voltage vs. capacity graph
        # Two electrode cells
        # X-axis is capacity
        if isinstance(self.v_vs_cap_2el_x_low_lim, float):
            self.v_vs_cap_2el_x_low_lim = CapacityLimit(
                numerical_value=self.v_vs_cap_2el_x_low_lim
            )
        if isinstance(self.v_vs_cap_2el_x_up_lim, float):
            self.v_vs_cap_2el_x_up_lim = CapacityLimit(
                numerical_value=self.v_vs_cap_2el_x_up_lim
            )
        # Y-axis is voltage
        if isinstance(self.v_vs_cap_2el_y_low_lim, float):
            self.v_vs_cap_2el_y_low_lim = VoltageLimit(
                numerical_value=self.v_vs_cap_2el_y_low_lim
            )
        if isinstance(self.v_vs_cap_2el_y_up_lim, float):
            self.v_vs_cap_2el_y_up_lim = VoltageLimit(
                numerical_value=self.v_vs_cap_2el_y_up_lim
            )
        # Three electrode cells
        # X-axis is capacity
        # if isinstance(self.v_vs_cap_3el_x_low_lim, float):
        #     self.v_vs_cap_3el_x_low_lim = CapacityLimit(
        #         numerical_value=self.v_vs_cap_3el_x_low_lim
        #     )
        # if isinstance(self.v_vs_cap_3el_x_up_lim, float):
        #     self.v_vs_cap_3el_x_up_lim = CapacityLimit(
        #         numerical_value=self.v_vs_cap_3el_x_up_lim
        #     )
        # Y-axis is voltage
        if isinstance(self.v_vs_cap_3el_left_y_low_lim, float):
            self.v_vs_cap_3el_left_y_low_lim = VoltageLimit(
                numerical_value=self.v_vs_cap_3el_left_y_low_lim
            )
        if isinstance(self.v_vs_cap_3el_left_y_up_lim, float):
            self.v_vs_cap_3el_left_y_up_lim = VoltageLimit(
                numerical_value=self.v_vs_cap_3el_left_y_up_lim
            )
        # dQ/dV vs. voltage graph
        # Two electrode cells
        # X-axis is voltage
        if isinstance(self.dq_dv_vs_v_2el_x_low_lim, float):
            self.dq_dv_vs_v_2el_x_low_lim = VoltageLimit(
                numerical_value=self.dq_dv_vs_v_2el_x_low_lim
            )
        if isinstance(self.dq_dv_vs_v_2el_x_up_lim, float):
            self.dq_dv_vs_v_2el_x_up_lim = VoltageLimit(
                numerical_value=self.dq_dv_vs_v_2el_x_up_lim
            )
        # Y-axis is dQ/dV
        if isinstance(self.dq_dv_vs_v_2el_y_low_lim, float):
            self.dq_dv_vs_v_2el_y_low_lim = ChargePerVoltageLimit(
                numerical_value=self.dq_dv_vs_v_2el_y_low_lim
            )
        if isinstance(self.dq_dv_vs_v_2el_y_up_lim, float):
            self.dq_dv_vs_v_2el_y_up_lim = ChargePerVoltageLimit(
                numerical_value=self.dq_dv_vs_v_2el_y_up_lim
            )
        # Three electrode cells
        # X-axis is voltage
        if isinstance(self.dq_dv_vs_v_3el_left_x_low_lim, float):
            self.dq_dv_vs_v_3el_left_x_low_lim = VoltageLimit(
                numerical_value=self.dq_dv_vs_v_3el_left_x_low_lim
            )
        if isinstance(self.dq_dv_vs_v_3el_left_x_up_lim, float):
            self.dq_dv_vs_v_3el_left_x_up_lim = VoltageLimit(
                numerical_value=self.dq_dv_vs_v_3el_left_x_up_lim
            )
        # Y-axis is dQ/dV
        # if isinstance(self.dq_dv_vs_v_3el_y_low_lim, float):
        #     self.dq_dv_vs_v_3el_y_low_lim = ChargePerVoltageLimit(
        #         numerical_value=self.dq_dv_vs_v_3el_y_low_lim
        #     )
        # if isinstance(self.dq_dv_vs_v_3el_y_up_lim, float):
        #     self.dq_dv_vs_v_3el_y_up_lim = ChargePerVoltageLimit(
        #         numerical_value=self.dq_dv_vs_v_3el_y_up_lim
        #     )
        # Params
        if isinstance(self.dv_min_increment_default, float):
            self.dv_min_increment_default = VoltageIncrement(
                numerical_value=self.dv_min_increment_default
            )


def cfg(field: str):
    return get_model_field_default_value(Configuration, field)


class SelectionRow(BaseModel):
    index: int
    """Index of the row in the table."""
    process_file: bool = True
    """Whether to process the file."""
    input_dir: Union[str, Path] = cfg("input_dir_default")
    """Directory containing the file."""
    file_name: str
    """Name of the file with suffix(es)."""
    _file_path: Path = PrivateAttr(None)
    """Path of the file."""
    sample_name: str
    """Name of the sample, which will be used for labeling."""
    capacity: Union[float, Capacity] = cfg("capacity_default")
    """Rated, nominal or theoretical capacity of the sample. Any float value will be
    considered as the capacity in mAh."""
    active_material_mass: Union[float, Mass] = cfg("active_material_mass_default")
    """Active material mass of the sample, which will be used for normalization. Any
    float value will be considered as the mass in g. None means no normalization."""
    three_electrode_cell: bool = cfg("three_electrode_cell_default")
    """Whether the cell is a three-electrode cell."""
    _sample: Sample = PrivateAttr(None)
    """Sample object."""
    cycles_to_plot: Union[CyclesToPlotOption, List[int]] = cfg("cycles_to_plot_default")
    """List of cycle numbers to be plotted. Default: all cycles."""
    export_converted_raw_data: bool = cfg("export_converted_raw_data_default")
    """Whether to export the converted raw data."""
    export_graphs: bool = cfg("export_graphs_default")
    """Whether to export the graphs."""
    export_analysis_data: bool = cfg("export_analysis_data_default")
    """Whether to export the analysis data."""
    plot_cap_ce_vs_cn: bool = cfg("plot_cap_ce_vs_cn_default")
    """Whether to plot the (charge and discharge) capacity & Coulombic efficiency vs.
    cycle number graph."""
    plot_volt_vs_cap: bool = cfg("plot_volt_vs_cap_default")
    """Whether to plot the voltage vs. capacity graph."""
    plot_dq_dv: bool = cfg("plot_dq_dv_default")
    """Whether to plot the dQ/dV graph."""
    plot_dv_dq: bool = cfg("plot_dv_dq_default")  # todo: missing in the selection table
    """Whether to plot the dV/dQ graph."""
    dv_min_increment: Union[float, VoltageIncrement] = cfg("dv_min_increment_default")
    """dVmin / mV - minimum step size between two considered voltage samples. Higher
    value increases SNR but decreases resolution. Default: 2, max: 10"""
    dq_min_increment: Union[float, ChargeIncrement] = cfg("dq_min_increment_default")
    """dQmin / mAh - minimum step size between two considered charge samples."""
    dq_dv_filter_selection: FilterSelectionOption = cfg(
        "dq_dv_filter_selection_default"
    )
    """Choose filter to be used for smoothing of dQ/dV result:
        0 = none, 1 = Savitzky Golay, 2 = Gaussian,
        3 = median filter, 4 = adjacent average"""
    dq_dv_filter_window_size: int = cfg("dq_dv_filter_window_size_default")
    """Filter window size. Enter uneven integer. Does not apply for Gaussian filter.
        Default: 5 points """
    dq_dv_filter_sigma_or_polynomial_order: int = Field(
        default=cfg("dq_dv_filter_sigma_or_polynomial_order_default"), ge=1, le=10
    )
    """For Gaussian filter enter sigma, for Savitzky Golay enter polynomial order."""
    _analysis_param: AnalysisParam = PrivateAttr(None)
    """Analysis parameter object."""
    export_dir: Union[str, Path] = cfg("export_dir_default")
    """Directory to export the results and data."""

    def __init__(self, **data):
        super().__init__(**data)
        # Ensure correct types
        if isinstance(self.capacity, float):
            self.capacity = Capacity(numerical_value=self.capacity)
        if isinstance(self.active_material_mass, float):
            self.active_material_mass = Mass(numerical_value=self.active_material_mass)
        if isinstance(self.cycles_to_plot, str):
            self.cycles_to_plot = parse_cycles_to_plot(self.cycles_to_plot)
        # Set default values
        if self.input_dir is None:
            self.input_dir = Path.cwd()
        if self.export_dir is None:
            self.export_dir = Path(self.input_dir)
        if self.dq_min_increment is None:
            self.dq_min_increment = ChargeIncrement(
                numerical_value=self.capacity.numerical_value / 100
            )
        # Construct _file_path
        self._file_path = Path(self.input_dir) / self.file_name
        # Construct _sample
        self._sample = Sample(
            name=self.sample_name,
            capacity=self.capacity,
            active_material_mass=self.active_material_mass,
            three_electrode_cell=self.three_electrode_cell,
        )
        # Construct _analysis_param
        self._analysis_param = AnalysisParam(
            cycles_to_plot=self.cycles_to_plot,
            export_graphs=self.export_graphs,
            export_converted_raw_data=self.export_converted_raw_data,
            export_analysis_data=self.export_analysis_data,
            plot_cap_ce_vs_cn=self.plot_cap_ce_vs_cn,
            plot_volt_vs_cap=self.plot_volt_vs_cap,
            plot_dq_dv=self.plot_dq_dv,
            plot_dv_dq=self.plot_dv_dq,
            dv_min_increment=self.dv_min_increment,
            dq_dv_filter_selection=self.dq_dv_filter_selection,
            dq_dv_filter_window_size=self.dq_dv_filter_window_size,
            dq_dv_filter_sigma_or_polynomial_order=(
                self.dq_dv_filter_sigma_or_polynomial_order
            ),
        )


class FileSelection(BaseModel):
    selection_rows: List[SelectionRow]
    configuration: Configuration


class DataTableLoc(BaseModel):
    row: int
    col: str


class DataTableSearch(BaseModel):
    rc: str
    """Row content"""
    ch: Optional[str] = "val2"
    """Column header"""


class TableToSelection(BaseModel):
    table: pd.DataFrame
    mapping: Optional[Dict[str, DataTableLoc]] = None
    search: Optional[Dict[str, DataTableSearch]] = None

    def search_content_return_loc(self, key: str) -> DataTableLoc:
        if key not in self.search.keys():
            raise ValueError(f"Key '{key}' not found in the search dictionary.")
        for col in self.table.columns:
            for row in self.table.index:
                # Find the row which contains the row content
                if self.table.at[row, col] == self.search[key].rc:
                    # Return the location with this row and the previously specified
                    #  column header
                    return DataTableLoc(row=row, col=self.search[key].ch)
        raise KeyError(
            f"No matching row found in the configuration table for the "
            f"argument '{key} and the row content '{self.search[key].rc}'"
        )

    def __init__(self, **data):
        super().__init__(**data)
        if self.search is not None:
            self.mapping = {}
            for key in self.search.keys():
                self.mapping[key] = self.search_content_return_loc(key)
        if self.mapping is None:
            raise ValueError(
                "If the attribute 'search' is not provided, 'mapping' must be provided."
            )
        missing_keys = []
        for key in Configuration.model_fields.keys():
            if key not in self.mapping.keys():
                missing_keys.append(key)
        if len(missing_keys) != 0:
            raise KeyError(
                f"The following Configuration attributes are not matched "
                f"in the mapping: {', '.join(missing_keys)}"
            )

    @field_validator("mapping")
    def validate_mapping(cls, mapping):
        if mapping is None:
            return mapping
        if not all(key in Configuration.model_fields for key in mapping.keys()):
            raise ValueError("Mapping keys must be valid Configuration fields.")
        return mapping

    @field_validator("search")
    def validate_search(cls, search):
        if search is None:
            return search
        if not all(key in Configuration.model_fields for key in search.keys()):
            raise ValueError("Search keys must be valid Configuration fields.")
        return search

    class Config:
        arbitrary_types_allowed = True


def is_none_or_nan(val):
    if isinstance(val, str):
        if val.lower() == "nan":
            return True
        return False
    if np.isnan(val):
        return True
    elif val is None:
        return True
    return False


def read_file_selection(
    fp: Union[str, Path], maccor_dll_fp: Union[str, Path] = None
) -> FileSelection:
    if isinstance(fp, str):
        fp = Path(fp)
    if maccor_dll_fp is not None:
        if isinstance(maccor_dll_fp, str):
            maccor_dll_fp = Path(maccor_dll_fp)

    pandas_version = PandasVersion()
    # python_version = sys.version_info

    if pandas_version.major == 0:
        fs_col_types = {
            "process_file": bool,
            "input_dir": str,
            "file_name": str,
            "sample_name": str,
            "capacity": float,
            "active_material_mass": float,
            "three_electrode_cell": bool,
            "cycles_to_plot": str,
            "export_converted_raw_data": bool,
            "export_graphs": bool,
            "export_analysis_data": bool,
            "plot_cap_ce_vs_cn": bool,
            "plot_volt_vs_cap": bool,
            "plot_dq_dv": bool,
            "dv_min_increment": float,
            "dq_dv_filter_selection": int,
            "dq_dv_filter_window_size": int,
            "dq_dv_filter_sigma_or_polynomial_order": int,
            "export_dir": str,
        }
        fs_col_names = list(fs_col_types.keys())
        fs_table = pd.read_excel(
            io=fp,
            sheet_name="Selection",
            header=1,
            index_col=0,
            names=fs_col_names,
            dtype=fs_col_types,
        )
    else:  # (pandas_version.major == 1 and pandas_version.minor < 5):
        fs_col_types = {
            "index": int,
            "process_file": bool,
            "input_dir": str,
            "file_name": str,
            "sample_name": str,
            "capacity": float,
            "active_material_mass": float,
            "three_electrode_cell": bool,
            "cycles_to_plot": str,
            "export_converted_raw_data": bool,
            "export_graphs": bool,
            "export_analysis_data": bool,
            "plot_cap_ce_vs_cn": bool,
            "plot_volt_vs_cap": bool,
            "plot_dq_dv": bool,
            "dv_min_increment": float,
            "dq_dv_filter_selection": int,
            "dq_dv_filter_window_size": int,
            "dq_dv_filter_sigma_or_polynomial_order": int,
            "export_dir": str,
        }
        fs_col_names = list(fs_col_types.keys())
        fs_table = pd.read_excel(
            io=fp,
            sheet_name="Selection",
            header=2,
            names=fs_col_names,
            usecols=fs_col_names,
        )
        fs_table = fs_table.dropna(subset=["file_name"], axis="index").astype(
            dtype=fs_col_types
        )

    cfg_col_names = ["param", "val1", "val2", "def"]

    cfg_table = pd.read_excel(
        io=fp, sheet_name="Configuration", header=2, names=cfg_col_names, dtype=None
    )

    cfg_map = TableToSelection(
        table=cfg_table,
        mapping=None,
        search={
            # Sample (battery cell) details
            # -----------------------------
            "capacity_default": DataTableSearch(
                rc="Rated, nominal or theoretical capacity / mAh"
            ),
            "active_material_mass_default": DataTableSearch(
                rc="Active material mass / g"
            ),
            "three_electrode_cell_default": DataTableSearch(
                rc="Three electrode cell (0 = no, 1 = yes)"
            ),
            # Input data
            # ----------
            "input_dir_default": DataTableSearch(rc="Input directory"),
            "input_source_type": DataTableSearch(
                rc="Source type. Accepted values: "
                "-11 (Maccor raw files), "
                "10 (MIMS client v1 export), "
                "11 (MaccorExport.exe v1), "
                "12 (MaccorExport.exe v2), "
                "13 (MIMS client v2 export), "
                "14 (MIMS server v2 export), "
                "20 (Digatron german client csv export)"
            ),
            "decimal_separator": DataTableSearch(
                rc="Decimal separator set on the system. Usually the decimal separator "
                "is set to comma when the operating systems language is set to "
                "German. If you set it to German (Switzerland) or to Englisch (UK, "
                "US, …), the decimal separator is dot. Accepted values for this "
                "option are . and , "
            ),
            "thousands_separator": DataTableSearch(
                rc="Thousands separator. Accepted values for this option are , and . "
                "and ' "
            ),
            "maccor_dll_path": DataTableSearch(
                rc="Path to the MacReadDataFileLIB.dll, used for reading Maccor raw "
                "files directly"
            ),
            # Output data
            # -----------
            "export_dir_default": DataTableSearch(rc="Output directory"),
            "export_converted_raw_data_default": DataTableSearch(
                rc="Export converted raw data"
            ),
            "export_analysis_data_default": DataTableSearch(
                rc="Export analysis data - to specified directory"
            ),
            "export_graphs_default": DataTableSearch(
                rc="Export graphs - to specified directory"
            ),
            # Drawing options
            # ---------------
            "draw_graphs_on_screen": DataTableSearch(
                rc="Draw graphs in python - uses a lot of memory! Changes in this "
                "option require a kernel or python restart to take effect!"
            ),
            "colormap": DataTableSearch(rc="Colormap name"),
            "cm_start": DataTableSearch(rc="Start (between 0 and 1)"),
            "cm_end": DataTableSearch(rc="End (between 0 and 1)"),
            # Graph selection - which graphs to create
            # ----------------------------------------
            "cycles_to_plot_default": DataTableSearch(rc="Cycles to plot"),
            "plot_cap_ce_vs_cn_default": DataTableSearch(
                rc="Charge and discharge capacity, Coulombic efficiency vs. cycle "
                "number"
            ),
            "plot_volt_vs_cap_default": DataTableSearch(rc="Voltage vs. capacity"),
            "plot_dq_dv_default": DataTableSearch(rc="dQ/dV"),
            "plot_dv_dq_default": DataTableSearch(rc="dV/dQ - not yet implemented"),
            # Graph axis limits
            # -----------------
            # Voltage vs. capacity plot
            # Two electrode cells
            "v_vs_cap_2el_x_low_lim": DataTableSearch(
                rc="Two electrode plot x-axis", ch="val1"
            ),
            "v_vs_cap_2el_x_up_lim": DataTableSearch(rc="Two electrode plot x-axis"),
            "v_vs_cap_2el_y_low_lim": DataTableSearch(
                rc="Two electrode plot y-axis", ch="val1"
            ),
            "v_vs_cap_2el_y_up_lim": DataTableSearch(rc="Two electrode plot y-axis"),
            # Three electrode cells
            "v_vs_cap_3el_left_y_low_lim": DataTableSearch(
                rc="Three electrode plot left y-axis", ch="val1"
            ),
            "v_vs_cap_3el_left_y_up_lim": DataTableSearch(
                rc="Three electrode plot left y-axis"
            ),
            "v_vs_cap_3el_right_y_low_lim": DataTableSearch(
                rc="Three electrode plot right y-axis", ch="val1"
            ),
            "v_vs_cap_3el_right_y_up_lim": DataTableSearch(
                rc="Three electrode plot right y-axis"
            ),
            # dQ/dV vs. voltage plot
            # Two electrode cells
            "dq_dv_vs_v_2el_x_low_lim": DataTableSearch(
                rc="Two electrode plot x-axis _", ch="val1"
            ),
            "dq_dv_vs_v_2el_x_up_lim": DataTableSearch(
                rc="Two electrode plot x-axis _"
            ),
            "dq_dv_vs_v_2el_y_low_lim": DataTableSearch(
                rc="Two electrode plot y-axis _", ch="val1"
            ),
            "dq_dv_vs_v_2el_y_up_lim": DataTableSearch(
                rc="Two electrode plot y-axis _"
            ),
            # Three electrode cells
            "dq_dv_vs_v_3el_left_x_low_lim": DataTableSearch(
                rc="Three electrode plot left x-axis _", ch="val1"
            ),
            "dq_dv_vs_v_3el_left_x_up_lim": DataTableSearch(
                rc="Three electrode plot left x-axis _"
            ),
            "dq_dv_vs_v_3el_right_x_low_lim": DataTableSearch(
                rc="Three electrode plot right x-axis _"
            ),
            "dq_dv_vs_v_3el_right_x_up_lim": DataTableSearch(
                rc="Three electrode plot right x-axis _"
            ),
            # Analysis options
            # ----------------
            # dV/dQ
            "dv_min_increment_default": DataTableSearch(
                rc="dVmin / mV - minimum step size between two considered voltage "
                "samples. Higher value increases SNR but decreases resolution. "
                "Maximum: 10"
            ),
            "dq_dv_filter_selection_default": DataTableSearch(
                rc="Choose filter to be used for smoothing of dQ/dV result: 0 = none, "
                "1 = Savitzky Golay, 2 = Gaussian, 3 = median filter, "
                "4 = adjacent average"
            ),
            "dq_dv_filter_window_size_default": DataTableSearch(
                rc="Filter window size. Enter uneven integer. Does not apply for "
                "Gaussian filter"
            ),
            "dq_dv_filter_sigma_or_polynomial_order_default": DataTableSearch(
                rc="For Gaussian filter enter sigma, for Savitzky Golay enter "
                "polynomial order"
            ),
            # dQ/dV
            "dq_min_increment_default": DataTableSearch(
                rc="dQmin / mAh - minimum step size between two considered capacity "
                "samples"
            ),
        },
    )
    cfg_data = {
        key: cfg_table.at[dtl.row, dtl.col]
        for key, dtl in cfg_map.mapping.items()
        if not is_none_or_nan(cfg_table.at[dtl.row, dtl.col])
    }
    if maccor_dll_fp is not None:
        if "maccor_dll_path" in cfg_data.keys():
            if is_none_or_nan(cfg_data["maccor_dll_path"]):
                cfg_data["maccor_dll_path"] = maccor_dll_fp
        else:
            cfg_data["maccor_dll_path"] = maccor_dll_fp

    selection_rows = [
        SelectionRow(
            **{k: v for k, v in fs_table.loc[row].to_dict().items() if v is not None}
        )
        for row in fs_table.index
    ]

    return FileSelection(
        selection_rows=selection_rows, configuration=Configuration(**cfg_data)
    )


if __name__ == "__main__":
    file_path = Path(__file__).parents[2] / "examples" / "File_Selection.xlsx"
    file_selection = read_file_selection(file_path)
