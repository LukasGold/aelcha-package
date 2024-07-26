# -*- coding: utf-8 -*-
from enum import Enum, IntEnum

import uncertainties as unc
from pydantic import BaseModel, Field, PrivateAttr
from typing_extensions import List, Optional, Union


class QuantityStatement(BaseModel):
    numerical_value: float
    """Numerical value of the quantity."""
    uncertainty: Optional[float] = 0
    """Uncertainty of the quantity. Default: 0."""
    unit: str
    """Unit of the quantity."""
    property: str
    """Semantic property of the quantity."""
    _value_str: PrivateAttr(str) = None
    """String representation of the quantity."""
    _uvalue: PrivateAttr(str) = None
    """Value of the quantity with uncertainty."""
    _uvalue_str: PrivateAttr(str) = None
    """String representation of the quantity."""

    def __init__(self, **data):
        super().__init__(**data)
        self._value_str = f"{self.numerical_value} {self.unit}"
        self._uvalue = unc.ufloat(self.numerical_value, self.uncertainty)
        self._uvalue_str = f"({self.numerical_value} ± {self.uncertainty}) {self.unit}"


class Mass(QuantityStatement):
    unit: str = "g"
    property: str = "Property:HasMass"


class Voltage(QuantityStatement):
    unit: str = "V"
    property: str = "Property:HasVoltage"


class Charge(QuantityStatement):
    unit: str = "mAh"
    property: str = "Property:HasCharge"


class Capacity(Charge):
    property: str = "Property:HasCapacity"


class ChargeIncrement(Charge):
    property: str = "Property:HasChargeIncrement"


class VoltageIncrement(Voltage):
    numerical_value: float = Field(default=2, le=10)
    unit: str = "mV"
    property: str = "Property:HasVoltageIncrement"


class ChargePerVoltage(QuantityStatement):
    unit: str = "mAh/V"
    property: str = "Property:HasChargePerVoltage"


class LimitOption(Enum):
    auto = 0


class VoltageLimit(Voltage):
    property: str = "Property:HasVoltageLimit"


class CapacityLimit(Capacity):
    property: str = "Property:HasCapacityLimit"


class ChargePerVoltageLimit(ChargePerVoltage):
    property: str = "Property:HasChargePerVoltageLimit"


class SeparatorOption(Enum):
    dot = "."
    """Dot separator."""
    comma = ","
    """Comma separator."""
    semicolon = ";"
    """Semicolon separator."""
    tab = "\t"
    """Tab separator."""
    space = " "
    """Space separator."""
    colon = ":"
    """Colon separator."""
    hyphen = "-"
    """Hyphen separator."""
    underscore = "_"
    """Underscore separator."""
    slash = "/"
    """Slash separator."""
    backslash = "\\"
    """Backslash separator."""
    pipe = "|"
    """Pipe separator."""
    hash = "#"
    """Hash separator."""
    apostrophe = "'"
    """Apostrophe separator."""


class MaccorPreprocessingOption(IntEnum):
    raw = -11  # historical
    """Raw data, must be read with the dll."""
    mims_client1 = 10
    """MIMS export file. From MIMS Options>View Data from File>File>Save as text."""
    maccor_export1 = 11
    """Maccor export file. Output of the macor export exe v1."""
    maccor_export2 = 12
    """Maccor export file. Output of the macor export exe v2."""
    mims_client2 = 13
    """MIMS Client version 2 export file."""
    mims_server2 = 14
    """MIMS Server version 2 export file."""


class DigatronPreprocessingOption(IntEnum):
    # raw_data = -21
    # """Raw data, must be retrieved from the database."""
    german_client_csv = 20
    """Digatron (German) client csv export file."""


class Sample(BaseModel):
    name: str
    """Sample name."""
    capacity: Capacity
    """Rated or theoretical capacity of the sample."""
    active_material_mass: Mass
    """Active material mass of the sample, which will be used for normalization."""
    three_electrode_cell: bool = False
    """Whether the cell is a three-electrode cell."""


class FilterSelectionOption(Enum):
    none = 0
    savitzky_golay = 1
    """Savitzky Golay filter"""
    gaussian = 2
    """Gaussian filter"""
    median = 3
    """Median filter"""
    adjacent_average = 4
    """Adjacent average"""


class FilterParam(BaseModel):
    filter_selection: FilterSelectionOption = FilterSelectionOption.none
    """Choose filter to be used for smoothing of dQ/dV result:
    0 = none, 1 = Savitzky Golay, 2 = Gaussian,
    3 = median filter, 4 = adjacent average"""
    window_size: int
    """Filter window size. Enter uneven integer. Does not apply for Gaussian filter.
    Default: 5 points """
    sigma_or_polynomial_order: int
    """For Gaussian filter, enter sigma. For Savitzky Golay, enter polynomial order."""


class CyclesToPlotOption(Enum):
    all = "all"


def parse_cycles_to_plot(cycles_to_plot: str) -> Union[CyclesToPlotOption, List[int]]:
    if cycles_to_plot == "all":
        return CyclesToPlotOption.all
    result = []
    if ";" in cycles_to_plot:
        cycles_to_plot = cycles_to_plot.replace(";", ",")
    parts = cycles_to_plot.split(",")
    for part in parts:
        if "-" in part:
            start, end = part.split("-")
            result.extend(list(range(int(start), int(end) + 1)))
        else:
            result.append(int(part))
    return result


def get_model_field_default_value(model: BaseModel, field: str):
    return model.model_fields[field].default


# todo: include data model / avoid redundant definitons
