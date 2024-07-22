from enum import Enum

from typing_extensions import Dict, List, Optional

from aelcha.data_model.core import DateTime, Metadata
from aelcha.data_model.items import Device, PhysicalItem
from aelcha.data_model.qaq_data import (
    Quality,
    Quantity,
    QuantityAnnotation,
    QuantityKindDimensionVector,
    TabularData,
    TabularDataMetadata,
    UnitOfMeasure,
)


class BatteryCellFormFactor(Enum):
    cylindrical = "cylindrical"
    prismatic = "prismatic"
    pouch = "pouch"
    coin = "coin"
    button = "button"
    other = "other"

    def __str__(self):
        return self.value


class BatteryCellFormat(Metadata):
    # name will be required!
    description: str
    form_factor: BatteryCellFormFactor
    dimensions: List[str]


class ElectrochemicalStorageDevice(PhysicalItem):
    pass


class ElectrochemicalCell(ElectrochemicalStorageDevice):
    pass


class BatteryCell(ElectrochemicalCell):
    cell_format: BatteryCellFormat


class BatteryModule(ElectrochemicalStorageDevice):
    cells: List[BatteryCell]


class BatteryPack(ElectrochemicalStorageDevice):
    modules: Optional[List[BatteryModule]] = None
    cells: Optional[List[BatteryCell]] = None
    """Use if cell to pack"""

    def __init__(self, **data):
        super().__init__(**data)
        if self.cells is None:
            if self.modules is None:
                raise ValueError("Either cells or modules must be provided.")
            self.cells = []
            for module in self.modules:
                self.cells.extend(module.cells)


second = UnitOfMeasure(
    name="second",
    description="SI base unit for time",
    symbol="s",
    dimension_vector=QuantityKindDimensionVector(time=1),
)
time = Quantity(
    name="time",
    data_type=float,
    applicable_units=[second],
    dimension_vector=QuantityKindDimensionVector(time=1),
)
volt = UnitOfMeasure(
    name="volt",
    symbol="V",
    dimension_vector=QuantityKindDimensionVector(
        time=-3, length=2, mass=1, electric_current=-1
    ),
)
voltage = Quantity(
    name="voltage",
    data_type=float,
    applicable_units=[volt],
    dimension_vector=QuantityKindDimensionVector(
        time=-3, length=2, mass=1, electric_current=-1
    ),
)
ampere = UnitOfMeasure(
    name="ampere",
    description="SI base unit for electrical current",
    symbol="A",
    dimension_vector=QuantityKindDimensionVector(electric_current=1),
)
current = Quantity(
    name="current",
    data_type=float,
    applicable_units=[ampere],
    dimension_vector=QuantityKindDimensionVector(electric_current=1),
)


class BatteryCyclingMetadata(TabularDataMetadata):
    """Defines the metadata for a battery cycling dataset. Includes:
    * Battery information
    * Measurement information
    * Cycling information / procedure / parameter / conditions
    """

    name: str = "BatteryCyclingMetadata"
    """Here a default name actually makes sense."""
    datetime: DateTime
    """Start time of the measurement"""
    dut_battery: ElectrochemicalStorageDevice
    """Device under test (DUT)"""
    climate_chamber: Optional[List[Device]] = None
    """Measurement tools"""
    battery_cycler: Optional[List[Device]] = None
    """Measurement tools"""
    columns: Dict[str, Quality] = {
        "index": QuantityAnnotation(
            name="index", description="The index of the measurement", data_type=int
        ),
        "time": QuantityAnnotation(
            name="time",
            description="Elapsed time since the start of the measurement",
            data_type=float,
            unit=second,
            quantity_kind=time,
        ),
        "voltage": QuantityAnnotation(
            name="voltage",
            description="Voltage of the battery",
            data_type=float,
            unit=volt,
            quantity_kind=voltage,
        ),
        "current": QuantityAnnotation(
            name="current",
            description="Current through the battery",
            data_type=float,
            unit=ampere,
            quantity_kind=current,
        ),
        # capacity
        # energy
        # step
        # cycle count
        # mode
    }


class BatteryCyclingData(TabularData):
    meta: BatteryCyclingMetadata
    """Information on the measurement process / the sample etc.
    cyling temperature: QuantityValue(numerical_value=25, quantity=temperature, unit=°C)
    """
    # data: Union[pd.DataFrame, np.ndarray]
