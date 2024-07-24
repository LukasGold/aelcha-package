from enum import Enum

from core import DataModel, DateTime
from items import Device, PhysicalItem
from typing_extensions import Dict, List, Optional

from aelcha.data_model.qaq_data import (
    Quality,
    QuantityAnnotation,
    TabularData,
    TabularDataMetadata,
)
from aelcha.data_model.quantities import current, time, voltage
from aelcha.data_model.units_of_measure import ampere, second, volt


class BatteryCellFormFactor(Enum):
    cylindrical = "cylindrical"
    prismatic = "prismatic"
    pouch = "pouch"
    coin = "coin"
    button = "button"
    other = "other"

    def __str__(self):
        return self.value


class BatteryCellFormat(DataModel):
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
