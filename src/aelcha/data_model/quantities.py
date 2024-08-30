from aelcha.data_model.qaq_data import Quantity, QuantityKindDimensionVector
from aelcha.data_model.units_of_measure import (
    ampere,
    ampere_hour,
    milli_ampere_hour,
    milli_volt,
    second,
    volt,
    watt_hour,
)

capacity = Quantity(
    name="Capacity",
    data_type=float,
    applicable_units=[ampere_hour, milli_ampere_hour],
    dimension_vector=QuantityKindDimensionVector(electric_current=1),
)
current = Quantity(
    name="Current",
    data_type=float,
    applicable_units=[ampere],
    dimension_vector=QuantityKindDimensionVector(electric_current=1),
)
energy = Quantity(
    name="Energy",
    data_type=float,
    applicable_units=[watt_hour],
    dimension_vector=QuantityKindDimensionVector(time=-2, length=2, mass=1),
)
time = Quantity(
    name="Time",
    data_type=float,
    applicable_units=[second],
    dimension_vector=QuantityKindDimensionVector(time=1),
)
voltage = Quantity(
    name="Voltage",
    data_type=float,
    applicable_units=[volt, milli_volt],
    dimension_vector=QuantityKindDimensionVector(
        time=-3, length=2, mass=1, electric_current=-1
    ),
)
