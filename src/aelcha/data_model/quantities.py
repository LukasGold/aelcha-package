from aelcha.data_model.qaq_data import Quantity, QuantityKindDimensionVector
from aelcha.data_model.units_of_measure import ampere, second, volt

time = Quantity(
    name="time",
    data_type=float,
    applicable_units=[second],
    dimension_vector=QuantityKindDimensionVector(time=1),
)
voltage = Quantity(
    name="voltage",
    data_type=float,
    applicable_units=[volt],
    dimension_vector=QuantityKindDimensionVector(
        time=-3, length=2, mass=1, electric_current=-1
    ),
)
current = Quantity(
    name="current",
    data_type=float,
    applicable_units=[ampere],
    dimension_vector=QuantityKindDimensionVector(electric_current=1),
)
