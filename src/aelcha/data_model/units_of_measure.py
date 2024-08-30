from aelcha.data_model.qaq_data import (
    QuantityKindDimensionVector,
    UnitOfMeasure,
    UnitPrefixOption,
)

none = UnitOfMeasure(
    name="None",
    description="No unit",
    symbol="",
    dimension_vector=QuantityKindDimensionVector(),
)
ampere = UnitOfMeasure(
    name="Ampere",
    description="SI base unit for electrical current",
    symbol="A",
    dimension_vector=QuantityKindDimensionVector(electric_current=1),
)
ampere_hour = UnitOfMeasure(
    name="Ampere hour",
    description="Unit of electric charge",
    symbol="Ah",
    dimension_vector=QuantityKindDimensionVector(electric_current=1, time=1),
)
milli_ampere_hour = UnitOfMeasure(
    name="Milliampere hour",
    description="Unit of electric charge",
    symbol="mAh",
    dimension_vector=QuantityKindDimensionVector(electric_current=1, time=1),
    non_prefixed_unit=ampere_hour,
    prefix=UnitPrefixOption.milli,
)

second = UnitOfMeasure(
    name="Second",
    description="SI base unit for time",
    symbol="s",
    dimension_vector=QuantityKindDimensionVector(time=1),
)
volt = UnitOfMeasure(
    name="Volt",
    symbol="V",
    description="SI unit for electric potential difference",
    dimension_vector=QuantityKindDimensionVector(
        time=-3, length=2, mass=1, electric_current=-1
    ),
)
milli_volt = UnitOfMeasure(
    name="Millivolt",
    description="Unit of electric potential difference",
    symbol="mV",
    non_prefixed_unit=volt,
    prefix=UnitPrefixOption.milli,
)
watt_hour = UnitOfMeasure(
    name="Watt hour",
    description="Unit of energy",
    symbol="Wh",
    dimension_vector=QuantityKindDimensionVector(time=-2, length=2, mass=1),
)
