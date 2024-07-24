from aelcha.data_model.qaq_data import QuantityKindDimensionVector, UnitOfMeasure

ampere = UnitOfMeasure(
    name="ampere",
    description="SI base unit for electrical current",
    symbol="A",
    dimension_vector=QuantityKindDimensionVector(electric_current=1),
)
second = UnitOfMeasure(
    name="second",
    description="SI base unit for time",
    symbol="s",
    dimension_vector=QuantityKindDimensionVector(time=1),
)
volt = UnitOfMeasure(
    name="volt",
    symbol="V",
    dimension_vector=QuantityKindDimensionVector(
        time=-3, length=2, mass=1, electric_current=-1
    ),
)
