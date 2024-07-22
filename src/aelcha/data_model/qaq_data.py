"""Module on quantitative and qualitative (qaq) data (models)."""

from enum import Enum

import numpy as np
import pandas as pd
import uncertainties as unc
from pydantic import BaseModel, Field, PrivateAttr, field_validator
from typing_extensions import Any, ClassVar, Dict, List, Optional, Union

from aelcha.data_model.core import (
    TYPE_OPTION,
    ClassMetadata,
    Metadata,
    SemanticProperty,
    SemanticVersioning,
)
from aelcha.data_model.items import Item


class UnitPrefix(BaseModel):
    name: str
    description: str
    symbol: str
    uri: str
    conversion_factor: float


class QuantityKindDimensionVector(Metadata):
    """Dimensional exponents of the quantity kind. Defines the relation of a quantity
    to the base quantities of a system of quantities as a product of factors
    corresponding to the base quantities, omitting any numerical factor.
    (According to VIM)"""

    name: str = None
    """Overwrite to save trouble, will be auto generated from the exponents."""
    time: float = 0
    """T dimensional exponent"""
    length: float = 0
    """L dimensional exponent"""
    mass: float = 0
    """M dimensional exponent"""
    electric_current: float = 0
    """I dimensional exponent"""
    thermodynamic_temperature: float = 0
    """Θ dimensional exponent"""
    amount_of_substance: float = 0
    """N dimensional exponent"""
    luminous_intensity: float = 0
    """J dimensional exponent"""

    def __init__(self, **data):
        name = data.get("name", None)
        if name is None:
            name = ""
            for base_quantity, symbol in {
                "time": "T",
                "length": "L",
                "mass": "M",
                "electric_current": "I",
                "thermodynamic_temperature": "Θ",
                "amount_of_substance": "N",
                "luminous_intensity": "J",
            }.items():
                exponent = data.get(base_quantity, 0)
                if exponent != 0:
                    sign = "" if exponent > 0 else "-"
                    abs_val = abs(exponent)
                    if exponent == 1:
                        name = name + f"{symbol}"
                    elif abs(exponent) == 0.5:
                        name = name + f"{symbol}^{sign}1/2"
                    else:
                        name = name + f"{symbol}^{sign}{abs_val}"
        super().__init__(name=name, **data)


class UnitPrefixOption(Enum):
    quetta = UnitPrefix(
        name="quetta",
        description="Quetta prefix",
        symbol="Q",
        uri="http://qudt.org/vocab/unit/PX_Q",
        conversion_factor=1e30,
    )
    ronna = UnitPrefix(
        name="ronna",
        description="Ronna prefix",
        symbol="R",
        uri="http://qudt.org/vocab/unit/PX_R",
        conversion_factor=1e27,
    )
    yotta = UnitPrefix(
        name="yotta",
        description="Yotta prefix",
        symbol="Y",
        uri="http://qudt.org/vocab/unit/PX_Y",
        conversion_factor=1e24,
    )
    zetta = UnitPrefix(
        name="zetta",
        description="Zetta prefix",
        symbol="Z",
        uri="http://qudt.org/vocab/unit/PX_Z",
        conversion_factor=1e21,
    )
    exa = UnitPrefix(
        name="exa",
        description="Exa prefix",
        symbol="E",
        uri="http://qudt.org/vocab/unit/PX_E",
        conversion_factor=1e18,
    )
    peta = UnitPrefix(
        name="peta",
        description="Peta prefix",
        symbol="P",
        uri="http://qudt.org/vocab/unit/PX_P",
        conversion_factor=1e15,
    )
    tera = UnitPrefix(
        name="tera",
        description="Tera prefix",
        symbol="T",
        uri="http://qudt.org/vocab/unit/PX_T",
        conversion_factor=1e12,
    )
    giga = UnitPrefix(
        name="giga",
        description="Giga prefix",
        symbol="G",
        uri="http://qudt.org/vocab/unit/PX_G",
        conversion_factor=1e9,
    )
    mega = UnitPrefix(
        name="mega",
        description="Mega prefix",
        symbol="M",
        uri="http://qudt.org/vocab/unit/PX_M",
        conversion_factor=1e6,
    )
    kilo = UnitPrefix(
        name="kilo",
        description="Kilo prefix",
        symbol="k",
        uri="http://qudt.org/vocab/unit/PX_k",
        conversion_factor=1e3,
    )
    hecto = UnitPrefix(
        name="hecto",
        description="Hecto prefix",
        symbol="h",
        uri="http://qudt.org/vocab/unit/PX_h",
        conversion_factor=1e2,
    )
    deca = UnitPrefix(
        name="deca",
        description="Deca prefix",
        symbol="da",
        uri="http://qudt.org/vocab/unit/PX_da",
        conversion_factor=1e1,
    )
    none = UnitPrefix(
        name="none",
        description="No prefix",
        symbol="",
        uri="http://qudt.org/vocab/unit/PX_",
        conversion_factor=1,
    )
    deci = UnitPrefix(
        name="deci",
        description="Deci prefix",
        symbol="d",
        uri="http://qudt.org/vocab/unit/PX_d",
        conversion_factor=1e-1,
    )
    centi = UnitPrefix(
        name="centi",
        description="Centi prefix",
        symbol="c",
        uri="http://qudt.org/vocab/unit/PX_c",
        conversion_factor=1e-2,
    )
    milli = UnitPrefix(
        name="milli",
        description="Milli prefix",
        symbol="m",
        uri="http://qudt.org/vocab/unit/PX_m",
        conversion_factor=1e-3,
    )
    micro = UnitPrefix(
        name="micro",
        description="Micro prefix",
        symbol="μ",
        uri="http://qudt.org/vocab/unit/PX_μ",
        conversion_factor=1e-6,
    )
    nano = UnitPrefix(
        name="nano",
        description="Nano prefix",
        symbol="n",
        uri="http://qudt.org/vocab/unit/PX_n",
        conversion_factor=1e-9,
    )
    pico = UnitPrefix(
        name="pico",
        description="Pico prefix",
        symbol="p",
        uri="http://qudt.org/vocab/unit/PX_p",
        conversion_factor=1e-12,
    )
    femto = UnitPrefix(
        name="femto",
        description="Femto prefix",
        symbol="f",
        uri="http://qudt.org/vocab/unit/PX_f",
        conversion_factor=1e-15,
    )
    atto = UnitPrefix(
        name="atto",
        description="Atto prefix",
        symbol="a",
        uri="http://qudt.org/vocab/unit/PX_a",
        conversion_factor=1e-18,
    )
    zepto = UnitPrefix(
        name="zepto",
        description="Zepto prefix",
        symbol="z",
        uri="http://qudt.org/vocab/unit/PX_z",
        conversion_factor=1e-21,
    )
    yocto = UnitPrefix(
        name="yocto",
        description="Yocto prefix",
        symbol="y",
        uri="http://qudt.org/vocab/unit/PX_y",
        conversion_factor=1e-24,
    )
    ronto = UnitPrefix(
        name="ronto",
        description="Ronto prefix",
        symbol="r",
        uri="http://qudt.org/vocab/unit/PX_r",
        conversion_factor=1e-27,
    )
    quecto = UnitPrefix(
        name="quecto",
        description="Quecto prefix",
        symbol="q",
        uri="http://qudt.org/vocab/unit/PX_q",
        conversion_factor=1e-30,
    )


class UnitOfMeasure(Metadata):
    symbol: str
    dimension_vector: Optional[QuantityKindDimensionVector] = None
    non_prefixed_unit: Optional["UnitOfMeasure"] = None
    prefix: Optional[UnitPrefixOption] = None

    def __init__(self, **data):
        super().__init__(**data)
        if self.non_prefixed_unit is None:
            if self.dimension_vector is None:
                raise ValueError(
                    "Either non_prefixed_unit or dimension_vector must be " "provided."
                )
        else:
            self.dimension_vector = self.non_prefixed_unit.dimension_vector

    @field_validator("prefix")
    def validate_prefix(cls, prefix, values):
        if prefix is None:
            return
        if prefix == UnitPrefixOption.none:
            if values["non_prefixed_unit"] is not None:
                raise ValueError("Non-prefixed unit must be None if prefix is 'none'.")
        else:
            if values["non_prefixed_unit"] is None:
                raise ValueError(
                    "Non-prefixed unit must be provided if prefix is not " "'none'."
                )


class Percentage(BaseModel):
    value: float = Field(..., ge=0, le=100)


class Uncertainty(BaseModel):
    symmetric: Optional[Union[float, Percentage, List[Union[float, Percentage]]]] = None
    lower_bound: Optional[Union[float, Percentage, List[Union[float, Percentage]]]] = (
        None
    )
    upper_bound: Optional[Union[float, Percentage, List[Union[float, Percentage]]]] = (
        None
    )

    @field_validator("symmetric")
    def validate_symmetric(cls, symmetric, values):
        if symmetric is None:
            return
        if values["lower_bound"] is not None or values["upper_bound"] is not None:
            raise ValueError(
                "Symmetric uncertainty cannot be provided if lower_bound "
                "or upper_bound is provided."
            )

    @field_validator("lower_bound")
    def validate_lower_bound(cls, lower_bound, values):
        if lower_bound is None:
            return
        if values["symmetric"] is not None:
            raise ValueError(
                "Lower bound uncertainty cannot be provided if symmetric "
                "uncertainty is provided."
            )
        if values["upper_bound"] is None:
            raise ValueError(
                "Upper bound has to be provided as well if lower bound is " "provided."
            )

    @field_validator("upper_bound")
    def validate_upper_bound(cls, upper_bound, values):
        if upper_bound is None:
            return
        if values["symmetric"] is not None:
            raise ValueError(
                "Upper bound uncertainty cannot be provided if symmetric "
                "uncertainty is provided."
            )
        if values["lower_bound"] is None:
            raise ValueError(
                "Lower bound has to be provided as well if upper bound is " "provided."
            )


# DRAFT
class Quality(Metadata):  # todo: discuss if parent should be entity
    """Quality of a thing or concept. Examples:
    Category:Person
    * Gender -> Property
    * Age / Weight / Height -> Quantity
    * PostalAddress --> Characteristic (composed of multiple qualities, in this case
                        Properties)
    * HeightTimeSeries --> Characteristic (composed of multiple qualities, in this case
                           Quantities, point in time + height [length])

    """

    data_type: TYPE_OPTION
    """Used for storage of the quality. E.g., when converting to json."""
    semantic_property: Optional[List[SemanticProperty]] = None
    """Semantic property of the quality."""


class Quantity(Quality):
    """Quantity(Kind in QUDT) of Measure"""

    applicable_units: List[UnitOfMeasure]
    dimension_vector: QuantityKindDimensionVector
    broader: Optional[List["Quantity"]] = None  # todo: list or single?
    narrower: Optional[List["Quantity"]] = None


class Property(Quality):
    pass


class QuantityAnnotation(Metadata):
    """Used to, e.g., describe the content of a column in a tabular data set."""

    uncertainty: Optional[Union[float, Uncertainty]] = None
    """Uncertainty of the quantity - Should be a list of plus, minus, upper and lower
    bound (flat and percentage)"""
    unit: Optional[UnitOfMeasure] = None
    """Unit of the quantity. Will be checked against the selected quantity kinds
    applicable units."""  # todo
    quantity_kind: Optional[Quantity] = None
    """Kind of quantity."""
    data_type: TYPE_OPTION
    """Data type used to store values of the quantity annotated by the instance of
    this class."""
    semantic_property: Optional[SemanticProperty] = None
    """Semantic property of the quantity."""

    def __init__(self, **data):
        super().__init__(**data)
        if isinstance(self.uncertainty, float):
            self.uncertainty = Uncertainty(symmetric=self.uncertainty)


class QuantityValue(QuantityAnnotation):
    """Synonymous to 'Quantity' - a physical quantity with a numerical value, an
    uncertainty and a unit."""

    numerical_value: float
    """Numerical value of the quantity."""
    _bare_value_str: PrivateAttr(str) = None
    """String representation of the quantity."""
    _uvalue: PrivateAttr(str) = None
    """Value of the quantity with uncertainty."""
    _uvalue_str: PrivateAttr(str) = None
    """String representation of the quantity."""

    def __init__(self, **data):
        super().__init__(**data)
        self._bare_value_str = f"{self.numerical_value} {self.unit}"
        # todo: needs rework according to uncertainty package and new Uncertainty model
        # todo: consider using PINTS (which uses uncertainties)
        self._uvalue = unc.ufloat(self.numerical_value, self.uncertainty)
        self._uvalue_str = f"({self.numerical_value} ± {self.uncertainty}) {self.unit}"


# DRAFT


class Data(Item):
    """Data container"""

    # name will be required!
    class_meta: ClassVar[ClassMetadata] = ClassMetadata(
        name="Data",
        version=SemanticVersioning(major=0, minor=0, patch=1),
        namespace="aelcha",
    )
    data: Any
    """The data contained"""


class TabularDataMetadata(Metadata):
    name: str = "TabularDataMetadata"
    columns: Dict[str, Quality]
    """A dict of qualities, with quality = description of the content of a column,
    which can be properties, quantities or other characteristics."""


class TabularData(Data):
    """Tabular data container"""

    # name will be required!
    class_meta: ClassVar[ClassMetadata] = ClassMetadata(
        name="TabularData",
        version=SemanticVersioning(major=0, minor=0, patch=1),
        namespace="aelcha",
    )
    meta: TabularDataMetadata
    """Metadata of the tabular data."""
    data: Union[pd.DataFrame, np.ndarray]
    """Numerical data as a pandas DataFrame or numpy array."""

    class Config:
        arbitrary_types_allowed = True