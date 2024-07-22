import datetime
import uuid as uuid_module
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from pydantic._internal._model_construction import ModelMetaclass
from typing_extensions import ClassVar, List, Optional, Type, TypeVar, Union

# To be able to reference the type of the class itself
T = TypeVar("T", bound=BaseModel)


# Schema server
SERVER = "https://osw-schema.org"


class PropertyTypeOption(Enum):
    """According to https://sintef.github.io/dlite/user_guide/type-system.html"""

    # blob = "blob"
    bool = "bool"
    int = "int"
    # uint = "uint"
    float = "float"
    ufloat = "ufloat"
    # fixstring = "fixstring"
    # ref = "ref"
    string = "string"
    # relation = "relation"
    # dimension = "dimension"
    # property = "property"
    # array = "array"  # not in the documentation of DLite


TYPE_OPTION = Union[PropertyTypeOption, Type[int], Type[float], Type[str], Type[bool]]


class SemanticProperty(BaseModel):
    name: str
    description: str
    ontology_iris: List[str]
    subject_range: List[str]  # Type of Instance
    object_range: List[str]  # Type of Instance
    uuid: Optional[UUID] = Field(default_factory=uuid_module.uuid4)


class SemanticVersioning(BaseModel):
    major: int
    minor: int
    patch: int
    _version: PrivateAttr(str) = None

    def __init__(self, **data):
        super().__init__(**data)
        self._version = f"{self.major}.{self.minor}.{self.patch}"

    def __str__(self):
        version = f"{self.major}.{self.minor}.{self.patch}"
        return version


class Date(BaseModel):
    year: int
    month: int
    day: int


class DateTime(Date):
    hour: int
    minute: int
    second: float

    @staticmethod
    def now():
        return from_datetime(dt=datetime.datetime.now())


def from_datetime(dt: datetime.datetime):
    return DateTime(
        year=dt.year,
        month=dt.month,
        day=dt.day,
        hour=dt.hour,
        minute=dt.minute,
        second=dt.second + dt.microsecond / 1_000_000,
    )


class ModelUri(BaseModel):
    """Create a URI from the metadata. The 'name' suffix is included solely for
    readability."""

    namespace: str
    version: SemanticVersioning
    uuid: UUID
    name: str

    def __str__(self):
        return f"{SERVER}/{self.namespace}/{self.version}/{self.uuid}/{self.name}"


class Metadata(BaseModel):
    """Or 'DataModel'.

    An instance of this class is a metadata object, which is used to store the
    properties of a class or instance of a class (a data model).
    """

    name: str
    """The name of the class or instance"""
    description: Optional[str] = None
    """A description of the class or instance"""
    ontology_equivalent_iris: Optional[List[str]] = None
    """The IRIs of equivalent terms in other ontologies"""
    meta: Optional["Metadata"] = None
    """Metadata on the instance of this class or subclass ( Metadata / DataModel)."""
    version: Optional[SemanticVersioning] = SemanticVersioning(
        major=0, minor=0, patch=1
    )
    namespace: Optional[str] = "aelcha"
    """Defines the context or domain for this metadata."""
    uuid: Optional[UUID] = Field(default_factory=uuid_module.uuid4)
    """A unique identifier."""
    uri: Optional[Union[str, ModelUri]] = None
    """A unique identifier. If not provided, it will be generated from namespace,
    version, UUID and name. Have a look at SINTEFs DLite for inspiration."""
    datetime: Optional[Union[Date, DateTime]] = Field(default_factory=DateTime.now)
    """The date and time of the creation of the instance of this class."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        super().__init__(**data)
        # todo: rework
        if self.uri is None:
            self.uri = ModelUri(
                namespace=self.namespace,
                version=self.version,
                uuid=self.uuid,
                name=self.name,
            )
        if self.version is None:  # todo: implement automated versioning -> remote
            #  server / git
            self.version = SemanticVersioning(major=0, minor=0, patch=1)
        if self.meta is None:
            if self.__class__ != Metadata:
                self.meta = Metadata(
                    name=self.name, version=self.version, namespace=self.namespace
                )


class ClassMetadata(Metadata):
    """Class to hold information on a class in a structured way to be used
    in a class variable. An Instance of this (meta)class is a class definition,
    which is used to store the properties of the class.

    Design decisions
    ----------------
    * This class is a subclass of Metadata, because it is a metadata
    * New class properties that are common to all classes can be added here
    * For properties that are specific to a class, a new child of MetaClass should be
      defined

    """

    parents: Optional[List[Type[T]]] = Field(default=[])
    """List of parent classes. If not provided, it will be inferred from the class
    definition."""


class PythonMetaclass(ModelMetaclass):
    """Metaclass to implement class metadata inheritance in Python, required for but
    not part of the data model.
    ."""

    def __new__(cls, name, bases, attrs):
        if "class_meta" in attrs:
            # Get the parents of this class and set class_meta.parents to this list
            attrs["class_meta"].parents = [
                parent.class_meta for parent in bases if hasattr(parent, "class_meta")
            ]
        return super().__new__(cls, name, bases, attrs)


# metadata= ??
class Entity(Metadata, metaclass=PythonMetaclass):
    """Instances of this class are entities. This class is an Instance of MetaClass.

    Resources
    ---------
    https://medium.com/@miguel.amezola/demystifying-python-metaclasses-understanding-and
    -harnessing-the-power-of-custom-class-creation-d7dff7b68de8
    """

    class_meta: ClassVar[ClassMetadata] = ClassMetadata(
        name="Entity",
        version=SemanticVersioning(major=0, minor=0, patch=1),
        namespace="aelcha",
    )
    """This variable is a class variable, which holds the information about this data
    model (type: ClassMetadata). This equals the jsondata slot of a OSW Category."""
    type: Optional[List[Type[BaseModel]]] = None
    # todo: does not document the version, namespace, etc. of the class, yet :)

    def __init__(self, **data):
        super().__init__(**data)
        if self.type is None:
            self.type = [self.__class__]
        elif self.__class__ not in self.type:
            self.type.append(self.__class__)
