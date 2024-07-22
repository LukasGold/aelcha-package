from typing_extensions import ClassVar

from aelcha.data_model.core import ClassMetadata, Entity, SemanticVersioning


class Item(Entity):
    class_meta: ClassVar[ClassMetadata] = ClassMetadata(
        name="Item",
        version=SemanticVersioning(major=0, minor=0, patch=1),
        namespace="aelcha",
    )


class PhysicalItem(Item):
    class_meta: ClassVar[ClassMetadata] = ClassMetadata(
        name="PhysicalItem",
        version=SemanticVersioning(major=0, minor=0, patch=1),
        namespace="aelcha",
    )


class Tool(Item):
    class_meta: ClassVar[ClassMetadata] = ClassMetadata(
        name="Tool",
        version=SemanticVersioning(major=0, minor=0, patch=1),
        namespace="aelcha",
    )


class Device(Tool, PhysicalItem):
    class_meta: ClassVar[ClassMetadata] = ClassMetadata(
        name="Device",
        version=SemanticVersioning(major=0, minor=0, patch=1),
        namespace="aelcha",
    )
