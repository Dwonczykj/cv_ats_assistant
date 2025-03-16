from typing import TypedDict, get_type_hints, Type
from typing import Literal
from dataclasses import dataclass

def keys_of(cls: Type) -> Type:
    """Return a Literal type with keys of the given class."""
    return Literal[tuple(get_type_hints(cls).keys())]

@dataclass
class Example:
    name: str
    age: int

# Get the keys as a Literal type
ExampleKeys = keys_of(Example)

# ExampleKeys is now equivalent to Literal['name', 'age']
def create_typed_dict_from_class(cls: Type) -> Type[TypedDict]:
    """Create a TypedDict from a class or dataclass."""
    fields = get_type_hints(cls)
    return TypedDict(cls.__name__ + "Dict", {k: v for k, v in fields.items()})

ExampleDict = create_typed_dict_from_class(Example)

# ExampleDict is now equivalent to:
# class ExampleDict(TypedDict):
#     name: str
#     age: int