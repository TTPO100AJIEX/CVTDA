import typing

from .base import BaseDumper
from .devnull import DevNullDumper
from .dump_name_concat import dump_name_concat
from .numpy import NumpyDumper

if typing.TYPE_CHECKING:
    from .universal import UniversalDumper

__all__ = [
    "BaseDumper",
    "DevNullDumper",
    "NumpyDumper",
    "UniversalDumper",
    "dump_name_concat",
]


def __getattr__(name: str):
    # Keep optional serializers out of the basic NumPy dumping API's imports.
    if name == "UniversalDumper":
        from .universal import UniversalDumper

        return UniversalDumper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def dumper() -> BaseDumper:
    """
    Returns
    -------
    ``BaseLogger``
        Current active dumper.
    """
    if BaseDumper.current_dumper is None:
        BaseDumper.current_dumper = NumpyDumper("./")
    return BaseDumper.current_dumper
