import typing

from .base import BaseDumper

T = typing.TypeVar("T")

class DevNullDumper(BaseDumper[T]):
    def execute(self, function: typing.Callable[[typing.Any], T], name: str, *function_args) -> T:
        return function(*function_args)
