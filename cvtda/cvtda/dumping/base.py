import abc
import typing

T = typing.TypeVar("T")

class BaseDumper(abc.ABC, typing.Generic[T]):
    current_dumper = None
    
    def __enter__(self):
        self.__previous = BaseDumper.current_dumper
        BaseDumper.current_dumper = self

    def __exit__(self, *args):
        BaseDumper.current_dumper = self.__previous

    @abc.abstractmethod
    def execute(self, function: typing.Callable[[typing.Any], T], name: str, *function_args) -> T:
        pass
