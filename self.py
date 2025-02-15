"""Decorators for class methods."""

import logging
from typing import Union, Iterable, Callable
from functools import wraps

logger = logging.getLogger(__name__)


class Self:
    """Decorator to allow passing an object itself to methods that actually take some attribute of that object."""
    def __init__(self, attr: str, cls: Union[Iterable, str] = None):
        self.attr = attr
        self.cls = {cls,} if isinstance(cls, str) else cls

    def _filter(self, cls: type):
        return bool({c.__name__ for c in cls.mro()} & set(self.cls))

    def __call__(self, func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*[getattr(arg, self.attr) if self._filter(arg.__class__) else arg for arg in args], **kwargs)
        return wrapper

    @staticmethod
    def inplace(attr: str):
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(self, *args, inplace: bool = True, return_self: bool = False, **kwargs):
                r = func(self, *args, **kwargs)
                if inplace:
                    setattr(self, attr, r)
                    if return_self:
                        return self
                return r
            return wrapper
        return decorator


# class SelfMixin(Self):
#     """Mixin class that provides a class method as decorator to allow passing the object itself to the wrapped functions that actually take some attribute of that object."""
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     @classmethod
#     def _filter(self, cls: type):
#         return issubclass(cls, self)
