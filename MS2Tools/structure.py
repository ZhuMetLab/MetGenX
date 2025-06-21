# -*- coding: utf-8 -*-

#########################################################################################
# Partially obtained and modified from pyteomics (https://github.com/levitsky/pyteomics)
# under apache 2.0 license (https://github.com/levitsky/pyteomics/blob/master/LICENSE)
##########################################################################################
_UNIT_CV_INTERN_TABLE = {}
class _MappingOverAttributeProxy:
    """A replacement for __dict__ for unpickling an object which once
    has __slots__ now but did not before."""

    def __init__(self, obj):
        self.obj = obj

    def __getitem__(self, key):
        return getattr(self.obj, key)

    def __setitem__(self, key, value):
        setattr(self.obj, key, value)

    def __contains__(self, key):
        return hasattr(self.obj, key)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.obj})"


class UnitInt(int):
    """
    Represents an integer value with a unit name.

    Behaves identically to a built-in :class:`int` type.

    Attributes
    ----------
    unit_info : :class:`str`
        The name of the unit this value possesses.
    """

    def __new__(cls, value, unit_info=None):
        inst = int.__new__(cls, value)
        inst.unit_info = unit_info
        return inst

    def __reduce__(self):
        return self.__class__, (int(self), self.unit_info)

    def _repr_pretty_(self, p, cycle):
        base = super().__repr__()
        string = f"{base} {self.unit_info}" if self.unit_info else base
        p.text(string)


class UnitFloat(float):
    """
    Represents a float value with a unit name.

    Behaves identically to a built-in :class:`float` type.

    Attributes
    ----------
    unit_info : :class:`str`
        The name of the unit this value possesses.
    """

    __slots__ = ("unit_info",)

    def __new__(cls, value, unit_info=None):
        inst = float.__new__(cls, value)
        inst.unit_info = unit_info
        return inst

    @property
    def __dict__(self):
        return _MappingOverAttributeProxy(self)

    def __reduce__(self):
        return self.__class__, (float(self), self.unit_info)

    def _repr_pretty_(self, p, cycle):
        base = super().__repr__()
        string = f"{base} {self.unit_info}" if self.unit_info else base
        p.text(string)

