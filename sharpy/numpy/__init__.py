from .. import _selectSharding, empty, float32


def fromfunction(function, shape, *, dtype=float32, device="", sharding=None):
    t = empty(
        shape, dtype=dtype, device=device, sharding=_selectSharding(sharding)
    )
    t._t.map(function)
    return t
