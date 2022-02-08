creators = [
    "arange",  # (start, /, stop=None, step=1, *, dtype=None, device=None)
    "asarray",  # (obj, /, *, dtype=None, device=None, copy=None)
    "empty",
    "empty_like",  # (x, /, *, dtype=None, device=None)
    "eye",  # (n_rows, n_cols=None, /, *, k=0, dtype=None, device=None)
    "from_dlpack",  # (x, /)
    "full",
    "full_like",  # (x, /, fill_value, *, dtype=None, device=None)
    "linspace",  # (start, stop, /, num, *, dtype=None, device=None, endpoint=True)
    "meshgrid",  # (*arrays, indexing=’xy’)
    "ones",
    "ones_like",  # (x, /, *, dtype=None, device=None)
    "zeros",
    "zeros_like",  # (x, /, *, dtype=None, device=None)
]

ew_unary_methods = [
    "__abs__",  # (self, /)
    "__invert__",  # (self, /)
    "__neg__",  # (self, /)
    "__pos__",  # (self, /)
]

ew_unary_ops = [
    "abs",  # (x, /)
    "acos",  # (x, /)
    "acosh",  # (x, /)
    "asin",  # (x, /)
    "asinh",  # (x, /)
    "atan",  # (x, /)
    "atanh",  # (x, /)
    "bitwise_invert",  # (x, /)
    "ceil",  # (x, /)
    "cos",  # (x, /)
    "cosh",  # (x, /)
    "exp",  # (x, /)
    "expm1",  # (x, /)
    "floor",  # (x, /)
    "isfinite",  # (x, /)
    "isinf",  # (x, /)
    "isnan",  # (x, /)
    "logical_not",  # (x, /)
    "log",  # (x, /)
    "log1p",  # (x, /)
    "log2",  # (x, /)
    "log10",  # (x, /)
    "negative",  # (x, /)
    "positive",  # (x, /)
    "round",  # (x, /)
    "sign",  # (x, /)
    "sin",  # (x, /)
    "sinh",  # (x, /)
    "square",  # (x, /)
    "sqrt",  # (x, /)
    "tan",  # (x, /)
    "tanh",  # (x, /)
    "trunc",  # (x, /)
]

ew_binary_methods_inplace = [
    # inplace operators
    "__iadd__",
    "__iand__",
    "__ifloordiv__",
    "__ilshift__",
    "__imod__",
    "__imul__",
    "__ior__",
    "__ipow__",
    "__irshift__",
    "__isub__",
    "__itruediv__",
    "__ixor__",
]

ew_binary_methods = [
    "__add__",  # (self, other, /)
    "__and__",  # (self, other, /)
    "__eq__",  # (self, other, /)
    "__floordiv__",  # (self, other, /)
    "__ge__",  # (self, other, /)
    "__gt__",  # (self, other, /)
    "__le__",  # (self, other, /)
    "__lshift__",  # (self, other, /)
    "__lt__",  # (self, other, /)
    "__matmul__",  # (self, other, /)
    "__mod__",  # (self, other, /)
    "__mul__",  # (self, other, /)
    "__ne__",  # (self, other, /)
    "__or__",  # (self, other, /)
    "__pow__",  # (self, other, /)
    "__rshift__",  # (self, other, /)
    "__sub__",  # (self, other, /)
    "__truediv__",  # (self, other, /)
    "__xor__",  # (self, other, /)
    # reflected operators
    "__radd__",
    "__rand__",
    "__rfloordiv__",
    "__rlshift__",
    "__rmod__",
    "__rmul__",
    "__ror__",
    "__rpow__",
    "__rrshift__",
    "__rsub__",
    "__rtruediv__",
    "__rxor__",
]

ew_binary_ops = [
    "add",  # (x1, x2, /)
    "atan2",  # (x1, x2, /)
    "bitwise_and",  # (x1, x2, /)
    "bitwise_left_shift",  # (x1, x2, /)
    "bitwise_or",  # (x1, x2, /)
    "bitwise_right_shift",  # (x1, x2, /)
    "bitwise_xor",  # (x1, x2, /)
    "divide",  # (x1, x2, /)
    "equal",  # (x1, x2, /)
    "floor_divide",  # (x1, x2, /)
    "greater",  # (x1, x2, /)
    "greater_equal",  # (x1, x2, /)
    "less_equal",  # (x1, x2, /)
    "logaddexp",  # (x1, x2)
    "logical_and",  # (x1, x2, /)
    "logical_or",  # (x1, x2, /)
    "logical_xor",  # (x1, x2, /)
    "multiply",  # (x1, x2, /)
    "less",  # (x1, x2, /)
    "not_equal",  # (x1, x2, /)
    "pow",  # (x1, x2, /)
    "remainder",  # (x1, x2, /)
    "subtract",  # (x1, x2, /)
]
