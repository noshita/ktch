"""
Deprecated module: ktch.outline

This module has been renamed to ktch.harmonic.
Please update your imports accordingly.

This compatibility shim will be removed in version 0.9.
"""

import warnings

warnings.filterwarnings(
    "once",
    category=FutureWarning,
    module=r"ktch\.outline",
)

warnings.warn(
    "The 'outline' module is deprecated and will be removed in version 0.9. "
    "Use 'harmonic' instead.",
    FutureWarning,
    stacklevel=2,
)


from ..harmonic import *

try:
    from ktch.harmonic import __all__
except ImportError:
    pass
