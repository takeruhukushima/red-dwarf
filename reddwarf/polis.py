import warnings
import sys

# Show where this was imported from
caller = sys._getframe(1).f_globals.get('__file__', '<unknown>')

# Issue deprecation warning
warnings.warn(
    f"Importing from 'reddwarf.polis' is deprecated and will be removed in future versions. "
    f"Please import 'reddwarf.implementations.polis_legacy' instead, or better still, "
    f"use `reddwarf.implementations.polis`. (imported from {caller})",
    DeprecationWarning,
    stacklevel=2
)

# Import everything from the new location
from reddwarf.implementations.polis_legacy import *