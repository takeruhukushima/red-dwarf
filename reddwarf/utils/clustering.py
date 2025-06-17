import warnings
import sys

# Show where this was imported from
caller = sys._getframe(1).f_globals.get("__file__", "<unknown>")

# Issue deprecation warning
warnings.warn(
    f"Importing from 'reddwarf.utils.clustering' is deprecated and will be removed in future versions. "
    f"Please import 'reddwarf.utils.clusterer.clustering' instead. (imported from {caller})",
    DeprecationWarning,
    stacklevel=2,
)

# Import everything from the new location
from reddwarf.utils.clusterer.clustering import *
