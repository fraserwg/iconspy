# Package metadata
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("iconspy")
except PackageNotFoundError:
    __version__ = "0.0.0"  # fallback for local/dev usage

__author__ = "Fraser William Goldsworth"

# Import key functionality to simplify access
from .core import (
    TargetStation,
    WetModelStation,
    BoundaryModelStation,
    Section,
    LandSection,
    CombinedSection,
    Region
)

from .utils import convert_tgrid_data

# Define the package's public API
# __all__ = [
#     "process_icon",
#     "some_utility_function",
# ]
