# Package metadata
__version__ = "0.1.0"
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
