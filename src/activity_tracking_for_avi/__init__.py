"""
Activity tracking for AVI files napari plugin.
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.1.0"

from ._reader import napari_get_reader
from ._widget import ActivityAnalysisWidget

__all__ = (
    "ActivityAnalysisWidget",
    "napari_get_reader",
)
