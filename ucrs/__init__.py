
"""Unified CRS (UCRS) class for seamless conversion between pyproj, cartopy, and osgeo."""

from __future__ import annotations

from functools import cached_property
from typing import cast
from typing import TypeAlias
from typing import TYPE_CHECKING

import pyproj

try:
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version("ucrs")
    except PackageNotFoundError:
        __version__ = "unknown"
except ImportError:
    __version__ = "unknown"

try:
    import cartopy.crs as ccrs
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False  # pyright: ignore[reportConstantRedefinition]

try:
    import osgeo
    from osgeo.osr import SpatialReference
    OSGEO_AVAILABLE = True
except ImportError:
    OSGEO_AVAILABLE = False  # pyright: ignore[reportConstantRedefinition]

# Type alias for accepted input types
if TYPE_CHECKING:
    import cartopy.crs as ccrs
    from osgeo.osr import SpatialReference

    CRSInput: TypeAlias = (
        pyproj.CRS
        | ccrs.CRS
        | ccrs.Projection
        | SpatialReference
        | str
        | int
        | dict[str, str]
    )


class UCRS:
    """Unified CRS for seamless conversion between pyproj, cartopy, and osgeo.

    Parameters
    ----------
    obj : CRSInput
        Can be anything that pyproj.crs.CRS.from_user_input() accepts, or
        instances of cartopy.crs.CRS, cartopy.crs.Projection, pyproj.CRS,
        or osgeo.osr.SpatialReference.

    Examples
    --------
    >>> ucrs = UCRS(4326)
    >>> ucrs = UCRS("EPSG:4326")
    >>> ucrs = UCRS(pyproj.CRS.from_epsg(4326))

    Access different CRS representations:
    >>> proj_crs = ucrs.proj
    >>> cart_crs = ucrs.cartopy  # if cartopy available
    >>> osgeo_crs = ucrs.osgeo   # if osgeo available
    """

    def __init__(self, obj: CRSInput) -> None:
        """Initialize UCRS from various CRS representations."""
        self._original: CRSInput = obj

        # Convert input to pyproj.CRS
        match obj:
            case pyproj.CRS():
                self._pyproj_crs: pyproj.CRS = obj

            case _ if OSGEO_AVAILABLE and isinstance(obj, SpatialReference):
                # Convert from osgeo to pyproj
                wkt: str
                if osgeo.version_info.major < 3:
                    wkt = cast(str, obj.ExportToWkt())
                else:
                    wkt = cast(str, obj.ExportToWkt(["FORMAT=WKT2_2018"]))
                self._pyproj_crs = pyproj.CRS.from_wkt(wkt)

            case _ if CARTOPY_AVAILABLE and isinstance(obj, (ccrs.CRS, ccrs.Projection)):
                # cartopy.crs.CRS inherits from pyproj.CRS, so use from_user_input
                self._pyproj_crs = pyproj.CRS.from_user_input(obj)

            case _:
                # Handle all other inputs via from_user_input (str, int, dict, etc.)
                self._pyproj_crs = pyproj.CRS.from_user_input(obj)

    @cached_property
    def proj(self) -> pyproj.CRS:
        """Return pyproj.CRS representation."""
        return self._pyproj_crs

    @cached_property
    def cartopy(self) -> ccrs.CRS | ccrs.Projection:
        """Return cartopy.crs.CRS or cartopy.crs.Projection representation.

        Returns
        -------
        ccrs.CRS or ccrs.Projection
            Geographic CRS returns ccrs.CRS, projected CRS returns ccrs.Projection.

        Raises
        ------
        ImportError
            If cartopy is not installed.
        RuntimeError
            If the CRS cannot be converted (e.g., WKT2 not available).
        """
        if not CARTOPY_AVAILABLE:
            raise ImportError(
                "cartopy is not installed. Install it with: pip install cartopy"
            )

        proj_crs = self.proj

        # Check if geographic or projected
        if proj_crs.is_geographic:
            return ccrs.CRS(proj_crs)
        else:
            return ccrs.Projection(proj_crs)

    @cached_property
    def osgeo(self) -> SpatialReference:
        """Return osgeo.osr.SpatialReference representation.

        Returns
        -------
        SpatialReference
            GDAL/OGR Spatial Reference object.

        Raises
        ------
        ImportError
            If osgeo is not installed.
        """
        if not OSGEO_AVAILABLE:
            raise ImportError(
                "osgeo (GDAL) is not installed. Install it with: pip install gdal"
            )

        from pyproj.enums import WktVersion

        proj_crs = self.proj
        osr_crs = SpatialReference()

        # Convert from pyproj to osgeo
        if osgeo.version_info.major < 3:
            osr_crs.ImportFromWkt(proj_crs.to_wkt(WktVersion.WKT1_GDAL))
        else:
            osr_crs.ImportFromWkt(proj_crs.to_wkt())

        return osr_crs

    def __repr__(self) -> str:  # pyright: ignore[reportImplicitOverride]
        """Return string representation."""
        return f"UCRS({self.proj.name})"

    def __str__(self) -> str:  # pyright: ignore[reportImplicitOverride]
        """Return string representation."""
        return str(self.proj)
