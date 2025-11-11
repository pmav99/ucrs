
"""Unified CRS - Seamless conversion between geospatial CRS representations.

UCRS provides a unified interface for working with Coordinate Reference Systems (CRS)
across the major Python geospatial libraries: pyproj, cartopy, and osgeo/GDAL.

The UCRS class accepts any CRS input format and provides lazy, cached conversion to
different library-specific representations through simple properties.

Key Features
------------
- Single class to handle all CRS types
- Accepts any input: EPSG codes, WKT, PROJ strings, or library-specific objects
- Lazy conversion with caching for performance
- Automatic handling of optional dependencies
- Full type annotation support

Basic Usage
-----------
>>> from ucrs import UCRS
>>> # Create from EPSG code
>>> crs = UCRS(4326)
>>>
>>> # Create from WKT file
>>> crs = UCRS.from_file("path/to/crs.wkt")
>>>
>>> # Access different representations
>>> proj_crs = crs  # UCRS inherits from pyproj.CRS
>>> cart_crs = crs.cartopy  # cartopy.crs.CRS (if cartopy installed)
>>> osgeo_sr = crs.osgeo    # osgeo.osr.SpatialReference (if GDAL installed)

Supported Input Types
---------------------
- EPSG codes (int): 4326, 3857, etc.
- EPSG strings: "EPSG:4326", "epsg:3857"
- WKT strings (WKT1 or WKT2)
- PROJ strings: "+proj=longlat +datum=WGS84 +no_defs"
- pyproj.CRS objects
- cartopy.crs.CRS or cartopy.crs.Projection objects (if cartopy available)
- osgeo.osr.SpatialReference objects (if osgeo available)
- Dictionary representations

Dependencies
------------
Required:
    - pyproj

Optional:
    - cartopy (for .cartopy property)
    - osgeo/GDAL (for .osgeo property)

The library gracefully handles missing optional dependencies, raising informative
ImportError messages when attempting to use unavailable conversions.
"""

from __future__ import annotations

from functools import cached_property
from os import PathLike
from pathlib import Path
from typing import cast
from typing import TypeAlias
from typing import TYPE_CHECKING

import pyproj.crs

try:
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version("ucrs")
    except PackageNotFoundError:
        __version__ = "unknown"
except ImportError:
    __version__ = "unknown"

__all__ = ["UCRS", "__version__"]

# Type aliases
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

    # Type aliases for return types
    CartopyCRS: TypeAlias = ccrs.CRS
    CartopyProjection: TypeAlias = ccrs.Projection
    OSGeoSpatialReference: TypeAlias = SpatialReference


class UCRS(pyproj.crs.CustomConstructorCRS):
    """Unified CRS for seamless conversion between pyproj, cartopy, and osgeo.

    UCRS is a wrapper class that inherits from pyproj.CRS, allowing it to be used
    directly as a pyproj CRS object while providing convenient access to cartopy
    and osgeo representations through cached properties.

    The class stores all CRS data internally as pyproj.CRS (the canonical representation)
    and lazily converts to other formats only when requested. All conversions are cached
    for optimal performance.

    Parameters
    ----------
    obj : CRSInput
        Any valid CRS input. This can be:
        - EPSG code as int (e.g., 4326)
        - EPSG string (e.g., "EPSG:4326")
        - WKT string (WKT1 or WKT2)
        - PROJ string (e.g., "+proj=longlat +datum=WGS84")
        - pyproj.CRS object
        - cartopy.crs.CRS or cartopy.crs.Projection object
        - osgeo.osr.SpatialReference object
        - Dictionary representation

    Attributes
    ----------
    cartopy : cartopy.crs.CRS or cartopy.crs.Projection
        Lazy conversion to cartopy representation. Returns CRS for geographic
        coordinate systems and Projection for projected coordinate systems.
        Requires cartopy to be installed.
    osgeo : osgeo.osr.SpatialReference
        Lazy conversion to GDAL/OGR SpatialReference representation.
        Requires osgeo/GDAL to be installed.

    Notes
    -----
    - Since UCRS inherits from pyproj.CRS, it can be used directly wherever
      a pyproj.CRS is expected
    - Conversions are cached using @cached_property, so repeated access is fast
    - The class handles version differences in GDAL (2.x vs 3.x) automatically
    - If optional dependencies are missing, accessing their properties raises
      informative ImportError messages

    Examples
    --------
    Create from various input types:

    >>> # From EPSG code
    >>> crs = UCRS(4326)
    >>> crs.name
    'WGS 84'

    >>> # From WKT string
    >>> wkt = 'GEOGCS["WGS 84",DATUM["WGS_1984",...]]'
    >>> crs = UCRS(wkt)

    >>> # From pyproj.CRS
    >>> import pyproj
    >>> proj_crs = pyproj.CRS.from_epsg(3857)
    >>> crs = UCRS(proj_crs)

    Access different representations:

    >>> crs = UCRS(4326)
    >>> # Use as pyproj.CRS directly
    >>> crs.is_geographic
    True

    >>> # Convert to cartopy (if installed)
    >>> cart = crs.cartopy
    >>> type(cart).__name__
    'PlateCarree'

    >>> # Convert to osgeo (if installed)
    >>> sr = crs.osgeo
    >>> sr.GetAuthorityCode(None)
    '4326'
    """

    def __init__(self, obj: CRSInput) -> None:
        """Initialize UCRS from various CRS representations.

        This constructor accepts a wide variety of CRS input formats and converts
        them internally to pyproj.CRS, which serves as the canonical representation.
        The conversion process handles different input types through runtime type
        checking with graceful fallback.

        Parameters
        ----------
        obj : CRSInput
            The input CRS in any supported format:
            - **int**: EPSG code (e.g., 4326 for WGS 84)
            - **str**: EPSG string ("EPSG:4326"), WKT string, or PROJ string
            - **pyproj.CRS**: Passed through directly
            - **cartopy.crs.CRS or Projection**: Converted via pyproj (if cartopy available)
            - **osgeo.osr.SpatialReference**: Converted via WKT (if osgeo available)
            - **dict**: Dictionary representation of CRS

        Notes
        -----
        - GDAL version is automatically detected for proper WKT version handling
        - All subsequent conversions are performed lazily and cached

        Examples
        --------
        >>> # From EPSG code
        >>> crs = UCRS(4326)

        >>> # From EPSG string
        >>> crs = UCRS("EPSG:3857")

        >>> # From PROJ string
        >>> crs = UCRS("+proj=longlat +datum=WGS84 +no_defs")

        >>> # From existing library objects
        >>> import pyproj
        >>> crs = UCRS(pyproj.CRS.from_epsg(4326))
        """
        # Convert input to pyproj.CRS
        # Check types in order of expected usage frequency

        # Try to handle cartopy CRS/Projection (most common library-specific usage)
        try:
            import cartopy.crs as ccrs
            if isinstance(obj, ccrs.Projection):
                # cartopy Projection
                # cartopy.crs.CRS/Projection inherit from pyproj.CRS
                self._pyproj_crs = pyproj.CRS.from_user_input(obj)
            elif isinstance(obj, ccrs.CRS):
                # cartopy CRS - cartopy.crs.CRS inherits from pyproj.CRS
                self._pyproj_crs = pyproj.CRS.from_user_input(obj)
            else:
                raise TypeError("Not a cartopy CRS")
        except (ImportError, TypeError):
            # Check if already a pyproj.CRS object
            if isinstance(obj, pyproj.CRS):
                self._pyproj_crs = obj
            else:
                # Try to handle osgeo SpatialReference
                try:
                    import osgeo
                    from osgeo.osr import SpatialReference
                    if isinstance(obj, SpatialReference):
                        # Convert from osgeo to pyproj using WKT
                        # Use WKT2_2018 for GDAL 3+, WKT1 for older versions
                        wkt: str
                        if osgeo.version_info.major < 3:
                            wkt = cast(str, obj.ExportToWkt())
                        else:
                            wkt = cast(str, obj.ExportToWkt(["FORMAT=WKT2_2018"]))
                        self._pyproj_crs = pyproj.CRS.from_wkt(wkt)
                    else:
                        raise TypeError("Not a SpatialReference")
                except (ImportError, TypeError):
                    # Handle all other inputs via from_user_input
                    # (str, int, dict, WKT, PROJ string, etc.)
                    self._pyproj_crs = pyproj.CRS.from_user_input(obj)

        # Initialize parent CustomConstructorCRS with the pyproj CRS
        super().__init__(self._pyproj_crs.to_json_dict())

    @classmethod
    def from_file(cls, filepath: str | PathLike[str]) -> UCRS:
        """Create UCRS instance from a file containing WKT CRS definition.

        Reads a text file containing a WKT (Well-Known Text) representation of a
        Coordinate Reference System and creates a UCRS instance from it.

        Parameters
        ----------
        filepath : str or os.PathLike[str]
            Path to the file containing WKT CRS definition. Can be a string path,
            a pathlib.Path object, or any path-like object.

        Returns
        -------
        UCRS
            A new UCRS instance created from the WKT in the file.

        Notes
        -----
        - The file is expected to contain plain text WKT in UTF-8 encoding
        - Whitespace (leading/trailing) is automatically stripped
        - The WKT validation is performed by pyproj.CRS

        Examples
        --------
        >>> # Create from file path string
        >>> crs = UCRS.from_file("/path/to/crs.wkt")

        >>> # Create from pathlib.Path
        >>> from pathlib import Path
        >>> crs = UCRS.from_file(Path("crs.wkt"))

        >>> # The resulting UCRS can be used like any other
        >>> crs.is_geographic
        True
        """
        path = Path(filepath)
        wkt_content = path.read_text(encoding="utf-8").strip()
        return cls(wkt_content)

    @cached_property
    def cartopy(self) -> CartopyCRS | CartopyProjection:
        """Convert to cartopy CRS representation (lazy, cached).

        Returns
        -------
        cartopy.crs.CRS or cartopy.crs.Projection
            Returns Projection for projected CRS, CRS for geographic CRS.

        Notes
        -----
        Cartopy requires CRS created with WKT2, PROJ JSON, or a spatial
        reference ID (i.e. EPSG) with the area of use defined. Otherwise,
        x_limits and y_limits will not work properly.

        Examples
        --------
        >>> ucrs = UCRS(4326)
        >>> cart = ucrs.cartopy
        >>> isinstance(cart, cartopy.crs.CRS)
        True

        >>> ucrs = UCRS(6933)  # Projected CRS
        >>> cart = ucrs.cartopy
        >>> isinstance(cart, cartopy.crs.Projection)
        True
        """
        try:
            import cartopy.crs as ccrs
        except ImportError as e:
            raise ImportError(
                "cartopy is not installed. Install it with: pip install cartopy"
            ) from e

        try:
            # Check if this CRS is projected or geographic
            # Use Projection for projected CRS, CRS for geographic
            if self.is_projected:
                return ccrs.Projection(self)
            else:
                return ccrs.CRS(self)
        except Exception as e:
            raise RuntimeError(
                f"Failed to convert to cartopy CRS. This may occur if the CRS "
                f"was not created with WKT2, PROJ JSON, or an EPSG code with "
                f"area of use defined. Original error: {e}"
            ) from e

    @cached_property
    def osgeo(self) -> OSGeoSpatialReference:
        """Convert to osgeo SpatialReference representation (lazy, cached).

        Returns
        -------
        osgeo.osr.SpatialReference
            The osgeo SpatialReference object.

        Notes
        -----
        Uses WKT2_2018 for GDAL 3+ and WKT1_GDAL for older versions
        to ensure maximum compatibility.

        Examples
        --------
        >>> ucrs = UCRS(4326)
        >>> osgeo_sr = ucrs.osgeo
        >>> osgeo_sr.GetAuthorityCode(None)
        '4326'
        """
        try:
            import osgeo
            from osgeo.osr import SpatialReference
        except ImportError as e:
            raise ImportError(
                "osgeo (GDAL) is not installed. Install it with: pip install gdal"
            ) from e

        from pyproj.enums import WktVersion

        osr_crs = SpatialReference()

        # Use appropriate WKT version based on GDAL version
        if osgeo.version_info.major < 3:
            # GDAL 2.x - use WKT1_GDAL
            wkt = self.to_wkt(WktVersion.WKT1_GDAL)
        else:
            # GDAL 3+ - use WKT2
            wkt = self.to_wkt()

        osr_crs.ImportFromWkt(wkt)
        return osr_crs

    def summary(self) -> str:
        attributes = [
           'is_bound',
           'is_compound',
           'is_deprecated',
           'is_derived',
           'is_engineering',
           'is_geocentric',
           'is_geographic',
           'is_projected',
           'is_vertical',
        ]
        data = {attr: getattr(self, attr) for attr in attributes}
        print(data)
