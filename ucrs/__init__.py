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
>>> # Create from WKT file path
>>> crs = UCRS("path/to/crs.wkt")
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

import errno

from functools import cached_property, lru_cache
from collections.abc import Sequence
from pathlib import Path
from typing import cast
from typing import Literal
from typing import overload
from typing import TypeAlias
from typing import TYPE_CHECKING
from typing import final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyproj.crs.crs import CustomConstructorCRS

try:
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version("ucrs")
    except PackageNotFoundError:
        __version__ = "unknown"
except ImportError:
    __version__ = "unknown"

__all__ = ["CRSInput", "UCRS", "__version__", "transform", "transform_coords"]

# Type aliases
if TYPE_CHECKING:
    import pyproj
    import cartopy.crs as ccrs
    from osgeo.osr import SpatialReference  # pyright: ignore[reportMissingImports,reportUnknownVariableType]
    from shapely.geometry.base import BaseGeometry

    CRSInput: TypeAlias = (
        pyproj.CRS
        | ccrs.CRS
        | ccrs.Projection
        #| SpatialReference
        | Path
        | str
        | int
        | dict[str, str]
    )

    # Type aliases for return types
    CartopyCRS: TypeAlias = ccrs.CRS
    CartopyProjection: TypeAlias = ccrs.Projection
else:
    # Runtime version - no optional dependency imports
    import pyproj

    CRSInput: TypeAlias = pyproj.CRS | Path | str | int | dict[str, str]


@final
class UCRS(CustomConstructorCRS):
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
        if isinstance(obj, str):
            try:
                with open(obj, "r", encoding="utf-8") as fd:
                    obj = fd.read().strip()
            except OSError as e:
                if e.errno not in (errno.ENOENT, errno.ENAMETOOLONG):
                    raise
        elif isinstance(obj, Path):
            obj = cast(str, obj.read_text(encoding="utf-8")).strip()  # pyright: ignore[reportUnknownMemberType]
        else:
            pass

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
                    import osgeo  # pyright: ignore[reportMissingImports]
                    from osgeo.osr import SpatialReference  # pyright: ignore[reportMissingImports,reportUnknownVariableType]
                    if isinstance(obj, SpatialReference):
                        # Convert from osgeo to pyproj using WKT
                        # Use WKT2_2018 for GDAL 3+, WKT1 for older versions
                        wkt: str
                        if osgeo.version_info.major < 3:  # pyright: ignore[reportUnknownMemberType]
                            wkt = cast(str, obj.ExportToWkt())  # pyright: ignore[reportUnknownMemberType,reportAttributeAccessIssue]
                        else:
                            wkt = cast(str, obj.ExportToWkt(["FORMAT=WKT2_2018"]))  # pyright: ignore[reportUnknownMemberType,reportAttributeAccessIssue]
                        self._pyproj_crs = pyproj.CRS.from_wkt(wkt)
                    else:
                        raise TypeError("Not a SpatialReference")
                except (ImportError, TypeError):
                    # Handle all other inputs via from_user_input
                    # (str, int, dict, WKT, PROJ string, etc.)
                    self._pyproj_crs = pyproj.CRS.from_user_input(obj)

        # Initialize parent CustomConstructorCRS with the pyproj CRS
        super().__init__(self._pyproj_crs.to_json_dict())  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]

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
            raise RuntimeError(f"Failed to convert to cartopy CRS. Original error: {e}") from e

    @cached_property
    def osgeo(self) -> SpatialReference:  # pyright: ignore[reportUnknownParameterType]
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
            import osgeo  # pyright: ignore[reportMissingImports]
            from osgeo.osr import SpatialReference  # pyright: ignore[reportMissingImports,reportUnknownVariableType]
        except ImportError as e:
            raise ImportError(
                "osgeo (GDAL) is not installed. Install it with: pip install gdal"
            ) from e

        from pyproj.enums import WktVersion

        osr_crs = SpatialReference()  # pyright: ignore[reportUnknownVariableType]

        # Use appropriate WKT version based on GDAL version
        if osgeo.version_info.major < 3:  # pyright: ignore[reportUnknownMemberType]
            # GDAL 2.x - use WKT1_GDAL
            wkt = self.to_wkt(WktVersion.WKT1_GDAL)
        else:
            # GDAL 3+ - use WKT2
            wkt = self.to_wkt()

        osr_crs.ImportFromWkt(wkt)  # pyright: ignore[reportUnknownMemberType]
        return osr_crs  # pyright: ignore[reportUnknownVariableType]

    def summary(self) -> dict[str, str]:
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
        return data


# ============================================================================
# Shared transformer cache
# ============================================================================

@lru_cache(maxsize=256)
def _get_transformer(
    source_crs: CRSInput,
    target_crs: CRSInput,
    always_xy: bool,
) -> pyproj.Transformer:
    src = source_crs if isinstance(source_crs, UCRS) else UCRS(source_crs)
    tgt = target_crs if isinstance(target_crs, UCRS) else UCRS(target_crs)
    return pyproj.Transformer.from_crs(src, tgt, always_xy=always_xy)


# ============================================================================
# Coordinate transformation
# ============================================================================

# Scalar point type: (x, y) or (x, y, z) where each element is a number.
ScalarPoint2D: TypeAlias = tuple[float, float]
ScalarPoint3D: TypeAlias = tuple[float, float, float]
ScalarPoint: TypeAlias = ScalarPoint2D | ScalarPoint3D

@overload
def transform_coords(
    source_crs: CRSInput,
    target_crs: CRSInput,
    coords: ArrayLike | tuple[ArrayLike, ...],
    *,
    always_xy: bool = ...,
    output: Literal["array"],
) -> NDArray[np.float64]: ...

@overload
def transform_coords(
    source_crs: CRSInput,
    target_crs: CRSInput,
    coords: ArrayLike | tuple[ArrayLike, ...],
    *,
    always_xy: bool = ...,
    output: Literal["tuple"],
) -> tuple[NDArray[np.float64], ...] | ScalarPoint: ...

@overload
def transform_coords(
    source_crs: CRSInput,
    target_crs: CRSInput,
    coords: ArrayLike | tuple[ArrayLike, ...],
    *,
    always_xy: bool = ...,
    output: Literal["auto"] = ...,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], ...] | ScalarPoint: ...

def transform_coords(
    source_crs: CRSInput,
    target_crs: CRSInput,
    coords: ArrayLike | tuple[ArrayLike, ...],
    *,
    always_xy: bool = True,
    output: Literal["auto", "array", "tuple"] = "auto",
) -> NDArray[np.float64] | tuple[NDArray[np.float64], ...] | ScalarPoint:
    """Transform coordinates between CRS.

    Parameters
    ----------
    source_crs : CRSInput
        Source coordinate reference system (any input accepted by UCRS).
    target_crs : CRSInput
        Target coordinate reference system (any input accepted by UCRS).
    coords : ArrayLike | tuple[ArrayLike, ...] | tuple[float, float] | tuple[float, float, float]
        One of:
        - A single scalar point ``(x, y)`` or ``(x, y, z)`` (tuple of numbers).
        - A single (N, 2) or (N, 3) array-like.
        - A tuple of 2–3 1-D array-likes ``(x, y)`` / ``(x, y, z)``.
    always_xy : bool, optional
        If True (default), coordinate order is x/y (lon/lat) regardless
        of CRS axis order.
    output : ``"auto"`` | ``"array"`` | ``"tuple"``, optional
        Controls the return type:
        - ``"auto"`` (default): match the input format.
        - ``"array"``: always return a single (N, 2|3) ndarray.
        - ``"tuple"``: always return a tuple of 1-D ndarrays.

    Returns
    -------
    np.ndarray | tuple[np.ndarray, ...] | tuple[float, ...]
        Transformed coordinates in the requested format.
        Scalar input returns a tuple of plain floats by default.
    """
    transformer = _get_transformer(source_crs, target_crs, always_xy)

    # Determine whether the caller passed a scalar point, a tuple-of-arrays,
    # or a single array.
    is_scalar_input = False
    is_tuple_input = False

    if isinstance(coords, tuple) and len(coords) in (2, 3) and all(isinstance(c, (int, float)) for c in coords):
        # Scalar point: (x, y) or (x, y, z) where each element is a number.
        is_scalar_input = True
        x = np.array([coords[0]], dtype=np.float64)
        y = np.array([coords[1]], dtype=np.float64)
        z = np.array([coords[2]], dtype=np.float64) if len(coords) == 3 else None
    elif isinstance(coords, tuple):
        is_tuple_input = True
        arrays = [np.asarray(a, dtype=np.float64) for a in coords]
        if len(arrays) == 2:
            x, y = arrays
            z = None
        elif len(arrays) == 3:
            x, y, z = arrays
        else:
            raise ValueError(
                f"Tuple input must have 2 or 3 elements, got {len(arrays)}"
            )
        sizes = {a.shape[0] for a in arrays}
        if len(sizes) != 1:
            raise ValueError(
                f"All arrays in tuple must have the same length, got {[a.shape[0] for a in arrays]}"
            )
    else:
        arr = np.asarray(coords, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] not in (2, 3):
            raise ValueError(
                f"Array input must have shape (N, 2) or (N, 3), got {arr.shape}"
            )
        x = arr[:, 0]
        y = arr[:, 1]
        z = arr[:, 2] if arr.shape[1] == 3 else None

    # Perform the transformation.
    if z is not None:
        tx, ty, tz = transformer.transform(x, y, z)
    else:
        tx, ty = transformer.transform(x, y)
        tz = None

    tx = np.asarray(tx, dtype=np.float64)
    ty = np.asarray(ty, dtype=np.float64)
    if tz is not None:
        tz = np.asarray(tz, dtype=np.float64)

    # Build the result in the requested format.
    if is_scalar_input and output in ("auto", "tuple"):
        if tz is None:
            return (float(tx[0]), float(ty[0]))
        return (float(tx[0]), float(ty[0]), float(tz[0]))

    if output == "auto":
        output = "tuple" if is_tuple_input else "array"

    if output == "tuple":
        if tz is None:
            return (tx, ty)
        return (tx, ty, tz)
    else:  # "array"
        if tz is None:
            return np.column_stack([tx, ty])
        return np.column_stack([tx, ty, tz])


# ============================================================================
# Geometry transformation
# ============================================================================

@overload
def transform(
    source_crs: CRSInput,
    target_crs: CRSInput,
    geom: BaseGeometry,
    *,
    always_xy: bool = ...,
) -> BaseGeometry: ...

@overload
def transform(
    source_crs: CRSInput,
    target_crs: CRSInput,
    geom: Sequence[BaseGeometry],
    *,
    always_xy: bool = ...,
) -> list[BaseGeometry]: ...

def transform(
    source_crs: CRSInput,
    target_crs: CRSInput,
    geom: BaseGeometry | Sequence[BaseGeometry],
    *,
    always_xy: bool = True,
) -> BaseGeometry | list[BaseGeometry]:
    """Transform shapely geometries between CRS.

    Parameters
    ----------
    source_crs : CRSInput
        Source coordinate reference system (any input accepted by UCRS).
    target_crs : CRSInput
        Target coordinate reference system (any input accepted by UCRS).
    geom : BaseGeometry | Sequence[BaseGeometry]
        A single shapely geometry or a sequence of geometries.
    always_xy : bool, optional
        If True (default), coordinate order is x/y (lon/lat) regardless
        of CRS axis order.

    Returns
    -------
    BaseGeometry | list[BaseGeometry]
        Transformed geometry/geometries. Single input → single out,
        sequence input → list out.
    """
    try:
        from shapely import transform as shp_transform
        from shapely.geometry.base import BaseGeometry
    except ImportError as e:
        raise ImportError(
            "shapely is not installed. Install it with: pip install shapely"
        ) from e

    is_single = isinstance(geom, BaseGeometry)
    geoms: list[BaseGeometry] = [geom] if is_single else list(geom)

    results: list[BaseGeometry] = [
        shp_transform(g, lambda coords: transform_coords(source_crs, target_crs, coords, always_xy=always_xy, output="array"))
        for g in geoms
    ]

    if is_single:
        return results[0]
    return results
