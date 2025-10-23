"""Pytest configuration and shared fixtures for UCRS test suite."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest
import pyproj

if TYPE_CHECKING:
    from collections.abc import Generator

# Check optional dependencies
try:
    import cartopy.crs as ccrs
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False

try:
    from osgeo import osr
    from osgeo.osr import SpatialReference
    # Enable exceptions to avoid FutureWarning
    osr.UseExceptions()
    OSGEO_AVAILABLE = True
except ImportError:
    OSGEO_AVAILABLE = False


# ============================================================================
# Skip Markers for Optional Dependencies
# ============================================================================

requires_cartopy = pytest.mark.skipif(
    not CARTOPY_AVAILABLE,
    reason="cartopy not installed"
)

requires_osgeo = pytest.mark.skipif(
    not OSGEO_AVAILABLE,
    reason="osgeo (GDAL) not installed"
)


# ============================================================================
# Common CRS Fixtures
# ============================================================================

@pytest.fixture
def epsg_4326() -> int:
    """EPSG code for WGS84 (geographic)."""
    return 4326


@pytest.fixture
def epsg_3857() -> int:
    """EPSG code for Web Mercator (projected)."""
    return 3857


@pytest.fixture
def wgs84_pyproj() -> pyproj.CRS:
    """WGS84 as pyproj.CRS."""
    return pyproj.CRS.from_epsg(4326)


@pytest.fixture
def web_mercator_pyproj() -> pyproj.CRS:
    """Web Mercator as pyproj.CRS."""
    return pyproj.CRS.from_epsg(3857)


@pytest.fixture
def wgs84_wkt() -> str:
    """WGS84 as WKT string."""
    return pyproj.CRS.from_epsg(4326).to_wkt()


@pytest.fixture
def epsg_string() -> str:
    """EPSG:4326 as string."""
    return "EPSG:4326"


@pytest.fixture
def proj_dict() -> dict[str, str]:
    """PROJ dictionary for WGS84."""
    return {"proj": "longlat", "datum": "WGS84", "no_defs": "True"}


# ============================================================================
# Cartopy Fixtures
# ============================================================================

if CARTOPY_AVAILABLE:
    @pytest.fixture
    def wgs84_cartopy() -> ccrs.CRS:
        """WGS84 as cartopy.crs.CRS (geographic)."""
        # Use Geodetic for a true geographic CRS
        return ccrs.Geodetic()

    @pytest.fixture
    def web_mercator_cartopy() -> ccrs.Projection:
        """Web Mercator as cartopy.crs.Projection."""
        return ccrs.Mercator.GOOGLE


# ============================================================================
# OSGEO Fixtures
# ============================================================================

if OSGEO_AVAILABLE:
    @pytest.fixture
    def wgs84_osgeo() -> SpatialReference:
        """WGS84 as osgeo.osr.SpatialReference."""
        srs = SpatialReference()
        srs.ImportFromEPSG(4326)
        return srs

    @pytest.fixture
    def web_mercator_osgeo() -> SpatialReference:
        """Web Mercator as osgeo.osr.SpatialReference."""
        srs = SpatialReference()
        srs.ImportFromEPSG(3857)
        return srs


# ============================================================================
# Module Mocking Fixtures
# ============================================================================

@pytest.fixture
def mock_no_cartopy(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    """Mock cartopy as unavailable."""
    # Remove cartopy from sys.modules if present
    modules_to_remove = [key for key in sys.modules if key.startswith('cartopy')]
    for module in modules_to_remove:
        monkeypatch.delitem(sys.modules, module, raising=False)

    # Mock the import to raise ImportError
    monkeypatch.setitem(sys.modules, 'cartopy', None)
    monkeypatch.setitem(sys.modules, 'cartopy.crs', None)

    yield

    # Cleanup is handled by monkeypatch


@pytest.fixture
def mock_no_osgeo(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    """Mock osgeo as unavailable."""
    # Remove osgeo from sys.modules if present
    modules_to_remove = [key for key in sys.modules if key.startswith('osgeo')]
    for module in modules_to_remove:
        monkeypatch.delitem(sys.modules, module, raising=False)

    # Mock the import to raise ImportError
    monkeypatch.setitem(sys.modules, 'osgeo', None)
    monkeypatch.setitem(sys.modules, 'osgeo.osr', None)

    yield

    # Cleanup is handled by monkeypatch


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_cartopy: mark test as requiring cartopy"
    )
    config.addinivalue_line(
        "markers", "requires_osgeo: mark test as requiring osgeo/GDAL"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
