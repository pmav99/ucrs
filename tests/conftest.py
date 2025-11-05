"""Pytest configuration and shared fixtures for UCRS test suite."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import pyproj

if TYPE_CHECKING:
    from collections.abc import Generator

# ============================================================================
# Helper Functions for Optional Dependency Detection
# ============================================================================

def _check_cartopy_available() -> bool:
    """Check if cartopy is available."""
    try:
        import cartopy.crs  # noqa: F401
        return True
    except ImportError:
        return False


def _check_osgeo_available() -> bool:
    """Check if osgeo is available."""
    try:
        from osgeo import osr  # noqa: F401
        from osgeo.osr import SpatialReference  # noqa: F401
        # Enable exceptions to get clearer error messages
        osr.UseExceptions()
        return True
    except ImportError:
        return False


# ============================================================================
# Skip Markers for Optional Dependencies
# ============================================================================

requires_cartopy = pytest.mark.skipif(
    not _check_cartopy_available(),
    reason="cartopy not installed"
)

requires_osgeo = pytest.mark.skipif(
    not _check_osgeo_available(),
    reason="osgeo (GDAL) not installed"
)


# ============================================================================
# Common CRS Fixtures - Basic Types
# ============================================================================

@pytest.fixture
def epsg_4326() -> int:
    """EPSG code 4326 (WGS 84 geographic CRS)."""
    return 4326


@pytest.fixture
def epsg_3857() -> int:
    """EPSG code 3857 (Web Mercator projected CRS)."""
    return 3857


@pytest.fixture
def epsg_string() -> str:
    """EPSG:4326 as string."""
    return "EPSG:4326"


@pytest.fixture
def wgs84_wkt() -> str:
    """WGS 84 as WKT string."""
    return pyproj.CRS.from_epsg(4326).to_wkt()


@pytest.fixture
def proj_dict() -> dict[str, str]:
    """PROJ dictionary for WGS 84."""
    return {"proj": "longlat", "datum": "WGS84", "no_defs": "True"}


# ============================================================================
# Pyproj Fixtures
# ============================================================================

@pytest.fixture
def wgs84_pyproj() -> pyproj.CRS:
    """WGS 84 as pyproj.CRS."""
    return pyproj.CRS.from_epsg(4326)


@pytest.fixture
def web_mercator_pyproj() -> pyproj.CRS:
    """Web Mercator as pyproj.CRS."""
    return pyproj.CRS.from_epsg(3857)


# ============================================================================
# Cartopy Fixtures (only if cartopy is available)
# ============================================================================

if _check_cartopy_available():
    import cartopy.crs as ccrs

    @pytest.fixture
    def wgs84_cartopy() -> ccrs.CRS:
        """WGS 84 as cartopy.crs.CRS (geographic CRS).

        Uses Geodetic which is cartopy's representation of WGS 84.
        """
        return ccrs.Geodetic()

    @pytest.fixture
    def web_mercator_cartopy() -> ccrs.Projection:
        """Web Mercator as cartopy.crs.Projection.

        Uses the Google Maps Web Mercator projection.
        """
        return ccrs.Mercator.GOOGLE


# ============================================================================
# OSGEO Fixtures (only if osgeo is available)
# ============================================================================

if _check_osgeo_available():
    from osgeo.osr import SpatialReference

    @pytest.fixture
    def wgs84_osgeo() -> SpatialReference:
        """WGS 84 as osgeo.osr.SpatialReference."""
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
# Module Mocking Fixtures (for advanced testing scenarios)
# ============================================================================

@pytest.fixture
def mock_no_cartopy(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    """Mock cartopy as unavailable for testing error handling.

    Note: This fixture has limited usefulness since module-level imports
    happen before tests run. Use separate test environments for proper
    testing of missing dependencies.
    """
    import sys

    # Remove cartopy from sys.modules if present
    modules_to_remove = [key for key in sys.modules if key.startswith('cartopy')]
    for module in modules_to_remove:
        monkeypatch.delitem(sys.modules, module, raising=False)

    # Mock the import to raise ImportError
    monkeypatch.setitem(sys.modules, 'cartopy', None)
    monkeypatch.setitem(sys.modules, 'cartopy.crs', None)

    yield


@pytest.fixture
def mock_no_osgeo(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    """Mock osgeo as unavailable for testing error handling.

    Note: This fixture has limited usefulness since module-level imports
    happen before tests run. Use separate test environments for proper
    testing of missing dependencies.
    """
    import sys

    # Remove osgeo from sys.modules if present
    modules_to_remove = [key for key in sys.modules if key.startswith('osgeo')]
    for module in modules_to_remove:
        monkeypatch.delitem(sys.modules, module, raising=False)

    # Mock the import to raise ImportError
    monkeypatch.setitem(sys.modules, 'osgeo', None)
    monkeypatch.setitem(sys.modules, 'osgeo.osr', None)

    yield


# ============================================================================
# Pytest Configuration Hooks
# ============================================================================

def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers and configuration."""
    config.addinivalue_line(
        "markers", "requires_cartopy: mark test as requiring cartopy"
    )
    config.addinivalue_line(
        "markers", "requires_osgeo: mark test as requiring osgeo/GDAL"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_report_header(config: pytest.Config) -> list[str]:
    """Add optional dependency status to pytest header."""
    return [
        f"cartopy available: {_check_cartopy_available()}",
        f"osgeo available: {_check_osgeo_available()}",
    ]
