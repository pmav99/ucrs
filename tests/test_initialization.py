"""Tests for UCRS initialization with various input types."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import pyproj

from ucrs import UCRS
from tests.conftest import requires_cartopy, requires_osgeo, CARTOPY_AVAILABLE, OSGEO_AVAILABLE

if TYPE_CHECKING:
    if CARTOPY_AVAILABLE:
        import cartopy.crs as ccrs
    if OSGEO_AVAILABLE:
        from osgeo.osr import SpatialReference


class TestInitializationFromInt:
    """Test UCRS initialization from EPSG integer codes."""

    def test_from_epsg_int_geographic(self, epsg_4326: int) -> None:
        """Test initialization from EPSG integer (geographic CRS)."""
        ucrs = UCRS(epsg_4326)
        assert isinstance(ucrs.proj, pyproj.CRS)
        assert ucrs.proj.to_epsg() == 4326
        assert ucrs.proj.is_geographic

    def test_from_epsg_int_projected(self, epsg_3857: int) -> None:
        """Test initialization from EPSG integer (projected CRS)."""
        ucrs = UCRS(epsg_3857)
        assert isinstance(ucrs.proj, pyproj.CRS)
        assert ucrs.proj.to_epsg() == 3857
        assert ucrs.proj.is_projected

    @pytest.mark.parametrize("epsg_code", [
        4326,  # WGS84
        3857,  # Web Mercator
        32633,  # UTM Zone 33N
        2154,  # Lambert-93 (France)
    ])
    def test_from_various_epsg_codes(self, epsg_code: int) -> None:
        """Test initialization from various EPSG codes."""
        ucrs = UCRS(epsg_code)
        assert ucrs.proj.to_epsg() == epsg_code


class TestInitializationFromString:
    """Test UCRS initialization from string representations."""

    def test_from_epsg_string(self, epsg_string: str) -> None:
        """Test initialization from EPSG string format."""
        ucrs = UCRS(epsg_string)
        assert ucrs.proj.to_epsg() == 4326

    def test_from_wkt_string(self, wgs84_wkt: str) -> None:
        """Test initialization from WKT string."""
        ucrs = UCRS(wgs84_wkt)
        assert ucrs.proj.to_epsg() == 4326

    @pytest.mark.parametrize("epsg_format", [
        "EPSG:4326",
        "epsg:4326",
        "EPSG:3857",
    ])
    def test_from_epsg_string_formats(self, epsg_format: str) -> None:
        """Test various EPSG string format variations."""
        ucrs = UCRS(epsg_format)
        expected_code = int(epsg_format.split(":")[1])
        assert ucrs.proj.to_epsg() == expected_code


class TestInitializationFromPyproj:
    """Test UCRS initialization from pyproj.CRS objects."""

    def test_from_pyproj_crs(self, wgs84_pyproj: pyproj.CRS) -> None:
        """Test initialization from pyproj.CRS object."""
        ucrs = UCRS(wgs84_pyproj)
        assert ucrs.proj is wgs84_pyproj  # Should be same object
        assert ucrs.proj.to_epsg() == 4326

    def test_from_pyproj_crs_projected(self, web_mercator_pyproj: pyproj.CRS) -> None:
        """Test initialization from pyproj.CRS (projected)."""
        ucrs = UCRS(web_mercator_pyproj)
        assert ucrs.proj is web_mercator_pyproj
        assert ucrs.proj.to_epsg() == 3857


class TestInitializationFromDict:
    """Test UCRS initialization from PROJ dictionary."""

    def test_from_proj_dict(self, proj_dict: dict[str, str]) -> None:
        """Test initialization from PROJ dictionary."""
        ucrs = UCRS(proj_dict)
        assert isinstance(ucrs.proj, pyproj.CRS)
        assert ucrs.proj.is_geographic
        # Note: PROJ dict might not have exact EPSG match


@requires_cartopy
class TestInitializationFromCartopy:
    """Test UCRS initialization from cartopy CRS objects."""

    def test_from_cartopy_crs(self, wgs84_cartopy: ccrs.CRS) -> None:
        """Test initialization from cartopy.crs.CRS."""
        ucrs = UCRS(wgs84_cartopy)
        assert isinstance(ucrs.proj, pyproj.CRS)
        assert ucrs.proj.is_geographic

    def test_from_cartopy_projection(self, web_mercator_cartopy: ccrs.Projection) -> None:
        """Test initialization from cartopy.crs.Projection."""
        ucrs = UCRS(web_mercator_cartopy)
        assert isinstance(ucrs.proj, pyproj.CRS)
        assert ucrs.proj.is_projected

    @pytest.mark.parametrize("cartopy_crs_class", [
        pytest.param(lambda: __import__("cartopy.crs", fromlist=["PlateCarree"]).PlateCarree(), id="PlateCarree"),
        pytest.param(lambda: __import__("cartopy.crs", fromlist=["Mercator"]).Mercator(), id="Mercator"),
        pytest.param(lambda: __import__("cartopy.crs", fromlist=["Robinson"]).Robinson(), id="Robinson"),
    ])
    def test_from_various_cartopy_projections(self, cartopy_crs_class) -> None:
        """Test initialization from various cartopy projections."""
        crs = cartopy_crs_class()
        ucrs = UCRS(crs)
        assert isinstance(ucrs.proj, pyproj.CRS)


@requires_osgeo
class TestInitializationFromOsgeo:
    """Test UCRS initialization from osgeo SpatialReference objects."""

    def test_from_osgeo_spatial_reference(self, wgs84_osgeo: SpatialReference) -> None:
        """Test initialization from osgeo.osr.SpatialReference."""
        ucrs = UCRS(wgs84_osgeo)
        assert isinstance(ucrs.proj, pyproj.CRS)
        assert ucrs.proj.to_epsg() == 4326

    def test_from_osgeo_projected(self, web_mercator_osgeo: SpatialReference) -> None:
        """Test initialization from osgeo.osr.SpatialReference (projected)."""
        ucrs = UCRS(web_mercator_osgeo)
        assert isinstance(ucrs.proj, pyproj.CRS)
        assert ucrs.proj.to_epsg() == 3857


class TestOriginalPreservation:
    """Test that original input is preserved."""

    def test_original_preserved_int(self, epsg_4326: int) -> None:
        """Test that original input is stored."""
        ucrs = UCRS(epsg_4326)
        assert ucrs._original == epsg_4326

    def test_original_preserved_string(self, epsg_string: str) -> None:
        """Test that original string input is stored."""
        ucrs = UCRS(epsg_string)
        assert ucrs._original == epsg_string

    def test_original_preserved_pyproj(self, wgs84_pyproj: pyproj.CRS) -> None:
        """Test that original pyproj.CRS is stored."""
        ucrs = UCRS(wgs84_pyproj)
        assert ucrs._original is wgs84_pyproj
