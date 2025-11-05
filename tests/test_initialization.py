"""Tests for UCRS initialization with various input types."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import pyproj

from ucrs import UCRS
from tests.conftest import requires_cartopy, requires_osgeo

if TYPE_CHECKING:
    import cartopy.crs as ccrs
    from osgeo.osr import SpatialReference


class TestInitializationFromInt:
    """Test UCRS initialization from EPSG integer codes."""

    def test_from_epsg_int_geographic(self, epsg_4326: int) -> None:
        """Test initialization from EPSG integer (geographic CRS)."""
        ucrs = UCRS(epsg_4326)
        assert isinstance(ucrs, pyproj.CRS)
        assert ucrs.to_epsg() == 4326
        assert ucrs.is_geographic

    def test_from_epsg_int_projected(self, epsg_3857: int) -> None:
        """Test initialization from EPSG integer (projected CRS)."""
        ucrs = UCRS(epsg_3857)
        assert isinstance(ucrs, pyproj.CRS)
        assert ucrs.to_epsg() == 3857
        assert ucrs.is_projected

    @pytest.mark.parametrize("epsg_code,expected_type", [
        (4326, "geographic"),  # WGS84
        (3857, "projected"),   # Web Mercator
        (32633, "projected"),  # UTM Zone 33N
        (2154, "projected"),   # Lambert-93 (France)
    ])
    def test_from_various_epsg_codes(self, epsg_code: int, expected_type: str) -> None:
        """Test initialization from various EPSG codes."""
        ucrs = UCRS(epsg_code)
        assert ucrs.to_epsg() == epsg_code

        if expected_type == "geographic":
            assert ucrs.is_geographic
        else:
            assert ucrs.is_projected


class TestInitializationFromString:
    """Test UCRS initialization from string representations."""

    def test_from_epsg_string(self, epsg_string: str) -> None:
        """Test initialization from EPSG string format."""
        ucrs = UCRS(epsg_string)
        assert ucrs.to_epsg() == 4326

    def test_from_wkt_string(self, wgs84_wkt: str) -> None:
        """Test initialization from WKT string."""
        ucrs = UCRS(wgs84_wkt)
        assert ucrs.to_epsg() == 4326

    @pytest.mark.parametrize("epsg_format", [
        "EPSG:4326",
        "epsg:4326",
        "EPSG:3857",
    ])
    def test_from_epsg_string_formats(self, epsg_format: str) -> None:
        """Test various EPSG string format variations."""
        ucrs = UCRS(epsg_format)
        expected_code = int(epsg_format.split(":")[1])
        assert ucrs.to_epsg() == expected_code

    def test_from_proj_string(self) -> None:
        """Test initialization from PROJ string."""
        proj_str = "+proj=longlat +datum=WGS84 +no_defs"
        ucrs = UCRS(proj_str)
        assert isinstance(ucrs, pyproj.CRS)
        assert ucrs.is_geographic


class TestInitializationFromPyproj:
    """Test UCRS initialization from pyproj.CRS objects."""

    def test_from_pyproj_crs(self, wgs84_pyproj: pyproj.CRS) -> None:
        """Test initialization from pyproj.CRS object."""
        ucrs = UCRS(wgs84_pyproj)
        # Should store the same object
        assert ucrs._pyproj_crs is wgs84_pyproj
        assert ucrs.to_epsg() == 4326

    def test_from_pyproj_crs_projected(self, web_mercator_pyproj: pyproj.CRS) -> None:
        """Test initialization from pyproj.CRS (projected)."""
        ucrs = UCRS(web_mercator_pyproj)
        assert ucrs._pyproj_crs is web_mercator_pyproj
        assert ucrs.to_epsg() == 3857

    def test_pyproj_input_not_copied(self, wgs84_pyproj: pyproj.CRS) -> None:
        """Test that pyproj.CRS input is not copied (efficiency)."""
        ucrs = UCRS(wgs84_pyproj)
        # Internal CRS should be the same object (not copied)
        assert ucrs._pyproj_crs is wgs84_pyproj


class TestInitializationFromDict:
    """Test UCRS initialization from PROJ dictionary."""

    def test_from_proj_dict(self, proj_dict: dict[str, str]) -> None:
        """Test initialization from PROJ dictionary."""
        ucrs = UCRS(proj_dict)
        assert isinstance(ucrs, pyproj.CRS)
        assert ucrs.is_geographic

    def test_from_proj_dict_custom(self) -> None:
        """Test initialization from custom PROJ dictionary."""
        proj_dict = {
            "proj": "tmerc",
            "lat_0": "0",
            "lon_0": "15",
            "k": "0.9996",
            "x_0": "500000",
            "y_0": "0",
            "datum": "WGS84",
        }
        ucrs = UCRS(proj_dict)
        assert isinstance(ucrs, pyproj.CRS)
        assert ucrs.is_projected


@requires_cartopy
class TestInitializationFromCartopy:
    """Test UCRS initialization from cartopy CRS objects."""

    def test_from_cartopy_crs(self, wgs84_cartopy: ccrs.CRS) -> None:
        """Test initialization from cartopy.crs.CRS."""
        ucrs = UCRS(wgs84_cartopy)
        assert isinstance(ucrs, pyproj.CRS)
        assert ucrs.is_geographic

    def test_from_cartopy_projection(self, web_mercator_cartopy: ccrs.Projection) -> None:
        """Test initialization from cartopy.crs.Projection."""
        ucrs = UCRS(web_mercator_cartopy)
        assert isinstance(ucrs, pyproj.CRS)
        assert ucrs.is_projected

    @pytest.mark.parametrize("cartopy_crs_class", [
        pytest.param(lambda: __import__("cartopy.crs", fromlist=["PlateCarree"]).PlateCarree(), id="PlateCarree"),
        pytest.param(lambda: __import__("cartopy.crs", fromlist=["Mercator"]).Mercator(), id="Mercator"),
        pytest.param(lambda: __import__("cartopy.crs", fromlist=["Robinson"]).Robinson(), id="Robinson"),
    ])
    def test_from_various_cartopy_projections(self, cartopy_crs_class) -> None:
        """Test initialization from various cartopy projections."""
        crs = cartopy_crs_class()
        ucrs = UCRS(crs)
        assert isinstance(ucrs, pyproj.CRS)


@requires_osgeo
class TestInitializationFromOsgeo:
    """Test UCRS initialization from osgeo SpatialReference objects."""

    def test_from_osgeo_spatial_reference(self, wgs84_osgeo: SpatialReference) -> None:
        """Test initialization from osgeo.osr.SpatialReference."""
        ucrs = UCRS(wgs84_osgeo)
        assert isinstance(ucrs, pyproj.CRS)
        assert ucrs.to_epsg() == 4326

    def test_from_osgeo_projected(self, web_mercator_osgeo: SpatialReference) -> None:
        """Test initialization from osgeo.osr.SpatialReference (projected)."""
        ucrs = UCRS(web_mercator_osgeo)
        assert isinstance(ucrs, pyproj.CRS)
        assert ucrs.to_epsg() == 3857

    def test_osgeo_via_wkt_conversion(self, wgs84_osgeo: SpatialReference) -> None:
        """Test that osgeo input is converted via WKT."""
        ucrs = UCRS(wgs84_osgeo)
        # Should have been converted via WKT
        assert isinstance(ucrs._pyproj_crs, pyproj.CRS)
        assert ucrs.to_epsg() == 4326


class TestInternalState:
    """Test internal state management of UCRS."""

    def test_pyproj_crs_is_set(self, epsg_4326: int) -> None:
        """Test that _pyproj_crs is properly set."""
        ucrs = UCRS(epsg_4326)
        assert hasattr(ucrs, '_pyproj_crs')
        assert isinstance(ucrs._pyproj_crs, pyproj.CRS)

    @requires_cartopy
    def test_cartopy_projection_creates_valid_ucrs(self) -> None:
        """Test that cartopy Projection input creates valid UCRS."""
        import cartopy.crs as ccrs
        proj = ccrs.Mercator()
        ucrs = UCRS(proj)
        assert isinstance(ucrs, pyproj.CRS)
        assert isinstance(ucrs._pyproj_crs, pyproj.CRS)

    def test_inheritance_from_custom_constructor_crs(self, epsg_4326: int) -> None:
        """Test that UCRS inherits from CustomConstructorCRS."""
        ucrs = UCRS(epsg_4326)
        assert isinstance(ucrs, pyproj.crs.CustomConstructorCRS)
        assert isinstance(ucrs, pyproj.CRS)
