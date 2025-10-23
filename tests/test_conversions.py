"""Tests for CRS conversions between different library formats."""

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


class TestPropjConversion:
    """Test conversion to pyproj.CRS."""

    def test_proj_property_returns_pyproj_crs(self, epsg_4326: int) -> None:
        """Test that .proj returns pyproj.CRS instance."""
        ucrs = UCRS(epsg_4326)
        proj_crs = ucrs.proj
        assert isinstance(proj_crs, pyproj.CRS)

    def test_proj_property_is_cached(self, epsg_4326: int) -> None:
        """Test that .proj property uses caching."""
        ucrs = UCRS(epsg_4326)
        proj1 = ucrs.proj
        proj2 = ucrs.proj
        assert proj1 is proj2  # Same object due to cached_property

    def test_proj_preserves_geographic_crs(self, wgs84_pyproj: pyproj.CRS) -> None:
        """Test that geographic CRS is preserved in .proj."""
        ucrs = UCRS(wgs84_pyproj)
        assert ucrs.proj.is_geographic
        assert ucrs.proj.to_epsg() == 4326

    def test_proj_preserves_projected_crs(self, web_mercator_pyproj: pyproj.CRS) -> None:
        """Test that projected CRS is preserved in .proj."""
        ucrs = UCRS(web_mercator_pyproj)
        assert ucrs.proj.is_projected
        assert ucrs.proj.to_epsg() == 3857


@requires_cartopy
class TestCartopyConversion:
    """Test conversion to cartopy CRS/Projection."""

    def test_cartopy_property_geographic(self, epsg_4326: int) -> None:
        """Test that geographic CRS returns cartopy.crs.CRS."""
        import cartopy.crs as ccrs
        ucrs = UCRS(epsg_4326)
        cart_crs = ucrs.cartopy
        assert isinstance(cart_crs, ccrs.CRS)

    def test_cartopy_property_projected(self, epsg_3857: int) -> None:
        """Test that projected CRS returns cartopy.crs.Projection."""
        import cartopy.crs as ccrs
        ucrs = UCRS(epsg_3857)
        cart_crs = ucrs.cartopy
        assert isinstance(cart_crs, ccrs.Projection)

    def test_cartopy_property_is_cached(self, epsg_4326: int) -> None:
        """Test that .cartopy property uses caching."""
        ucrs = UCRS(epsg_4326)
        cart1 = ucrs.cartopy
        cart2 = ucrs.cartopy
        assert cart1 is cart2

    def test_cartopy_roundtrip(self, wgs84_cartopy: ccrs.CRS) -> None:
        """Test cartopy -> UCRS -> cartopy roundtrip."""
        import cartopy.crs as ccrs
        ucrs = UCRS(wgs84_cartopy)
        result = ucrs.cartopy
        # Should return either CRS or Projection (both are valid cartopy types)
        assert isinstance(result, (ccrs.CRS, ccrs.Projection))

    def test_cartopy_projection_roundtrip(self, web_mercator_cartopy: ccrs.Projection) -> None:
        """Test cartopy Projection -> UCRS -> cartopy roundtrip."""
        ucrs = UCRS(web_mercator_cartopy)
        result = ucrs.cartopy
        assert isinstance(result, type(web_mercator_cartopy).__bases__[0])  # Base class check

    @pytest.mark.parametrize("epsg_code,expected_type", [
        (4326, "CRS"),  # Geographic
        (3857, "Projection"),  # Projected
        (32633, "Projection"),  # UTM
    ])
    def test_cartopy_type_detection(self, epsg_code: int, expected_type: str) -> None:
        """Test that cartopy returns correct type based on CRS."""
        import cartopy.crs as ccrs
        ucrs = UCRS(epsg_code)
        cart_crs = ucrs.cartopy

        if expected_type == "CRS":
            assert isinstance(cart_crs, ccrs.CRS)
        elif expected_type == "Projection":
            assert isinstance(cart_crs, ccrs.Projection)


@requires_osgeo
class TestOsgeoConversion:
    """Test conversion to osgeo SpatialReference."""

    def test_osgeo_property_returns_spatial_reference(self, epsg_4326: int) -> None:
        """Test that .osgeo returns SpatialReference instance."""
        from osgeo.osr import SpatialReference
        ucrs = UCRS(epsg_4326)
        osgeo_crs = ucrs.osgeo
        assert isinstance(osgeo_crs, SpatialReference)

    def test_osgeo_property_is_cached(self, epsg_4326: int) -> None:
        """Test that .osgeo property uses caching."""
        ucrs = UCRS(epsg_4326)
        osgeo1 = ucrs.osgeo
        osgeo2 = ucrs.osgeo
        assert osgeo1 is osgeo2

    def test_osgeo_roundtrip(self, wgs84_osgeo: SpatialReference) -> None:
        """Test osgeo -> UCRS -> osgeo roundtrip."""
        ucrs = UCRS(wgs84_osgeo)
        result = ucrs.osgeo
        # Compare EPSG codes
        assert result.GetAuthorityCode(None) == "4326"

    def test_osgeo_preserves_epsg(self, epsg_3857: int) -> None:
        """Test that EPSG code is preserved through osgeo conversion."""
        ucrs = UCRS(epsg_3857)
        osgeo_crs = ucrs.osgeo
        assert osgeo_crs.GetAuthorityCode(None) == "3857"

    def test_osgeo_wkt_export(self, epsg_4326: int) -> None:
        """Test that osgeo CRS can export to WKT."""
        ucrs = UCRS(epsg_4326)
        osgeo_crs = ucrs.osgeo
        wkt = osgeo_crs.ExportToWkt()
        assert isinstance(wkt, str)
        assert len(wkt) > 0
        assert "WGS" in wkt or "4326" in wkt


class TestCrossLibraryConversions:
    """Test conversions between different libraries."""

    @requires_cartopy
    @requires_osgeo
    def test_cartopy_to_osgeo(self, wgs84_cartopy: ccrs.CRS) -> None:
        """Test conversion from cartopy to osgeo."""
        from osgeo.osr import SpatialReference
        ucrs = UCRS(wgs84_cartopy)
        osgeo_crs = ucrs.osgeo
        assert isinstance(osgeo_crs, SpatialReference)

    @requires_cartopy
    @requires_osgeo
    def test_osgeo_to_cartopy(self, wgs84_osgeo: SpatialReference) -> None:
        """Test conversion from osgeo to cartopy."""
        import cartopy.crs as ccrs
        ucrs = UCRS(wgs84_osgeo)
        cart_crs = ucrs.cartopy
        assert isinstance(cart_crs, ccrs.CRS)

    @requires_cartopy
    def test_int_to_cartopy_to_proj(self, epsg_4326: int) -> None:
        """Test conversion chain: int -> cartopy -> proj."""
        ucrs = UCRS(epsg_4326)
        cart_crs = ucrs.cartopy
        proj_crs = ucrs.proj
        assert isinstance(cart_crs, type(cart_crs))
        assert isinstance(proj_crs, pyproj.CRS)
        assert proj_crs.to_epsg() == 4326

    @requires_osgeo
    def test_string_to_osgeo_to_proj(self, epsg_string: str) -> None:
        """Test conversion chain: string -> osgeo -> proj."""
        from osgeo.osr import SpatialReference
        ucrs = UCRS(epsg_string)
        osgeo_crs = ucrs.osgeo
        proj_crs = ucrs.proj
        assert isinstance(osgeo_crs, SpatialReference)
        assert isinstance(proj_crs, pyproj.CRS)
        assert proj_crs.to_epsg() == 4326


class TestConversionConsistency:
    """Test that conversions are consistent across different input types."""

    @requires_cartopy
    def test_same_crs_different_inputs_cartopy(
        self,
        epsg_4326: int,
        epsg_string: str,
        wgs84_pyproj: pyproj.CRS
    ) -> None:
        """Test that same CRS from different inputs produces equivalent cartopy objects."""
        ucrs1 = UCRS(epsg_4326)
        ucrs2 = UCRS(epsg_string)
        ucrs3 = UCRS(wgs84_pyproj)

        # All should produce cartopy CRS objects (not exact equality, but same type)
        cart1 = ucrs1.cartopy
        cart2 = ucrs2.cartopy
        cart3 = ucrs3.cartopy

        assert type(cart1) == type(cart2) == type(cart3)

    @requires_osgeo
    def test_same_crs_different_inputs_osgeo(
        self,
        epsg_4326: int,
        epsg_string: str,
        wgs84_pyproj: pyproj.CRS
    ) -> None:
        """Test that same CRS from different inputs produces equivalent osgeo objects."""
        ucrs1 = UCRS(epsg_4326)
        ucrs2 = UCRS(epsg_string)
        ucrs3 = UCRS(wgs84_pyproj)

        osgeo1 = ucrs1.osgeo
        osgeo2 = ucrs2.osgeo
        osgeo3 = ucrs3.osgeo

        # All should have same EPSG code
        assert osgeo1.GetAuthorityCode(None) == "4326"
        assert osgeo2.GetAuthorityCode(None) == "4326"
        assert osgeo3.GetAuthorityCode(None) == "4326"

    def test_same_crs_different_inputs_proj(
        self,
        epsg_4326: int,
        epsg_string: str,
        wgs84_pyproj: pyproj.CRS
    ) -> None:
        """Test that same CRS from different inputs produces equivalent proj objects."""
        ucrs1 = UCRS(epsg_4326)
        ucrs2 = UCRS(epsg_string)
        ucrs3 = UCRS(wgs84_pyproj)

        proj1 = ucrs1.proj
        proj2 = ucrs2.proj
        proj3 = ucrs3.proj

        # All should have same EPSG code
        assert proj1.to_epsg() == 4326
        assert proj2.to_epsg() == 4326
        assert proj3.to_epsg() == 4326
