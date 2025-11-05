"""Tests for CRS conversions between different library formats."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import pyproj

from ucrs import UCRS
from tests.conftest import requires_cartopy, requires_osgeo

if TYPE_CHECKING:
    import cartopy.crs as ccrs
    from osgeo.osr import SpatialReference


@requires_cartopy
class TestCartopyConversion:
    """Test conversion to cartopy CRS/Projection."""

    def test_cartopy_property_geographic(self, epsg_4326: int) -> None:
        """Test that geographic CRS returns cartopy.crs.CRS."""
        import cartopy.crs as ccrs
        ucrs = UCRS(epsg_4326)
        cart_crs = ucrs.cartopy
        assert isinstance(cart_crs, ccrs.CRS)
        assert not isinstance(cart_crs, ccrs.Projection)

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

    def test_cartopy_roundtrip_preserves_crs_type(self, wgs84_cartopy: ccrs.CRS) -> None:
        """Test cartopy CRS -> UCRS -> cartopy roundtrip."""
        import cartopy.crs as ccrs
        ucrs = UCRS(wgs84_cartopy)
        result = ucrs.cartopy
        assert isinstance(result, ccrs.CRS)

    def test_cartopy_projection_roundtrip(self, web_mercator_cartopy: ccrs.Projection) -> None:
        """Test cartopy Projection -> UCRS -> cartopy roundtrip."""
        import cartopy.crs as ccrs
        ucrs = UCRS(web_mercator_cartopy)
        result = ucrs.cartopy
        assert isinstance(result, ccrs.Projection)

    @pytest.mark.parametrize("epsg_code,expected_type", [
        (4326, "CRS"),  # Geographic
        (3857, "Projection"),  # Projected
        (32633, "Projection"),  # UTM Zone 33N
    ])
    def test_cartopy_type_detection(self, epsg_code: int, expected_type: str) -> None:
        """Test that cartopy returns correct type based on CRS characteristics."""
        import cartopy.crs as ccrs
        ucrs = UCRS(epsg_code)
        cart_crs = ucrs.cartopy

        if expected_type == "CRS":
            assert isinstance(cart_crs, ccrs.CRS)
            assert not isinstance(cart_crs, ccrs.Projection)
        else:
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

    def test_osgeo_roundtrip_preserves_epsg(self, wgs84_osgeo: SpatialReference) -> None:
        """Test osgeo -> UCRS -> osgeo roundtrip preserves EPSG code."""
        ucrs = UCRS(wgs84_osgeo)
        result = ucrs.osgeo
        assert result.GetAuthorityCode(None) == "4326"

    def test_osgeo_preserves_epsg_from_int(self, epsg_3857: int) -> None:
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

    @pytest.mark.parametrize("epsg_code", [4326, 3857, 32633])
    def test_osgeo_various_projections(self, epsg_code: int) -> None:
        """Test osgeo conversion works for various projection types."""
        ucrs = UCRS(epsg_code)
        osgeo_crs = ucrs.osgeo
        assert osgeo_crs.GetAuthorityCode(None) == str(epsg_code)


class TestPyprojInterface:
    """Test that UCRS can be used as pyproj.CRS (inheritance)."""

    def test_ucrs_is_pyproj_crs(self, epsg_4326: int) -> None:
        """Test that UCRS instance is a pyproj.CRS."""
        ucrs = UCRS(epsg_4326)
        assert isinstance(ucrs, pyproj.CRS)

    def test_ucrs_has_pyproj_methods(self, epsg_4326: int) -> None:
        """Test that UCRS has pyproj.CRS methods."""
        ucrs = UCRS(epsg_4326)
        assert ucrs.to_epsg() == 4326
        assert ucrs.is_geographic
        assert not ucrs.is_projected

    def test_ucrs_to_wkt(self, epsg_3857: int) -> None:
        """Test that UCRS can export to WKT."""
        ucrs = UCRS(epsg_3857)
        wkt = ucrs.to_wkt()
        assert isinstance(wkt, str)
        assert len(wkt) > 0

    def test_ucrs_name_property(self, epsg_4326: int) -> None:
        """Test that UCRS has CRS name from pyproj."""
        ucrs = UCRS(epsg_4326)
        assert hasattr(ucrs, 'name')
        assert 'WGS' in ucrs.name or '84' in ucrs.name


@requires_cartopy
@requires_osgeo
class TestCrossLibraryConversions:
    """Test conversions between different libraries."""

    def test_cartopy_to_osgeo(self, wgs84_cartopy: ccrs.CRS) -> None:
        """Test conversion from cartopy to osgeo."""
        from osgeo.osr import SpatialReference
        ucrs = UCRS(wgs84_cartopy)
        osgeo_crs = ucrs.osgeo
        assert isinstance(osgeo_crs, SpatialReference)
        assert ucrs.is_geographic

    def test_osgeo_to_cartopy(self, wgs84_osgeo: SpatialReference) -> None:
        """Test conversion from osgeo to cartopy."""
        import cartopy.crs as ccrs
        ucrs = UCRS(wgs84_osgeo)
        cart_crs = ucrs.cartopy
        assert isinstance(cart_crs, ccrs.CRS)
        assert ucrs.to_epsg() == 4326

    def test_int_to_both_libraries(self, epsg_4326: int) -> None:
        """Test conversion from int to both cartopy and osgeo."""
        import cartopy.crs as ccrs
        from osgeo.osr import SpatialReference

        ucrs = UCRS(epsg_4326)

        # Both conversions should work
        cart_crs = ucrs.cartopy
        osgeo_crs = ucrs.osgeo

        assert isinstance(cart_crs, ccrs.CRS)
        assert isinstance(osgeo_crs, SpatialReference)
        assert ucrs.to_epsg() == 4326

    def test_chain_conversions(self, epsg_3857: int) -> None:
        """Test chained conversions: int -> cartopy -> UCRS -> osgeo."""
        import cartopy.crs as ccrs
        from osgeo.osr import SpatialReference

        # Start with cartopy
        cart_proj = ccrs.Mercator.GOOGLE
        ucrs1 = UCRS(cart_proj)

        # Convert to osgeo
        osgeo_crs = ucrs1.osgeo
        assert isinstance(osgeo_crs, SpatialReference)

        # Create new UCRS from osgeo
        ucrs2 = UCRS(osgeo_crs)

        # Convert back to cartopy
        cart_crs = ucrs2.cartopy
        assert isinstance(cart_crs, ccrs.Projection)


class TestConversionConsistency:
    """Test that conversions are consistent across different input types."""

    def test_same_epsg_produces_same_crs(
        self,
        epsg_4326: int,
        epsg_string: str,
        wgs84_pyproj: pyproj.CRS
    ) -> None:
        """Test that same EPSG from different inputs produces equivalent CRS."""
        ucrs1 = UCRS(epsg_4326)
        ucrs2 = UCRS(epsg_string)
        ucrs3 = UCRS(wgs84_pyproj)

        # All should have same EPSG code
        assert ucrs1.to_epsg() == 4326
        assert ucrs2.to_epsg() == 4326
        assert ucrs3.to_epsg() == 4326

        # All should be geographic
        assert ucrs1.is_geographic
        assert ucrs2.is_geographic
        assert ucrs3.is_geographic

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

        cart1 = ucrs1.cartopy
        cart2 = ucrs2.cartopy
        cart3 = ucrs3.cartopy

        # All should be same type
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
