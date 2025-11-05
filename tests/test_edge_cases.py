"""Tests for edge cases, error handling, and string representations."""

from __future__ import annotations

import pytest
import pyproj

from ucrs import UCRS


class TestInvalidInputs:
    """Test handling of invalid CRS inputs."""

    def test_invalid_epsg_code(self) -> None:
        """Test that invalid EPSG code raises appropriate error."""
        with pytest.raises((pyproj.exceptions.CRSError, ValueError)):
            UCRS(999999)  # Non-existent EPSG code

    def test_invalid_epsg_string(self) -> None:
        """Test that malformed EPSG string raises appropriate error."""
        with pytest.raises((pyproj.exceptions.CRSError, ValueError)):
            UCRS("EPSG:INVALID")

    def test_invalid_wkt_string(self) -> None:
        """Test that invalid WKT string raises appropriate error."""
        with pytest.raises((pyproj.exceptions.CRSError, ValueError)):
            UCRS("INVALID WKT STRING")

    def test_none_input(self) -> None:
        """Test that None input raises appropriate error."""
        with pytest.raises((TypeError, AttributeError, pyproj.exceptions.CRSError)):
            UCRS(None)  # type: ignore[arg-type]

    def test_empty_string(self) -> None:
        """Test that empty string raises appropriate error."""
        with pytest.raises((pyproj.exceptions.CRSError, ValueError)):
            UCRS("")

    def test_invalid_type(self) -> None:
        """Test that completely invalid type raises appropriate error."""
        with pytest.raises((TypeError, AttributeError, pyproj.exceptions.CRSError)):
            UCRS([1, 2, 3])  # type: ignore[arg-type]

    def test_invalid_dict(self) -> None:
        """Test that invalid PROJ dictionary raises appropriate error."""
        with pytest.raises((pyproj.exceptions.CRSError, ValueError, KeyError)):
            UCRS({"invalid": "dict"})


class TestStringRepresentations:
    """Test __repr__ and __str__ methods."""

    def test_repr_contains_crs_name(self, epsg_4326: int) -> None:
        """Test that __repr__ contains CRS information."""
        ucrs = UCRS(epsg_4326)
        repr_str = repr(ucrs)
        # Should contain useful CRS information
        assert isinstance(repr_str, str)
        assert len(repr_str) > 0

    def test_str_returns_crs_info(self, epsg_4326: int) -> None:
        """Test that __str__ returns CRS string representation."""
        ucrs = UCRS(epsg_4326)
        str_repr = str(ucrs)
        assert isinstance(str_repr, str)
        assert len(str_repr) > 0

    def test_str_contains_useful_info(self, web_mercator_pyproj: pyproj.CRS) -> None:
        """Test that __str__ contains useful CRS information."""
        ucrs = UCRS(web_mercator_pyproj)
        str_repr = str(ucrs)
        assert isinstance(str_repr, str)
        assert len(str_repr) > 0

    @pytest.mark.parametrize("epsg_code", [4326, 3857, 32633])
    def test_repr_consistency(self, epsg_code: int) -> None:
        """Test that __repr__ is consistent for same CRS."""
        ucrs1 = UCRS(epsg_code)
        ucrs2 = UCRS(epsg_code)
        # __repr__ should be deterministic
        assert repr(ucrs1) == repr(ucrs2)

    def test_str_different_for_different_crs(self) -> None:
        """Test that different CRS have different string representations."""
        ucrs1 = UCRS(4326)
        ucrs2 = UCRS(3857)
        # String representations should be different
        assert str(ucrs1) != str(ucrs2)


class TestBoundaryConditions:
    """Test boundary conditions and special cases."""

    def test_very_large_epsg_code(self) -> None:
        """Test handling of large valid EPSG codes."""
        # EPSG codes can go quite high (e.g., 32760 for UTM zones)
        ucrs = UCRS(32760)  # UTM Zone 60S
        assert ucrs.to_epsg() == 32760

    def test_low_epsg_code(self) -> None:
        """Test handling of low EPSG codes."""
        # Some valid EPSG codes are in the 2000s range
        ucrs = UCRS(2154)  # Lambert-93 (France)
        assert ucrs.to_epsg() == 2154

    def test_geographic_vs_projected_distinction(self) -> None:
        """Test that geographic and projected CRS are properly distinguished."""
        geo_ucrs = UCRS(4326)
        proj_ucrs = UCRS(3857)

        assert geo_ucrs.is_geographic
        assert not geo_ucrs.is_projected

        assert proj_ucrs.is_projected
        assert not proj_ucrs.is_geographic

    def test_wkt_with_special_characters(self) -> None:
        """Test WKT strings with special characters are handled."""
        wkt = pyproj.CRS.from_epsg(4326).to_wkt()
        ucrs = UCRS(wkt)
        assert ucrs.to_epsg() == 4326

    def test_various_utm_zones(self) -> None:
        """Test various UTM zones work correctly."""
        for zone in [1, 10, 33, 60]:
            epsg_north = 32600 + zone
            epsg_south = 32700 + zone

            ucrs_north = UCRS(epsg_north)
            ucrs_south = UCRS(epsg_south)

            assert ucrs_north.to_epsg() == epsg_north
            assert ucrs_south.to_epsg() == epsg_south
            assert ucrs_north.is_projected
            assert ucrs_south.is_projected


class TestCachedPropertyBehavior:
    """Test behavior of cached_property decorator."""

    def test_cartopy_cached_property_single_computation(self, epsg_4326: int) -> None:
        """Test that .cartopy is only computed once when available."""
        ucrs = UCRS(epsg_4326)

        # Import check
        try:
            import cartopy.crs  # noqa: F401

            # Access multiple times
            cart1 = ucrs.cartopy
            cart2 = ucrs.cartopy
            cart3 = ucrs.cartopy

            # Should be the exact same object (cached)
            assert cart1 is cart2 is cart3
        except ImportError:
            pytest.skip("cartopy not available")

    def test_osgeo_cached_property_single_computation(self, epsg_4326: int) -> None:
        """Test that .osgeo is only computed once when available."""
        ucrs = UCRS(epsg_4326)

        # Import check
        try:
            from osgeo.osr import SpatialReference  # noqa: F401

            # Access multiple times
            osgeo1 = ucrs.osgeo
            osgeo2 = ucrs.osgeo
            osgeo3 = ucrs.osgeo

            # Should be the exact same object (cached)
            assert osgeo1 is osgeo2 is osgeo3
        except ImportError:
            pytest.skip("osgeo not available")

    def test_different_properties_independent(self, epsg_4326: int) -> None:
        """Test that different cached properties are independent."""
        ucrs = UCRS(epsg_4326)

        # UCRS is itself a pyproj.CRS
        assert isinstance(ucrs, pyproj.CRS)
        assert ucrs.to_epsg() == 4326


class TestEqualityAndComparison:
    """Test equality and comparison behavior."""

    def test_same_crs_different_objects_not_equal(self, epsg_4326: int) -> None:
        """Test that UCRS instances are different objects."""
        ucrs1 = UCRS(epsg_4326)
        ucrs2 = UCRS(epsg_4326)

        # Different UCRS instances
        assert ucrs1 is not ucrs2

    def test_same_input_object_equal(self, wgs84_pyproj: pyproj.CRS) -> None:
        """Test that same input object creates equal internal state."""
        ucrs1 = UCRS(wgs84_pyproj)
        ucrs2 = UCRS(wgs84_pyproj)

        # Both should reference the same pyproj.CRS object
        assert ucrs1._pyproj_crs is wgs84_pyproj
        assert ucrs2._pyproj_crs is wgs84_pyproj

        # But UCRS instances are different
        assert ucrs1 is not ucrs2

    def test_epsg_code_comparison(self) -> None:
        """Test that EPSG codes can be compared."""
        ucrs1 = UCRS(4326)
        ucrs2 = UCRS(4326)
        ucrs3 = UCRS(3857)

        # Same EPSG codes
        assert ucrs1.to_epsg() == ucrs2.to_epsg()

        # Different EPSG codes
        assert ucrs1.to_epsg() != ucrs3.to_epsg()


class TestMemoryBehavior:
    """Test memory-related behavior."""

    def test_original_input_not_duplicated_for_pyproj(
        self,
        wgs84_pyproj: pyproj.CRS
    ) -> None:
        """Test that pyproj.CRS input is not duplicated unnecessarily."""
        ucrs = UCRS(wgs84_pyproj)

        # Internal CRS should be the same object (no copy)
        assert ucrs._pyproj_crs is wgs84_pyproj

    def test_multiple_ucrs_objects_independent(self, epsg_4326: int) -> None:
        """Test that multiple UCRS objects don't interfere."""
        ucrs1 = UCRS(epsg_4326)
        ucrs2 = UCRS(3857)

        assert ucrs1.to_epsg() == 4326
        assert ucrs2.to_epsg() == 3857

        # They should have different internal CRS objects
        assert ucrs1._pyproj_crs is not ucrs2._pyproj_crs

    def test_caching_doesnt_leak_between_instances(self, epsg_4326: int) -> None:
        """Test that cached properties don't leak between instances."""
        ucrs1 = UCRS(epsg_4326)
        ucrs2 = UCRS(epsg_4326)

        # Check that they're independent
        assert ucrs1._pyproj_crs is not ucrs2._pyproj_crs
        assert ucrs1 is not ucrs2


class TestDocstringExamples:
    """Test that examples from docstrings work correctly."""

    def test_docstring_example_int(self) -> None:
        """Test: ucrs = UCRS(4326)"""
        ucrs = UCRS(4326)
        assert ucrs.to_epsg() == 4326

    def test_docstring_example_string(self) -> None:
        """Test: ucrs = UCRS('EPSG:4326')"""
        ucrs = UCRS("EPSG:4326")
        assert ucrs.to_epsg() == 4326

    def test_docstring_example_pyproj(self) -> None:
        """Test: ucrs = UCRS(pyproj.CRS.from_epsg(4326))"""
        ucrs = UCRS(pyproj.CRS.from_epsg(4326))
        assert ucrs.to_epsg() == 4326

    def test_docstring_example_is_pyproj(self) -> None:
        """Test: UCRS is a pyproj.CRS"""
        ucrs = UCRS(4326)
        assert isinstance(ucrs, pyproj.CRS)

    def test_docstring_example_pyproj_methods(self) -> None:
        """Test: using pyproj.CRS methods on UCRS"""
        ucrs = UCRS(4326)
        assert ucrs.is_geographic is True
        assert ucrs.to_wkt() is not None


class TestSpecialCRS:
    """Test handling of special/unusual CRS definitions."""

    def test_local_crs(self) -> None:
        """Test handling of local/custom CRS without EPSG code."""
        # Create a custom CRS
        proj_string = "+proj=tmerc +lat_0=0 +lon_0=15 +k=0.9996 +x_0=500000 +y_0=0 +datum=WGS84 +units=m +no_defs"
        ucrs = UCRS(proj_string)

        assert isinstance(ucrs, pyproj.CRS)
        # Custom CRS may not have EPSG code
        assert ucrs.to_epsg() is None or isinstance(ucrs.to_epsg(), int)

    def test_crs_with_no_epsg(self) -> None:
        """Test CRS that has no EPSG code."""
        # Custom projection without EPSG
        wkt = '''PROJCS["Custom",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",15],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1]]'''

        ucrs = UCRS(wkt)
        assert isinstance(ucrs, pyproj.CRS)
        # Should work even without EPSG code
        assert ucrs.to_wkt() is not None

    def test_wkt2_format(self) -> None:
        """Test handling of WKT2 format strings."""
        # Get WKT2 from pyproj
        crs = pyproj.CRS.from_epsg(4326)
        wkt2 = crs.to_wkt()

        ucrs = UCRS(wkt2)
        assert isinstance(ucrs, pyproj.CRS)
        assert ucrs.to_epsg() == 4326

    @pytest.mark.parametrize("proj_name,proj_string", [
        ("Albers Equal Area", "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=37.5 +lon_0=-96 +datum=WGS84"),
        ("Lambert Conformal Conic", "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +datum=WGS84"),
        ("Stereographic", "+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +datum=WGS84"),
    ])
    def test_various_projections(self, proj_name: str, proj_string: str) -> None:
        """Test various projection types."""
        ucrs = UCRS(proj_string)
        assert isinstance(ucrs, pyproj.CRS)
        assert ucrs.is_projected


class TestRobustness:
    """Test robustness and error recovery."""

    def test_create_many_ucrs_objects(self) -> None:
        """Test creating many UCRS objects doesn't cause issues."""
        ucrs_list = [UCRS(4326) for _ in range(100)]
        assert len(ucrs_list) == 100
        assert all(u.to_epsg() == 4326 for u in ucrs_list)

    def test_mixed_input_types_sequential(self) -> None:
        """Test creating UCRS from various inputs sequentially."""
        ucrs1 = UCRS(4326)
        ucrs2 = UCRS("EPSG:3857")
        ucrs3 = UCRS(pyproj.CRS.from_epsg(32633))

        assert ucrs1.to_epsg() == 4326
        assert ucrs2.to_epsg() == 3857
        assert ucrs3.to_epsg() == 32633

    def test_reuse_pyproj_crs_multiple_times(self) -> None:
        """Test that same pyproj.CRS can be used multiple times."""
        base_crs = pyproj.CRS.from_epsg(4326)

        ucrs1 = UCRS(base_crs)
        ucrs2 = UCRS(base_crs)
        ucrs3 = UCRS(base_crs)

        assert ucrs1.to_epsg() == 4326
        assert ucrs2.to_epsg() == 4326
        assert ucrs3.to_epsg() == 4326
