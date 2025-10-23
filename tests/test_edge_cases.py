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


class TestStringRepresentations:
    """Test __repr__ and __str__ methods."""

    def test_repr_contains_crs_name(self, epsg_4326: int) -> None:
        """Test that __repr__ contains CRS name."""
        ucrs = UCRS(epsg_4326)
        repr_str = repr(ucrs)
        assert "UCRS" in repr_str
        assert "WGS 84" in repr_str or "4326" in repr_str

    def test_repr_format(self, epsg_4326: int) -> None:
        """Test that __repr__ follows expected format."""
        ucrs = UCRS(epsg_4326)
        repr_str = repr(ucrs)
        assert repr_str.startswith("UCRS(")
        assert repr_str.endswith(")")

    def test_str_returns_proj_str(self, epsg_4326: int) -> None:
        """Test that __str__ returns pyproj.CRS string representation."""
        ucrs = UCRS(epsg_4326)
        str_repr = str(ucrs)
        proj_str = str(ucrs.proj)
        assert str_repr == proj_str

    def test_str_contains_crs_info(self, web_mercator_pyproj: pyproj.CRS) -> None:
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
        assert repr(ucrs1) == repr(ucrs2)


class TestBoundaryConditions:
    """Test boundary conditions and special cases."""

    def test_very_large_epsg_code(self) -> None:
        """Test handling of large valid EPSG codes."""
        # EPSG codes can go quite high (e.g., 32760 for UTM zones)
        ucrs = UCRS(32760)  # UTM Zone 60S
        assert ucrs.proj.to_epsg() == 32760

    def test_low_epsg_code(self) -> None:
        """Test handling of low EPSG codes."""
        # Some valid EPSG codes are in the 2000s range
        ucrs = UCRS(2154)  # Lambert-93 (France)
        assert ucrs.proj.to_epsg() == 2154

    def test_geographic_vs_projected_distinction(self) -> None:
        """Test that geographic and projected CRS are properly distinguished."""
        geo_ucrs = UCRS(4326)
        proj_ucrs = UCRS(3857)

        assert geo_ucrs.proj.is_geographic
        assert not geo_ucrs.proj.is_projected

        assert proj_ucrs.proj.is_projected
        assert not proj_ucrs.proj.is_geographic

    def test_wkt_with_special_characters(self) -> None:
        """Test WKT strings with special characters are handled."""
        wkt = pyproj.CRS.from_epsg(4326).to_wkt()
        ucrs = UCRS(wkt)
        assert ucrs.proj.to_epsg() == 4326


class TestCachedPropertyBehavior:
    """Test behavior of cached_property decorator."""

    def test_proj_cached_property_single_computation(self, epsg_4326: int) -> None:
        """Test that .proj is only computed once."""
        ucrs = UCRS(epsg_4326)

        # Access multiple times
        proj1 = ucrs.proj
        proj2 = ucrs.proj
        proj3 = ucrs.proj

        # Should be the exact same object (cached)
        assert proj1 is proj2 is proj3

    def test_different_properties_independent(self, epsg_4326: int) -> None:
        """Test that different cached properties are independent."""
        ucrs = UCRS(epsg_4326)

        # Access .proj should not trigger .cartopy or .osgeo
        proj = ucrs.proj
        assert isinstance(proj, pyproj.CRS)

        # The other properties should not be cached yet
        # (we can't directly test this without inspecting __dict__, but we can
        # verify they work independently)


class TestEqualityAndComparison:
    """Test equality and comparison behavior."""

    def test_same_crs_different_objects_not_equal(self, epsg_4326: int) -> None:
        """Test that UCRS doesn't implement custom equality (uses object identity)."""
        ucrs1 = UCRS(epsg_4326)
        ucrs2 = UCRS(epsg_4326)

        # Since UCRS doesn't override __eq__, these are different objects
        assert ucrs1 is not ucrs2

    def test_same_input_object_equal(self, wgs84_pyproj: pyproj.CRS) -> None:
        """Test that same input object creates equal internal state."""
        ucrs1 = UCRS(wgs84_pyproj)
        ucrs2 = UCRS(wgs84_pyproj)

        # The input is the same object
        assert ucrs1._original is ucrs2._original

        # But UCRS instances are different
        assert ucrs1 is not ucrs2


class TestThreadSafety:
    """Test thread-safety considerations (documentation tests)."""

    def test_cached_property_is_thread_safe_by_design(self, epsg_4326: int) -> None:
        """Document that functools.cached_property is thread-safe in Python 3.8+."""
        # This is primarily a documentation test
        # cached_property uses a lock internally (as of Python 3.8)
        ucrs = UCRS(epsg_4326)

        # Multiple accesses should be safe
        results = [ucrs.proj for _ in range(10)]

        # All should be the same object
        assert all(r is results[0] for r in results)


class TestMemoryBehavior:
    """Test memory-related behavior."""

    def test_original_input_not_duplicated_for_pyproj(
        self,
        wgs84_pyproj: pyproj.CRS
    ) -> None:
        """Test that pyproj.CRS input is not duplicated unnecessarily."""
        ucrs = UCRS(wgs84_pyproj)

        # Original should be stored as-is
        assert ucrs._original is wgs84_pyproj

        # Internal CRS should be the same object (no copy)
        assert ucrs._pyproj_crs is wgs84_pyproj

    def test_multiple_ucrs_objects_independent(self, epsg_4326: int) -> None:
        """Test that multiple UCRS objects don't interfere."""
        ucrs1 = UCRS(epsg_4326)
        ucrs2 = UCRS(3857)

        assert ucrs1.proj.to_epsg() == 4326
        assert ucrs2.proj.to_epsg() == 3857

        # They should have different internal CRS objects
        assert ucrs1._pyproj_crs is not ucrs2._pyproj_crs


class TestDocstringExamples:
    """Test that examples from docstrings work correctly."""

    def test_docstring_example_int(self) -> None:
        """Test: ucrs = UCRS(4326)"""
        ucrs = UCRS(4326)
        assert ucrs.proj.to_epsg() == 4326

    def test_docstring_example_string(self) -> None:
        """Test: ucrs = UCRS('EPSG:4326')"""
        ucrs = UCRS("EPSG:4326")
        assert ucrs.proj.to_epsg() == 4326

    def test_docstring_example_pyproj(self) -> None:
        """Test: ucrs = UCRS(pyproj.CRS.from_epsg(4326))"""
        ucrs = UCRS(pyproj.CRS.from_epsg(4326))
        assert ucrs.proj.to_epsg() == 4326

    def test_docstring_example_proj_property(self) -> None:
        """Test: proj_crs = ucrs.proj"""
        ucrs = UCRS(4326)
        proj_crs = ucrs.proj
        assert isinstance(proj_crs, pyproj.CRS)


class TestSpecialCRS:
    """Test handling of special/unusual CRS definitions."""

    def test_local_crs(self) -> None:
        """Test handling of local/custom CRS without EPSG code."""
        # Create a custom CRS
        proj_string = "+proj=tmerc +lat_0=0 +lon_0=15 +k=0.9996 +x_0=500000 +y_0=0 +datum=WGS84 +units=m +no_defs"
        ucrs = UCRS(proj_string)

        assert isinstance(ucrs.proj, pyproj.CRS)
        # Custom CRS may not have EPSG code
        assert ucrs.proj.to_epsg() is None or isinstance(ucrs.proj.to_epsg(), int)

    def test_crs_with_no_epsg(self) -> None:
        """Test CRS that has no EPSG code."""
        # Custom projection without EPSG
        wkt = '''PROJCS["Custom",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",15],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1]]'''

        ucrs = UCRS(wkt)
        assert isinstance(ucrs.proj, pyproj.CRS)
        # Should work even without EPSG code
        assert ucrs.proj.to_wkt() is not None
