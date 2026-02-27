"""Tests for transform_coords coordinate transformation."""

from __future__ import annotations

import numpy as np
import pytest

from ucrs import UCRS, transform_coords


# ============================================================================
# Reference data: a few lon/lat points and their expected Web Mercator values
# ============================================================================

# (lon, lat) in EPSG:4326
LONLAT = np.array([
    [0.0, 0.0],
    [10.0, 20.0],
    [-73.9857, 40.7484],  # roughly New York
])

# Rough expected (x, y) in EPSG:3857 — Web Mercator meters.
# We only check order-of-magnitude / ballpark, not exact values.
EXPECTED_XY_3857 = np.array([
    [0.0, 0.0],
    [1_113_194.9, 2_273_030.9],
    [-8_237_494.4, 4_970_354.7],
])

WGS84 = 4326
WEB_MERCATOR = 3857


# ============================================================================
# 1. Single (N,2) array → transformed array
# ============================================================================

class TestArrayN2:
    def test_shape_and_ballpark(self) -> None:
        result = transform_coords(LONLAT, WGS84, WEB_MERCATOR)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 2)
        # X values should be in the millions-of-meters range for non-zero inputs
        np.testing.assert_allclose(result, EXPECTED_XY_3857, rtol=1e-3)


# ============================================================================
# 2a. Single (N,3) array with Z, 2D CRS — Z passes through unchanged
# ============================================================================

class TestArrayN3ZPassthrough:
    def test_z_unchanged_2d_crs(self) -> None:
        z_vals = [100.0, 200.0, 300.0]
        coords = np.column_stack([LONLAT, z_vals])
        assert coords.shape == (3, 3)

        result = transform_coords(coords, WGS84, WEB_MERCATOR)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 3)
        # Z should pass through unchanged (neither CRS has a vertical datum)
        np.testing.assert_allclose(result[:, 2], z_vals, atol=1e-9)
        # XY should still be correct
        np.testing.assert_allclose(result[:, :2], EXPECTED_XY_3857, rtol=1e-3)


# ============================================================================
# 2b. Single (N,3) array with Z, 3D CRS — Z is preserved
# ============================================================================

class TestArrayN3Z3DCRS:
    def test_z_present_3d_crs(self) -> None:
        """Transform from EPSG:4979 (WGS84 3D) to EPSG:4326 (2D).

        EPSG:4979 is the 3D geographic CRS (lon, lat, ellipsoidal height).
        Z should still be present in the output.
        """
        WGS84_3D = 4979
        z_vals = [100.0, 200.0, 300.0]
        coords = np.column_stack([LONLAT, z_vals])

        result = transform_coords(coords, WGS84_3D, WGS84)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 3)
        # lon/lat should be essentially unchanged (same horizontal datum)
        np.testing.assert_allclose(result[:, :2], LONLAT, atol=1e-6)


# ============================================================================
# 3. Tuple of 2 arrays → tuple of 2 arrays
# ============================================================================

class TestTuple2Arrays:
    def test_tuple_input_output(self) -> None:
        lons = LONLAT[:, 0]
        lats = LONLAT[:, 1]

        result = transform_coords((lons, lats), WGS84, WEB_MERCATOR)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(a, np.ndarray) for a in result)
        np.testing.assert_allclose(result[0], EXPECTED_XY_3857[:, 0], rtol=1e-3)
        np.testing.assert_allclose(result[1], EXPECTED_XY_3857[:, 1], rtol=1e-3)


# ============================================================================
# 4. Tuple of 3 arrays → tuple of 3 arrays
# ============================================================================

class TestTuple3Arrays:
    def test_tuple_3_input_output(self) -> None:
        lons = LONLAT[:, 0]
        lats = LONLAT[:, 1]
        zs = np.array([100.0, 200.0, 300.0])

        result = transform_coords((lons, lats, zs), WGS84, WEB_MERCATOR)
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(a, np.ndarray) for a in result)
        np.testing.assert_allclose(result[0], EXPECTED_XY_3857[:, 0], rtol=1e-3)
        np.testing.assert_allclose(result[1], EXPECTED_XY_3857[:, 1], rtol=1e-3)
        np.testing.assert_allclose(result[2], zs, atol=1e-9)


# ============================================================================
# 5. output="array" forces ndarray output
# ============================================================================

class TestOutputArray:
    def test_tuple_input_array_output(self) -> None:
        lons = LONLAT[:, 0]
        lats = LONLAT[:, 1]

        result = transform_coords(
            (lons, lats), WGS84, WEB_MERCATOR, output="array"
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 2)


# ============================================================================
# 6. output="tuple" forces tuple output
# ============================================================================

class TestOutputTuple:
    def test_array_input_tuple_output(self) -> None:
        result = transform_coords(
            LONLAT, WGS84, WEB_MERCATOR, output="tuple"
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(a, np.ndarray) for a in result)


# ============================================================================
# 7. output="auto" matches input format
# ============================================================================

class TestOutputAuto:
    @pytest.mark.parametrize(
        "coords,expected_type",
        [
            pytest.param(LONLAT, np.ndarray, id="array_in_array_out"),
            pytest.param(
                (LONLAT[:, 0], LONLAT[:, 1]),
                tuple,
                id="tuple_in_tuple_out",
            ),
        ],
    )
    def test_auto_matches_input(self, coords, expected_type) -> None:  # type: ignore[no-untyped-def]
        result = transform_coords(
            coords, WGS84, WEB_MERCATOR, output="auto"
        )
        assert isinstance(result, expected_type)


# ============================================================================
# 8. Identity transform (same source & target CRS)
# ============================================================================

class TestIdentityTransform:
    def test_same_crs_unchanged(self) -> None:
        result = transform_coords(LONLAT, WGS84, WGS84)
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, LONLAT, atol=1e-10)


# ============================================================================
# 9. List input works
# ============================================================================

class TestListInput:
    def test_plain_list_input(self) -> None:
        coords_list = [[0.0, 0.0], [10.0, 20.0]]
        result = transform_coords(coords_list, WGS84, WEB_MERCATOR)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)

    def test_tuple_of_lists(self) -> None:
        lons = [0.0, 10.0]
        lats = [0.0, 20.0]
        result = transform_coords((lons, lats), WGS84, WEB_MERCATOR)
        assert isinstance(result, tuple)
        assert all(isinstance(a, np.ndarray) for a in result)


# ============================================================================
# 10. Invalid shapes raise ValueError
# ============================================================================

class TestInvalidShapes:
    @pytest.mark.parametrize(
        "coords",
        [
            pytest.param(np.array([1.0, 2.0, 3.0]), id="1d_array"),
            pytest.param(np.array([[1, 2, 3, 4]]), id="n_by_4"),
            pytest.param(np.array([[1]]), id="n_by_1"),
            pytest.param(np.zeros((2, 3, 4)), id="3d_array"),
        ],
    )
    def test_invalid_array_shape(self, coords: np.ndarray) -> None:
        with pytest.raises(ValueError, match="Array input must have shape"):
            transform_coords(coords, WGS84, WEB_MERCATOR)

    @pytest.mark.parametrize(
        "coords",
        [
            pytest.param(([1.0],), id="tuple_of_1"),
            pytest.param(([1.0], [2.0], [3.0], [4.0]), id="tuple_of_4"),
        ],
    )
    def test_invalid_tuple_length(self, coords: tuple[list[float], ...]) -> None:
        with pytest.raises(ValueError, match="Tuple input must have 2 or 3"):
            transform_coords(coords, WGS84, WEB_MERCATOR)

    def test_mismatched_array_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            transform_coords(([1, 2, 3, 4], [1, 2, 3]), WGS84, WEB_MERCATOR)


# ============================================================================
# 11. CRSInput variants all produce the same result
# ============================================================================

class TestCRSInputVariants:
    @pytest.mark.parametrize(
        "source,target",
        [
            pytest.param(4326, 3857, id="int"),
            pytest.param("EPSG:4326", "EPSG:3857", id="string"),
            pytest.param(UCRS(4326), UCRS(3857), id="UCRS"),
        ],
    )
    def test_crs_input_types(self, source, target) -> None:  # type: ignore[no-untyped-def]
        result = transform_coords(LONLAT, source, target)
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, EXPECTED_XY_3857, rtol=1e-3)


# ============================================================================
# 12. Scalar / single-point input
# ============================================================================

class TestScalarInput:
    """Test that plain (x, y) and (x, y, z) float tuples are accepted."""

    def test_scalar_2d_returns_tuple_of_floats(self) -> None:
        result = transform_coords((10.0, 20.0), WGS84, WEB_MERCATOR)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)
        np.testing.assert_allclose(result[0], EXPECTED_XY_3857[1, 0], rtol=1e-3)
        np.testing.assert_allclose(result[1], EXPECTED_XY_3857[1, 1], rtol=1e-3)

    def test_scalar_3d_returns_tuple_of_floats(self) -> None:
        result = transform_coords((10.0, 20.0, 500.0), WGS84, WEB_MERCATOR)
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)
        np.testing.assert_allclose(result[0], EXPECTED_XY_3857[1, 0], rtol=1e-3)
        np.testing.assert_allclose(result[1], EXPECTED_XY_3857[1, 1], rtol=1e-3)
        np.testing.assert_allclose(result[2], 500.0, atol=1e-9)

    def test_scalar_2d_ints(self) -> None:
        """Integer scalars should also work."""
        result = transform_coords((10, 20), WGS84, WEB_MERCATOR)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)

    def test_scalar_identity(self) -> None:
        result = transform_coords((10.0, 20.0), WGS84, WGS84)
        assert isinstance(result, tuple)
        assert len(result) == 2
        np.testing.assert_allclose(result, (10.0, 20.0), atol=1e-10)

    def test_scalar_output_array(self) -> None:
        """output='array' with scalar input returns a (1, 2) array."""
        result = transform_coords((10.0, 20.0), WGS84, WEB_MERCATOR, output="array")
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 2)

    def test_scalar_output_tuple(self) -> None:
        """output='tuple' with scalar input returns tuple of floats."""
        result = transform_coords((10.0, 20.0), WGS84, WEB_MERCATOR, output="tuple")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)

    def test_scalar_vs_array_same_values(self) -> None:
        """Scalar and array paths should produce identical results."""
        scalar_result = transform_coords((10.0, 20.0), WGS84, WEB_MERCATOR)
        array_result = transform_coords(np.array([[10.0, 20.0]]), WGS84, WEB_MERCATOR)
        assert isinstance(scalar_result, tuple)
        assert isinstance(array_result, np.ndarray)
        np.testing.assert_allclose(scalar_result[0], array_result[0, 0])
        np.testing.assert_allclose(scalar_result[1], array_result[0, 1])
