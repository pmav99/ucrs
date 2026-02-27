"""Tests for transform_geom geometry transformation."""

from __future__ import annotations

import numpy as np
import pytest

from ucrs import UCRS, transform

shapely = pytest.importorskip("shapely")

from shapely import Point, Polygon  # noqa: E402

WGS84 = 4326
WEB_MERCATOR = 3857


# ============================================================================
# 1. Single Point geometry
# ============================================================================

class TestSinglePoint:
    def test_point_4326_to_3857(self) -> None:
        """Transform a Point from 4326→3857, verify Web Mercator coords."""
        pt = Point(10.0, 20.0)
        result = transform(pt, WGS84, WEB_MERCATOR)
        assert isinstance(result, Point)
        # 10° longitude ≈ 1_113_194.9 m in Web Mercator
        np.testing.assert_allclose(result.x, 1_113_194.9, rtol=1e-3)
        np.testing.assert_allclose(result.y, 2_273_030.9, rtol=1e-3)


# ============================================================================
# 2. Single Polygon
# ============================================================================

class TestSinglePolygon:
    def test_polygon_transformed(self) -> None:
        """Transform a simple polygon from 4326→3857."""
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        result = transform(poly, WGS84, WEB_MERCATOR)
        assert isinstance(result, Polygon)
        # All vertices should now be in Web Mercator meters, not degrees
        coords = np.array(result.exterior.coords)
        # At lon=1°, x ≈ 111_319.5 m
        assert np.all(np.abs(coords[:-1]) < 200_000)  # rough sanity check
        assert not np.allclose(coords, np.array(poly.exterior.coords))  # actually changed


# ============================================================================
# 3. Single geometry in → single geometry out (not a list)
# ============================================================================

class TestSingleInSingleOut:
    def test_returns_geometry_not_list(self) -> None:
        pt = Point(0, 0)
        result = transform(pt, WGS84, WEB_MERCATOR)
        assert not isinstance(result, list)
        assert isinstance(result, Point)


# ============================================================================
# 4. List of geometries in → list out
# ============================================================================

class TestListInListOut:
    def test_list_of_points(self) -> None:
        pts = [Point(0, 0), Point(10, 20)]
        result = transform(pts, WGS84, WEB_MERCATOR)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(g, Point) for g in result)
        # Second point should be transformed
        pt = result[1]
        assert isinstance(pt, Point)
        np.testing.assert_allclose(pt.x, 1_113_194.9, rtol=1e-3)


# ============================================================================
# 5. Identity transform — coordinates unchanged
# ============================================================================

class TestIdentityTransform:
    def test_same_crs_unchanged(self) -> None:
        pt = Point(10.0, 20.0)
        result = transform(pt, WGS84, WGS84)
        assert isinstance(result, Point)
        np.testing.assert_allclose(result.x, pt.x, atol=1e-10)
        np.testing.assert_allclose(result.y, pt.y, atol=1e-10)


# ============================================================================
# 6. CRSInput variants all produce the same result
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
        pt = Point(10.0, 20.0)
        result = transform(pt, source, target)
        assert isinstance(result, Point)
        np.testing.assert_allclose(result.x, 1_113_194.9, rtol=1e-3)
