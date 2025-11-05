"""Tests for optional dependency handling (cartopy and osgeo)."""

from __future__ import annotations

import pytest
import pyproj

from ucrs import UCRS


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
        from osgeo.osr import SpatialReference  # noqa: F401
        return True
    except ImportError:
        return False


class TestCartopyMissing:
    """Test behavior when cartopy is not installed."""

    @pytest.mark.skipif(_check_cartopy_available(), reason="cartopy is installed")
    def test_cartopy_import_error_when_missing(self, epsg_4326: int) -> None:
        """Test that accessing .cartopy raises ImportError when not installed."""
        ucrs = UCRS(epsg_4326)

        with pytest.raises(ImportError, match="cartopy is not installed"):
            _ = ucrs.cartopy

    @pytest.mark.skipif(_check_cartopy_available(), reason="cartopy is installed")
    def test_cartopy_error_message_helpful(self, epsg_4326: int) -> None:
        """Test that ImportError message includes installation instructions."""
        ucrs = UCRS(epsg_4326)

        with pytest.raises(ImportError, match="pip install cartopy"):
            _ = ucrs.cartopy

    @pytest.mark.skipif(_check_cartopy_available(), reason="cartopy is installed")
    def test_ucrs_works_without_cartopy(self, epsg_4326: int) -> None:
        """Test that UCRS works even when cartopy is missing."""
        ucrs = UCRS(epsg_4326)
        # UCRS is itself a pyproj.CRS
        assert isinstance(ucrs, pyproj.CRS)
        assert ucrs.to_epsg() == 4326


class TestOsgeoMissing:
    """Test behavior when osgeo is not installed."""

    @pytest.mark.skipif(_check_osgeo_available(), reason="osgeo is installed")
    def test_osgeo_import_error_when_missing(self, epsg_4326: int) -> None:
        """Test that accessing .osgeo raises ImportError when not installed."""
        ucrs = UCRS(epsg_4326)

        with pytest.raises(ImportError, match="osgeo .* is not installed"):
            _ = ucrs.osgeo

    @pytest.mark.skipif(_check_osgeo_available(), reason="osgeo is installed")
    def test_osgeo_error_message_helpful(self, epsg_4326: int) -> None:
        """Test that ImportError message includes installation instructions."""
        ucrs = UCRS(epsg_4326)

        with pytest.raises(ImportError, match="pip install gdal"):
            _ = ucrs.osgeo

    @pytest.mark.skipif(_check_osgeo_available(), reason="osgeo is installed")
    def test_ucrs_works_without_osgeo(self, epsg_4326: int) -> None:
        """Test that UCRS works even when osgeo is missing."""
        ucrs = UCRS(epsg_4326)
        # UCRS is itself a pyproj.CRS
        assert isinstance(ucrs, pyproj.CRS)
        assert ucrs.to_epsg() == 4326


class TestBothOptionalMissing:
    """Test behavior when both optional dependencies are missing."""

    @pytest.mark.skipif(
        _check_cartopy_available() or _check_osgeo_available(),
        reason="at least one optional dependency is installed"
    )
    def test_ucrs_works_with_only_pyproj(self, epsg_4326: int) -> None:
        """Test that UCRS works with only pyproj installed."""
        ucrs = UCRS(epsg_4326)
        assert isinstance(ucrs, pyproj.CRS)
        assert ucrs.to_epsg() == 4326

    @pytest.mark.skipif(
        _check_cartopy_available() or _check_osgeo_available(),
        reason="at least one optional dependency is installed"
    )
    def test_initialization_from_string_works(self, epsg_string: str) -> None:
        """Test string initialization works without optional deps."""
        ucrs = UCRS(epsg_string)
        assert ucrs.to_epsg() == 4326

    @pytest.mark.skipif(
        _check_cartopy_available() or _check_osgeo_available(),
        reason="at least one optional dependency is installed"
    )
    def test_initialization_from_pyproj_works(self, wgs84_pyproj: pyproj.CRS) -> None:
        """Test pyproj initialization works without optional deps."""
        ucrs = UCRS(wgs84_pyproj)
        assert ucrs._pyproj_crs is wgs84_pyproj

    @pytest.mark.skipif(
        _check_cartopy_available() or _check_osgeo_available(),
        reason="at least one optional dependency is installed"
    )
    def test_all_pyproj_methods_work(self, epsg_3857: int) -> None:
        """Test that all pyproj.CRS methods work without optional deps."""
        ucrs = UCRS(epsg_3857)
        assert ucrs.to_epsg() == 3857
        assert ucrs.is_projected
        assert not ucrs.is_geographic
        assert ucrs.to_wkt() is not None


class TestImportErrorConsistency:
    """Test that ImportError is raised consistently."""

    @pytest.mark.skipif(_check_cartopy_available(), reason="cartopy is installed")
    def test_cartopy_error_raised_consistently(self, epsg_4326: int) -> None:
        """Test that multiple accesses to .cartopy raise ImportError."""
        ucrs = UCRS(epsg_4326)

        with pytest.raises(ImportError, match="cartopy is not installed"):
            _ = ucrs.cartopy

        # Second access should also raise
        with pytest.raises(ImportError, match="cartopy is not installed"):
            _ = ucrs.cartopy

    @pytest.mark.skipif(_check_osgeo_available(), reason="osgeo is installed")
    def test_osgeo_error_raised_consistently(self, epsg_4326: int) -> None:
        """Test that multiple accesses to .osgeo raise ImportError."""
        ucrs = UCRS(epsg_4326)

        with pytest.raises(ImportError, match="osgeo .* is not installed"):
            _ = ucrs.osgeo

        # Second access should also raise
        with pytest.raises(ImportError, match="osgeo .* is not installed"):
            _ = ucrs.osgeo


class TestGracefulDegradation:
    """Test that UCRS degrades gracefully with missing dependencies."""

    def test_missing_cartopy_doesnt_affect_osgeo(self, epsg_4326: int) -> None:
        """Test that missing cartopy doesn't affect osgeo functionality."""
        if _check_cartopy_available() or not _check_osgeo_available():
            pytest.skip("Requires osgeo but not cartopy")

        ucrs = UCRS(epsg_4326)

        # osgeo should work
        osgeo_crs = ucrs.osgeo
        assert osgeo_crs is not None

        # cartopy should raise ImportError
        with pytest.raises(ImportError):
            _ = ucrs.cartopy

    def test_missing_osgeo_doesnt_affect_cartopy(self, epsg_4326: int) -> None:
        """Test that missing osgeo doesn't affect cartopy functionality."""
        if _check_osgeo_available() or not _check_cartopy_available():
            pytest.skip("Requires cartopy but not osgeo")

        ucrs = UCRS(epsg_4326)

        # cartopy should work
        cart_crs = ucrs.cartopy
        assert cart_crs is not None

        # osgeo should raise ImportError
        with pytest.raises(ImportError):
            _ = ucrs.osgeo

    @pytest.mark.skipif(
        _check_cartopy_available() or _check_osgeo_available(),
        reason="at least one optional dependency is installed"
    )
    def test_neither_dependency_affects_core(self, epsg_4326: int) -> None:
        """Test that missing both deps doesn't affect core functionality."""
        ucrs = UCRS(epsg_4326)

        # Core pyproj.CRS functionality should work
        assert ucrs.to_epsg() == 4326
        assert ucrs.is_geographic
        assert ucrs.to_wkt() is not None

        # Both conversions should raise ImportError
        with pytest.raises(ImportError):
            _ = ucrs.cartopy
        with pytest.raises(ImportError):
            _ = ucrs.osgeo


class TestErrorMessages:
    """Test that error messages are clear and helpful."""

    @pytest.mark.skipif(_check_cartopy_available(), reason="cartopy is installed")
    def test_cartopy_error_message_content(self, epsg_4326: int) -> None:
        """Test cartopy error message is clear."""
        ucrs = UCRS(epsg_4326)

        try:
            _ = ucrs.cartopy
            pytest.fail("Should have raised ImportError")
        except ImportError as e:
            error_msg = str(e)
            assert "cartopy" in error_msg.lower()
            assert "install" in error_msg.lower()
            assert "pip" in error_msg.lower()

    @pytest.mark.skipif(_check_osgeo_available(), reason="osgeo is installed")
    def test_osgeo_error_message_content(self, epsg_4326: int) -> None:
        """Test osgeo error message is clear."""
        ucrs = UCRS(epsg_4326)

        try:
            _ = ucrs.osgeo
            pytest.fail("Should have raised ImportError")
        except ImportError as e:
            error_msg = str(e)
            assert "osgeo" in error_msg.lower() or "gdal" in error_msg.lower()
            assert "install" in error_msg.lower()
            assert "pip" in error_msg.lower()
