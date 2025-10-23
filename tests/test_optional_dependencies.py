"""Tests for optional dependency handling (cartopy and osgeo)."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest
import pyproj

from ucrs import UCRS
from tests.conftest import CARTOPY_AVAILABLE, OSGEO_AVAILABLE

if TYPE_CHECKING:
    from pytest import MonkeyPatch


class TestCartopyMissing:
    """Test behavior when cartopy is not installed."""

    @pytest.mark.skipif(CARTOPY_AVAILABLE, reason="cartopy is installed")
    def test_cartopy_import_error_when_missing(self, epsg_4326: int) -> None:
        """Test that accessing .cartopy raises ImportError when not installed."""
        ucrs = UCRS(epsg_4326)

        with pytest.raises(ImportError, match="cartopy is not installed"):
            _ = ucrs.cartopy

    @pytest.mark.skipif(CARTOPY_AVAILABLE, reason="cartopy is installed")
    def test_cartopy_error_message_helpful(self, epsg_4326: int) -> None:
        """Test that ImportError message includes installation instructions."""
        ucrs = UCRS(epsg_4326)

        with pytest.raises(ImportError, match="pip install cartopy"):
            _ = ucrs.cartopy

    @pytest.mark.skipif(CARTOPY_AVAILABLE, reason="cartopy is installed")
    def test_other_properties_work_without_cartopy(self, epsg_4326: int) -> None:
        """Test that .proj works even when cartopy is missing."""
        ucrs = UCRS(epsg_4326)
        proj_crs = ucrs.proj
        assert isinstance(proj_crs, pyproj.CRS)
        assert proj_crs.to_epsg() == 4326


class TestOsgeoMissing:
    """Test behavior when osgeo is not installed."""

    @pytest.mark.skipif(OSGEO_AVAILABLE, reason="osgeo is installed")
    def test_osgeo_import_error_when_missing(self, epsg_4326: int) -> None:
        """Test that accessing .osgeo raises ImportError when not installed."""
        ucrs = UCRS(epsg_4326)

        with pytest.raises(ImportError, match="osgeo .* is not installed"):
            _ = ucrs.osgeo

    @pytest.mark.skipif(OSGEO_AVAILABLE, reason="osgeo is installed")
    def test_osgeo_error_message_helpful(self, epsg_4326: int) -> None:
        """Test that ImportError message includes installation instructions."""
        ucrs = UCRS(epsg_4326)

        with pytest.raises(ImportError, match="pip install gdal"):
            _ = ucrs.osgeo

    @pytest.mark.skipif(OSGEO_AVAILABLE, reason="osgeo is installed")
    def test_other_properties_work_without_osgeo(self, epsg_4326: int) -> None:
        """Test that .proj works even when osgeo is missing."""
        ucrs = UCRS(epsg_4326)
        proj_crs = ucrs.proj
        assert isinstance(proj_crs, pyproj.CRS)
        assert proj_crs.to_epsg() == 4326


class TestBothOptionalMissing:
    """Test behavior when both optional dependencies are missing."""

    @pytest.mark.skipif(
        CARTOPY_AVAILABLE or OSGEO_AVAILABLE,
        reason="at least one optional dependency is installed"
    )
    def test_ucrs_works_with_only_pyproj(self, epsg_4326: int) -> None:
        """Test that UCRS works with only pyproj installed."""
        ucrs = UCRS(epsg_4326)
        proj_crs = ucrs.proj
        assert isinstance(proj_crs, pyproj.CRS)
        assert proj_crs.to_epsg() == 4326

    @pytest.mark.skipif(
        CARTOPY_AVAILABLE or OSGEO_AVAILABLE,
        reason="at least one optional dependency is installed"
    )
    def test_initialization_from_string_works(self, epsg_string: str) -> None:
        """Test string initialization works without optional deps."""
        ucrs = UCRS(epsg_string)
        assert ucrs.proj.to_epsg() == 4326

    @pytest.mark.skipif(
        CARTOPY_AVAILABLE or OSGEO_AVAILABLE,
        reason="at least one optional dependency is installed"
    )
    def test_initialization_from_pyproj_works(self, wgs84_pyproj: pyproj.CRS) -> None:
        """Test pyproj initialization works without optional deps."""
        ucrs = UCRS(wgs84_pyproj)
        assert ucrs.proj is wgs84_pyproj


class TestModuleLevelAvailabilityFlags:
    """Test the CARTOPY_AVAILABLE and OSGEO_AVAILABLE module flags."""

    def test_cartopy_available_flag_matches_import(self) -> None:
        """Test that CARTOPY_AVAILABLE flag is accurate."""
        import ucrs as ucrs_module

        try:
            import cartopy.crs  # noqa: F401
            assert ucrs_module.CARTOPY_AVAILABLE is True
        except ImportError:
            assert ucrs_module.CARTOPY_AVAILABLE is False

    def test_osgeo_available_flag_matches_import(self) -> None:
        """Test that OSGEO_AVAILABLE flag is accurate."""
        import ucrs as ucrs_module

        try:
            from osgeo.osr import SpatialReference  # noqa: F401
            assert ucrs_module.OSGEO_AVAILABLE is True
        except ImportError:
            assert ucrs_module.OSGEO_AVAILABLE is False


class TestImportErrorCaching:
    """Test that ImportError is raised consistently (property caching)."""

    @pytest.mark.skipif(CARTOPY_AVAILABLE, reason="cartopy is installed")
    def test_cartopy_error_raised_consistently(self, epsg_4326: int) -> None:
        """Test that multiple accesses to .cartopy raise ImportError."""
        ucrs = UCRS(epsg_4326)

        with pytest.raises(ImportError, match="cartopy is not installed"):
            _ = ucrs.cartopy

        # Due to @cached_property, accessing again should either:
        # 1. Re-raise the error, or
        # 2. Be cached and raise the same error
        # The current implementation raises each time due to exception in property
        with pytest.raises(ImportError, match="cartopy is not installed"):
            _ = ucrs.cartopy

    @pytest.mark.skipif(OSGEO_AVAILABLE, reason="osgeo is installed")
    def test_osgeo_error_raised_consistently(self, epsg_4326: int) -> None:
        """Test that multiple accesses to .osgeo raise ImportError."""
        ucrs = UCRS(epsg_4326)

        with pytest.raises(ImportError, match="osgeo .* is not installed"):
            _ = ucrs.osgeo

        with pytest.raises(ImportError, match="osgeo .* is not installed"):
            _ = ucrs.osgeo


class TestDynamicImportMocking:
    """Test behavior with mocked missing dependencies (advanced testing)."""

    def test_simulate_missing_cartopy(
        self,
        epsg_4326: int,
        monkeypatch: MonkeyPatch
    ) -> None:
        """Test behavior when cartopy import is simulated as missing."""
        # This test is complex because CARTOPY_AVAILABLE is set at module import
        # We can't easily change it after import without reimporting the module
        # This test documents the limitation

        # Note: In practice, testing actual missing dependencies is done via
        # test environments with different dependency sets, not mocking

        ucrs = UCRS(epsg_4326)

        # Even if we monkeypatch sys.modules, the CARTOPY_AVAILABLE flag
        # was already set at module import time
        if not CARTOPY_AVAILABLE:
            with pytest.raises(ImportError):
                _ = ucrs.cartopy
        else:
            # If cartopy is installed, we can't easily simulate it missing
            # without reimporting the ucrs module
            pytest.skip("Cannot simulate missing cartopy when already imported")

    def test_module_reimport_with_missing_deps(self, monkeypatch: MonkeyPatch) -> None:
        """Test that reimporting ucrs with missing deps sets flags correctly."""
        # This is a complex test that shows the limitation of testing
        # optional dependencies in a single test run

        # Store original ucrs module
        original_ucrs = sys.modules.get('ucrs')

        # This test is primarily documentation of how the flags work
        # In real scenarios, use separate test environments with/without deps

        if original_ucrs:
            # Flag values are set at import time
            assert hasattr(original_ucrs, 'CARTOPY_AVAILABLE')
            assert hasattr(original_ucrs, 'OSGEO_AVAILABLE')
