"""Tests for UCRS initialization with various input types."""

from __future__ import annotations

from pathlib import Path
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


class TestInitializationFromFile:
    """Test UCRS initialization from WKT files."""

    def test_from_file_string_path(self, tmp_path: Path, wgs84_wkt: str) -> None:
        """Test initialization from file using string path."""
        wkt_file = tmp_path / "test_crs.wkt"
        wkt_file.write_text(wgs84_wkt, encoding="utf-8")

        ucrs = UCRS.from_file(str(wkt_file))
        assert isinstance(ucrs, pyproj.CRS)
        assert ucrs.to_epsg() == 4326
        assert ucrs.is_geographic

    def test_from_file_path_object(self, tmp_path: Path, wgs84_wkt: str) -> None:
        """Test initialization from file using pathlib.Path object."""
        wkt_file = tmp_path / "test_crs.wkt"
        wkt_file.write_text(wgs84_wkt, encoding="utf-8")

        ucrs = UCRS.from_file(wkt_file)
        assert isinstance(ucrs, pyproj.CRS)
        assert ucrs.to_epsg() == 4326
        assert ucrs.is_geographic

    def test_from_file_with_whitespace(self, tmp_path: Path, wgs84_wkt: str) -> None:
        """Test that whitespace is properly stripped from file content."""
        wkt_file = tmp_path / "test_crs.wkt"
        wkt_file.write_text(f"\n\n  {wgs84_wkt}  \n\n", encoding="utf-8")

        ucrs = UCRS.from_file(wkt_file)
        assert ucrs.to_epsg() == 4326

    def test_from_file_projected_crs(self, tmp_path: Path) -> None:
        """Test initialization from file with projected CRS."""
        wkt = pyproj.CRS.from_epsg(3857).to_wkt()
        wkt_file = tmp_path / "projected.wkt"
        wkt_file.write_text(wkt, encoding="utf-8")

        ucrs = UCRS.from_file(wkt_file)
        assert ucrs.to_epsg() == 3857
        assert ucrs.is_projected

    def test_from_file_nonexistent(self, tmp_path: Path) -> None:
        """Test that FileNotFoundError is raised for nonexistent file."""
        nonexistent = tmp_path / "does_not_exist.wkt"
        with pytest.raises(FileNotFoundError):
            UCRS.from_file(nonexistent)

    def test_from_file_invalid_wkt(self, tmp_path: Path) -> None:
        """Test that invalid WKT content raises appropriate error."""
        invalid_file = tmp_path / "invalid.wkt"
        invalid_file.write_text("This is not valid WKT", encoding="utf-8")

        with pytest.raises(Exception):  # pyproj will raise CRSError or similar
            UCRS.from_file(invalid_file)

    def test_from_file_utf8_encoding(self, tmp_path: Path, wgs84_wkt: str) -> None:
        """Test that file is read with UTF-8 encoding."""
        wkt_file = tmp_path / "utf8_test.wkt"
        # Write with UTF-8 explicitly
        wkt_file.write_text(wgs84_wkt, encoding="utf-8")

        ucrs = UCRS.from_file(wkt_file)
        assert ucrs.to_epsg() == 4326

    @pytest.mark.parametrize("epsg_code", [4326, 3857, 32633, 2154])
    def test_from_file_various_crs(self, tmp_path: Path, epsg_code: int) -> None:
        """Test from_file with various CRS definitions."""
        wkt = pyproj.CRS.from_epsg(epsg_code).to_wkt()
        wkt_file = tmp_path / f"crs_{epsg_code}.wkt"
        wkt_file.write_text(wkt, encoding="utf-8")

        ucrs = UCRS.from_file(wkt_file)
        assert ucrs.to_epsg() == epsg_code
