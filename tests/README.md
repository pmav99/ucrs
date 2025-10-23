# UCRS Test Suite

Comprehensive test suite for the UCRS (Unified CRS) library using pytest.

## Test Structure

```
tests/
├── __init__.py                      # Package marker
├── conftest.py                      # Pytest configuration and fixtures
├── test_initialization.py           # Initialization tests
├── test_conversions.py              # CRS conversion tests
├── test_optional_dependencies.py    # Optional dependency handling
└── test_edge_cases.py               # Edge cases and error handling
```

## Running Tests

### Basic Usage

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_initialization.py

# Run specific test class
pytest tests/test_initialization.py::TestInitializationFromInt

# Run specific test
pytest tests/test_initialization.py::TestInitializationFromInt::test_from_epsg_int_geographic
```

### With Coverage

```bash
# Run with coverage report
pytest --cov=ucrs --cov-report=term-missing

# Generate HTML coverage report
pytest --cov=ucrs --cov-report=html

# Generate coverage with branch analysis
pytest --cov=ucrs --cov-branch --cov-report=term-missing
```

### Parallel Execution

```bash
# Run tests in parallel (requires pytest-xdist)
pytest -n auto

# Run with 4 workers
pytest -n 4
```

### Filtering Tests

```bash
# Skip slow tests
pytest -m "not slow"

# Run only tests that require cartopy
pytest -m requires_cartopy

# Run only tests that require osgeo
pytest -m requires_osgeo

# Skip tests requiring optional dependencies
pytest -m "not requires_cartopy and not requires_osgeo"
```

## Test Categories

### 1. Initialization Tests (`test_initialization.py`)

Tests for UCRS object creation from various input types:
- EPSG integer codes
- EPSG strings
- WKT strings
- pyproj.CRS objects
- PROJ dictionaries
- cartopy CRS/Projection objects (requires cartopy)
- osgeo SpatialReference objects (requires osgeo)

### 2. Conversion Tests (`test_conversions.py`)

Tests for CRS conversions between libraries:
- Conversion to pyproj.CRS (`.proj` property)
- Conversion to cartopy CRS/Projection (`.cartopy` property)
- Conversion to osgeo SpatialReference (`.osgeo` property)
- Cross-library conversions
- Roundtrip conversions
- Consistency across different input types

### 3. Optional Dependencies Tests (`test_optional_dependencies.py`)

Tests for handling missing optional dependencies:
- ImportError when cartopy is not installed
- ImportError when osgeo is not installed
- Helpful error messages with installation instructions
- Core functionality works with only pyproj
- Module-level availability flags

### 4. Edge Cases Tests (`test_edge_cases.py`)

Tests for edge cases and error handling:
- Invalid EPSG codes
- Malformed input strings
- None and empty inputs
- String representations (`__repr__` and `__str__`)
- Boundary conditions
- cached_property behavior
- Thread safety considerations
- Memory behavior
- Custom/local CRS without EPSG codes

## Fixtures

Common fixtures available in all tests (defined in `conftest.py`):

### CRS Fixtures
- `epsg_4326` - WGS84 EPSG code (int)
- `epsg_3857` - Web Mercator EPSG code (int)
- `wgs84_pyproj` - WGS84 as pyproj.CRS
- `web_mercator_pyproj` - Web Mercator as pyproj.CRS
- `wgs84_wkt` - WGS84 as WKT string
- `epsg_string` - "EPSG:4326" string
- `proj_dict` - PROJ dictionary for WGS84

### Cartopy Fixtures (when cartopy is available)
- `wgs84_cartopy` - WGS84 as cartopy.crs.CRS
- `web_mercator_cartopy` - Web Mercator as cartopy.crs.Projection

### OSGEO Fixtures (when osgeo is available)
- `wgs84_osgeo` - WGS84 as SpatialReference
- `web_mercator_osgeo` - Web Mercator as SpatialReference

## Test Markers

Custom pytest markers:
- `@requires_cartopy` - Skip if cartopy not installed
- `@requires_osgeo` - Skip if osgeo/GDAL not installed
- `@pytest.mark.slow` - Mark slow-running tests

## Dependencies

### Required
- pytest >= 8.0.0
- pyproj (main dependency)

### Optional (for running all tests)
- pytest-cov >= 4.1.0 (coverage reporting)
- pytest-xdist >= 3.5.0 (parallel execution)
- cartopy (for cartopy-related tests)
- gdal (for osgeo-related tests)

### Installation

```bash
# Install test dependencies only
pip install -e ".[test]"

# Install with optional CRS libraries
pip install -e ".[test,full]"

# Install development dependencies
pip install -e ".[dev]"
```

## Coverage Goals

The test suite aims for:
- **Line coverage**: > 95%
- **Branch coverage**: > 90%

Key areas covered:
- All input type conversions
- All output format conversions
- Error handling for invalid inputs
- Optional dependency handling
- Caching behavior
- String representations

## Continuous Integration

The test suite is designed to work in CI environments with different dependency configurations:

1. **Minimal**: Only pyproj (core functionality)
2. **Cartopy**: With cartopy but without GDAL
3. **GDAL**: With GDAL but without cartopy
4. **Full**: All optional dependencies

Example CI matrix:
```yaml
matrix:
  deps:
    - "pyproj"
    - "pyproj cartopy"
    - "pyproj gdal"
    - "pyproj cartopy gdal"
```

## Best Practices

1. **Use fixtures**: Leverage shared fixtures from `conftest.py`
2. **Parametrize**: Use `@pytest.mark.parametrize` for similar test cases
3. **Mark dependencies**: Use `@requires_cartopy` and `@requires_osgeo` appropriately
4. **Clear test names**: Use descriptive test names that explain what is being tested
5. **One assertion per concept**: Focus each test on a single behavior
6. **Type annotations**: Include type hints in test functions

## Writing New Tests

When adding new tests:

1. Choose the appropriate test file based on what you're testing
2. Use existing fixtures where possible
3. Add new fixtures to `conftest.py` if needed
4. Apply appropriate markers for optional dependencies
5. Include docstrings explaining what is being tested
6. Follow the existing naming conventions

Example:
```python
@requires_cartopy
class TestNewFeature:
    """Test new cartopy-related feature."""

    def test_specific_behavior(self, wgs84_cartopy: ccrs.CRS) -> None:
        """Test that specific behavior works correctly."""
        ucrs = UCRS(wgs84_cartopy)
        result = ucrs.new_feature()
        assert result == expected_value
```
