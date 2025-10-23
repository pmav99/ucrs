# UCRS - Unified CRS

Working with geospatial data in Python often means juggling multiple libraries, each with its own CRS representation:
- **pyproj** uses `pyproj.CRS`
- **cartopy** uses `cartopy.crs.CRS` and `cartopy.crs.Projection`
- **GDAL/osgeo** uses `osgeo.osr.SpatialReference`

Converting between these formats is well
[documented](https://pyproj4.github.io/pyproj/stable/crs_compatibility.html),
but a bit tedious.

UCRS solves this by providing a single wrapper class that accepts any CRS input
and lazily converts to an instance of any library you need.

## Usage

```python
from ucrs import UCRS

# Create UCRS from any CRS representation
ucrs = UCRS(4326)                         # EPSG code
ucrs = UCRS("EPSG:4326")                  # EPSG string
ucrs = UCRS(pyproj.CRS.from_epsg(4326))   # pyproj.CRS
ucrs = UCRS(cartopy.crs.PlateCarree())    # cartopy CRS
ucrs = UCRS(srs)                          # osgeo.osr.SpatialReference

# Access different representations
ucrs.proj        # Always available
ucrs.cartopy     # Requires cartopy
ucrs.osgeo       # Requires GDAL
```

## How It Works

UCRS uses `pyproj.CRS` as its internal canonical representation. All conversions follow this pattern:

1. **Input** → `pyproj.CRS` (during initialization)
2. `pyproj.CRS` → **Output format** (via cached properties)

Conversions are lazy and cached, so accessing `.proj`, `.cartopy`, or `.osgeo` multiple times returns the same object without re-conversion.

## Requirements

- Python 3.10+
- pyproj (required)
- cartopy (optional)
- GDAL (optional)

## Installation

```bash
# Minimal installation (pyproj only)
pip install ucrs

# With cartopy support
pip install ucrs[cartopy]

# With GDAL support
pip install ucrs[gdal]

# With all optional dependencies
pip install ucrs[full]
```


## License

MIT License - see LICENSE file for details.
