# UCRS - Unified CRS

[![PyPI version](https://img.shields.io/pypi/v/ucrs.svg)](https://pypi.org/project/ucrs/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Stop juggling CRS formats. Start using UCRS.**

Working with geospatial data in Python means dealing with different CRS representations:
- **pyproj** → `pyproj.CRS`
- **cartopy** → `cartopy.crs.CRS` and `cartopy.crs.Projection`
- **GDAL/osgeo** → `osgeo.osr.SpatialReference`

Converting between these is [well-documented](https://pyproj4.github.io/pyproj/stable/crs_compatibility.html) but tedious. UCRS provides a single class that accepts any CRS input and seamlessly converts to whatever you need.

## Features

- **Universal Input**: Accepts EPSG codes, WKT, PROJ strings, WKT files, or library-specific CRS objects
- **Lazy Conversion**: Only imports and converts when you access a specific property
- **Cached Results**: Conversions are cached - repeated access returns the same object
- **Type Safe**: Full type annotations for IDE support
- **Inheritance**: UCRS is a `pyproj.CRS`, so it works anywhere pyproj does

## Quick Start

```python
from ucrs import UCRS

# Create from any CRS representation
crs = UCRS(4326)                         # EPSG code
crs = UCRS("EPSG:4326")                  # EPSG string
crs = UCRS("+proj=longlat +datum=WGS84") # PROJ string
crs = UCRS(wkt_string)                   # WKT
crs = UCRS(pyproj.CRS.from_epsg(4326))   # pyproj.CRS
crs = UCRS(cartopy.crs.PlateCarree())    # cartopy CRS
crs = UCRS(srs)                          # osgeo.osr.SpatialReference
crs = UCRS.from_file("path/to/crs.wkt")  # WKT from file

# Use as pyproj.CRS (UCRS inherits from it)
crs.to_epsg()        # Returns 4326
crs.is_geographic    # Returns True
crs.to_wkt()         # WKT string

# Convert to other libraries (lazy, cached)
crs.cartopy          # Returns cartopy CRS (requires cartopy)
crs.osgeo            # Returns SpatialReference (requires GDAL)
```

## Example Use Cases

**Working with matplotlib and cartopy:**

```python
import matplotlib.pyplot as plt
from ucrs import UCRS

# Create from EPSG code
crs = UCRS(3857)  # Web Mercator

# Use in cartopy plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection=crs.cartopy)
ax.coastlines()
```

**Converting between GDAL and pyproj:**

```python
from osgeo import gdal
from ucrs import UCRS

# Get CRS from GDAL dataset
dataset = gdal.Open("myfile.tif")
gdal_srs = dataset.GetSpatialRef()

# Convert to UCRS
crs = UCRS(gdal_srs)

# Now use pyproj methods
print(f"EPSG: {crs.to_epsg()}")
print(f"Is projected: {crs.is_projected}")
```

**Transforming coordinates:**

```python
from pyproj import Transformer
from ucrs import UCRS

wgs84 = UCRS(4326)
web_mercator = UCRS(3857)

# Since UCRS inherits from pyproj.CRS, use it directly
transformer = Transformer.from_crs(wgs84, web_mercator)
x, y = transformer.transform(40.7128, -74.0060)  # NYC coordinates
```

**Loading CRS from WKT files:**

```python
from pathlib import Path
from ucrs import UCRS

# Many GIS workflows store CRS definitions in .wkt or .prj files
crs = UCRS.from_file("projection.prj")  # Shapefile projection file
crs = UCRS.from_file(Path("data/crs.wkt"))  # Also accepts Path objects

# Use immediately with any library
print(f"EPSG: {crs.to_epsg()}")
ax.set_projection(crs.cartopy)
dataset.SetProjection(crs.osgeo.ExportToWkt())
```

## How It Works

UCRS inherits from `pyproj.CRS`, making it a drop-in replacement that works anywhere a pyproj CRS is expected. All conversions follow this pattern:

1. **Input** (any format) → `pyproj.CRS` (during initialization)
2. `pyproj.CRS` → **Output format** (via lazy, cached properties)

Since UCRS inherits from pyproj.CRS, you can use all pyproj methods directly. Conversions to cartopy and osgeo are performed lazily when their properties are first accessed, then cached for subsequent use.

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
pip install ucrs[complete]
```


## License

MIT License - see LICENSE file for details.
