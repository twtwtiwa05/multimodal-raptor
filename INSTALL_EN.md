# Installation Guide

##  Quick Install

```bash
# Clone repository
git clone https://github.com/username/multimodal-raptor.git
cd multimodal-raptor

# Install package
pip install -e .

# Test installation
mmraptor --origin "37.4979,127.0276" --dest "37.5007,127.0363"
```

##  Requirements

- **Python**: 3.8+ 
- **Memory**: 4GB+ RAM recommended
- **Storage**: ~2GB for complete dataset
- **OS**: Linux, macOS, Windows

##  Development Setup

```bash
# Clone with development extras
git clone https://github.com/username/multimodal-raptor.git
cd multimodal-raptor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install development dependencies
pip install -e ".[dev,viz]"

# Run tests
pytest tests/

# Code formatting
black src/ tests/
ruff check src/ tests/
```

##  Data Setup

The repository includes all necessary data files:

### Pre-processed Data (Ready to Use)
- ✅ **RAPTOR structures**: `data/processed/gangnam_raptor_data/`
- ✅ **OSM road network**: `data/processed/gangnam_road_network.pkl`
- ✅ **PM density maps**: `data/processed/grid_pm_data/`
- ✅ **Cleaned GTFS**: `data/processed/cleaned_gtfs_data/`

### Raw Data (For Reference)
- 📁 **Original GTFS**: `data/raw/gtfs/`
- 📁 **Bike stations**: `data/raw/bike_stations_simple/`
- 📁 **PM usage data**: `data/raw/PM_DATA/`

##  Verify Installation

```python
# Test basic functionality
from mmraptor import MultimodalRouter

router = MultimodalRouter()
routes = router.route(
    origin=(37.4979, 127.0276),    # Gangnam Station
    destination=(37.5007, 127.0363) # Yeoksam Station
)

print(f"Found {len(routes)} routes")
print(f"Best route: {routes[0]['total_time_min']:.1f}min")
```

Expected output:
```
Found 3 routes
Best route: 5.5min
```

## 🔄 Data Rebuilding (Optional)

If you want to rebuild data from scratch:

```bash
# 1. Clean GTFS data
python -m mmraptor.data.gtfs_loader data/raw/gtfs

# 2. Build RAPTOR structures  
python -m mmraptor.data.raptor_builder data/processed/cleaned_gtfs_data

# 3. Run example
python examples/gangnam_quick.py
```

##  Troubleshooting

### Import Errors
```bash
# Ensure scripts are in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/scripts"

# Or install in development mode
pip install -e .
```

### Memory Issues
```bash
# For large datasets, increase memory limit
export MEMORY_LIMIT=8GB
python examples/gangnam_quick.py
```

### File Not Found
```bash
# Verify data files exist
ls -la data/processed/gangnam_raptor_data/raptor_data.pkl
ls -la data/processed/gangnam_road_network.pkl

# Check working directory
pwd  # Should be in multimodal-raptor/
```

### Performance Issues
```bash
# Use smaller test datasets
python examples/gangnam_quick.py --limit 100

# Enable debug logging
export LOG_LEVEL=DEBUG
python examples/gangnam_quick.py
```

##  Platform-Specific Notes

### Windows
```cmd
# Use backslashes for paths
set PYTHONPATH=%PYTHONPATH%;%CD%\scripts

# Virtual environment activation
venv\Scripts\activate
```

### macOS
```bash
# May need to install Xcode command line tools
xcode-select --install

# For Apple Silicon Macs
pip install --upgrade pip setuptools wheel
```

### Linux
```bash
# Install spatial libraries (Ubuntu/Debian)
sudo apt-get install libspatialindex-dev libgeos-dev

# Install spatial libraries (CentOS/RHEL)
sudo yum install spatialindex-devel geos-devel
```

##  Docker (Optional)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN pip install -e .

CMD ["mmraptor", "--help"]
```

```bash
# Build and run
docker build -t multimodal-raptor .
docker run multimodal-raptor mmraptor --origin "37.4979,127.0276" --dest "37.5007,127.0363"
```

##  Next Steps

-  Read the [User Guide](README.md#usage-examples)
-  Run [Test Cases](examples/gangnam_quick.py)
-  Explore [Research Background](CHANGELOG.md)
-  Check [API Documentation](src/mmraptor/)

## 🆘 Getting Help

-  **Bug Reports**: Open an issue on GitHub
-  **Questions**: Check existing issues or start a discussion
-  **Research Collaboration**: Contact twdaniel@gachon.ac.kr
-  **Documentation**: Contribute to improve this guide