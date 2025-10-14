# Data Directory

This directory contains all data required for the Multimodal RAPTOR system.

##  Directory Structure

```
data/
├─ raw/                     # Original source data
│  ├─ gtfs/                 # Korean GTFS transit data (March 2023)
│  ├─ bike_stations_simple/ # Seoul bike-sharing stations (693 stations)
│  └─ PM_DATA/              # Swing e-scooter usage data (9,591 trips)
├─ processed/               # Cleaned and processed data
│  ├─ cleaned_gtfs_data/    # BOM-cleaned GTFS CSV files
│  ├─ gangnam_raptor_data/  # RAPTOR data structures (pickle)
│  ├─ grid_pm_data/         # PM density maps
│  ├─ gangnam_road_network.pkl  # OSM road network (NetworkX)
│  └─ gangnam_road_network.graphml
```

##  Quick Start

### 1. Process Raw Data
```bash
# Clean GTFS data (fixes Korean BOM encoding)
python -m mmraptor.data.gtfs_loader data/raw/gtfs data/processed/cleaned_gtfs_data

# Build RAPTOR structures
python -m mmraptor.data.raptor_builder data/processed/cleaned_gtfs_data data/processed/gangnam_raptor_data/raptor_data.pkl
```

### 2. Use Processed Data
```python
from mmraptor import MultimodalRouter

# Router automatically finds data in standard locations
router = MultimodalRouter(
    raptor_data_path="data/processed/gangnam_raptor_data/raptor_data.pkl",
    osm_graph_path="data/processed/gangnam_road_network.pkl"
)
```

##  Data Sources & Licenses

### 1. GTFS Transit Data
- **Source**: Korea Transport Database (KTDB)
- **Coverage**: Seoul Metropolitan Area, Gangnam District
- **Date**: March 2023
- **Format**: Korean BOM encoding (automatically handled)
- **License**: Public data from Korean government

### 2. Seoul Bike-sharing Stations (따릉이)
- **Source**: Seoul Open Data Plaza
- **Stations**: 693 stations in Gangnam District
- **Features**: Station location, capacity, real-time availability model
- **License**: Seoul Metropolitan Government Open Data License

### 3. Swing PM (E-scooter) Data
- **Source**: Swing (Private company)
- **Trips**: 9,591 anonymized trips in Gangnam area (May 2023)
- **Purpose**: Demand-based virtual station placement
- **Privacy**: Fully anonymized, no personal information
- **License**: Research use with permission

### 4. OpenStreetMap Road Network
- **Source**: OpenStreetMap Contributors
- **Coverage**: Entire Gangnam District road network
- **Purpose**: Real walking/riding distance calculation
- **Format**: NetworkX graph (cached as pkl/graphml)
- **License**: ODbL (Open Database License)

##  Data Statistics

### Transit Network
- **Stops**: 12,064 total (9,404 in Gangnam area)
- **Routes**: 944 routes
- **Trips**: 44,634 scheduled trips
- **Transfer Connections**: 45,406 walking transfers

### Mobility Network
- **Bike Stations**: 693 따릉이 stations
- **Virtual PM Stations**: 50 stations (500 vehicles)
- **OSM Road Nodes**: 8,311 nodes
- **OSM Road Edges**: 22,541 edges

### Geographic Coverage
- **Area**: Seoul Gangnam District
- **Bounds**: 37.460°-37.550°N, 127.000°-127.140°E
- **Population**: ~570,000 residents
- **Area**: ~39.5 km²

##  Data Processing Pipeline

### Error Correction
- **GTFS Error Rate**: Reduced from 83% to 0.33%
- **BOM Encoding**: Automatic UTF-8-BOM detection and cleaning
- **Coordinate Validation**: Gangnam district bounds checking
- **Time Format**: Standardized to minutes since midnight

### Spatial Indexing
- **KDTree**: Fast nearest neighbor searches for stops/stations
- **OSM Snapping**: All stops/stations snapped to road network nodes
- **Distance Caching**: Pre-computed walking/riding distances

### Performance Optimization
- **Pickle Caching**: Binary serialization for fast loading
- **Timetable Compression**: Optimized trip indexing
- **Memory Management**: Lazy loading for large datasets

##  Data Privacy & Security

### Personal Data Protection
- **No Personal Info**: All user data anonymized
- **Location Privacy**: Only aggregated location patterns used
- **GDPR Compliant**: European data protection standards followed

### Data Integrity
- **Checksums**: MD5 verification for large files
- **Version Control**: Git LFS for binary data files
- **Backup**: Regular data validation and backup procedures

##  Data Updates

### Update Frequency
- **GTFS Data**: Monthly updates from KTDB
- **Bike Stations**: Daily API updates (real-time availability)
- **PM Density**: Weekly regeneration from usage patterns
- **OSM Network**: Quarterly updates from OSM

### Update Process
```bash
# Update pipeline (automated)
./scripts/update_data.sh

# Manual verification
python -m mmraptor.data.validate_all
```

##  Support

For data-related issues:
- **GTFS Issues**: Check KTDB official documentation
- **OSM Issues**: Verify with OpenStreetMap
- **Processing Errors**: See troubleshooting in main README
- **Research Data**: Contact authors for access

---

**Last Updated**: December 2024  
**Data Version**: v1.0.0  
**Compatibility**: mmraptor v1.0.0+