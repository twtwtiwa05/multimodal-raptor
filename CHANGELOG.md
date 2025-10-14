# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2024-12-XX

### Added
-  **Wave-Expansion RAPTOR**: Dual-grid system (50m + 300m) with dynamic PM availability
-  **OSM Dijkstra RAPTOR**: Real road network distances with 4km buffer search  
- **Multimodal Integration**: Buses, subway, 따릉이 bike-sharing, PM e-scooters
- **High Performance**: ~7-8 second queries with realistic results
- **Data Pipeline**: GTFS, OSM, bike station, PM data processing
- **CLI Interface**: Command line routing with `mmraptor` command
- **Python Package**: Clean modular architecture with proper imports

### Technical Achievements
- **Accuracy**: GTFS error rate reduced from 83% to 0.33%
- **Realistic Results**: Gangnam→Yeoksam 5.5min (vs previous 175min bug)
- **Data Scale**: 12K stops, 944 routes, 693 bike stations, 50 PM stations
- **Coverage**: Seoul Gangnam District multimodal network

### Research Foundation  
- **PART2_NEW**: Complete RAPTOR implementation in Python
- **PART2_OTP**: Virtual station innovation (480x performance improvement)
- **PART2_HYBRID**: Zone-based adaptive routing 
- **PART3 Innovations**: Wave-expansion and OSM Dijkstra breakthroughs

### Data Sources
- Korea Transport Database (KTDB) - GTFS transit data
- Seoul Metropolitan Government - Bike station data
- Swing - Anonymized PM usage patterns  
- OpenStreetMap - Real road network geometry

## Research Context

**Author**: Taewoo Kim (Gachon University, Smart City Department)  
**Supervisor**: Jiho Yeo  
**Duration**: August-December 2024  
**Keywords**: RAPTOR, Wave-Expansion, OSM Dijkstra, Multimodal Routing