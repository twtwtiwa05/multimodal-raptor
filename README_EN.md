# Multimodal RAPTOR

[한국어](README.md) | English

A high-performance multimodal transportation routing system implementing Wave-Expansion and OSM Dijkstra RAPTOR algorithms for Seoul's Gangnam District.

##  Key Features

-  **Wave-Expansion RAPTOR**: Dual-grid system (50m + 300m) with dynamic PM availability
-  **OSM Dijkstra RAPTOR**: Real road network distances instead of straight-line approximations
-  **Complete Transit Integration**: Buses, subway, bike-sharing (따릉이), and e-scooters (PM)
-  **High Performance**: ~7-8 second queries with realistic results
-  **Accuracy**: GTFS error rate reduced from 83% to 0.33%

##  Quick Start

###  Online Demo (No Clone Required!)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/username/multimodal-raptor/main/app.py)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/username/multimodal-raptor?quickstart=1)

** One-Click Demo:**
1. Click "Open in GitHub Codespaces" button above
2. Environment automatically set up
3. Run in terminal: `streamlit run app.py`
4. Click on Gangnam district map in browser to test!

###  Local Installation
```bash
# Web demo
pip install -r requirements_web.txt
streamlit run app.py

# CLI package
pip install -r requirements.txt
python -m mmraptor.pipeline.route --origin "37.4979,127.0276" --dest "37.5007,127.0363"
```

##  Results

**Gangnam → Yeoksam (1.1km)**:
- 🛴 PM Direct: **5.5min**, 1,500₩
- 🚶 Walk: **15.0min**, 0₩  
- 🚌 Transit: **10.2min**, 2,760₩

##  Architecture

```
mmraptor/
├─ data/           # GTFS, OSM, PM data loaders
├─ graph/          # OSM Dijkstra, node snapping
├─ transit/        # RAPTOR core algorithm
├─ modes/          # Walk/PM/bike models
└─ pipeline/       # End-to-end workflows
```

##  Research

This project implements breakthrough algorithms for multimodal urban routing:

- **PART3_WAVE_EXPANSION**: Dual-grid wave expansion with PM continuous ride modeling
- **PART3_OSM_DIJKSTRA**: Real road network integration with 4km buffer search
- **Foundation**: Built on verified PART2 RAPTOR implementations

**Author**: Taewoo Kim (Gachon University, Smart City Department)  
**Supervisor**: Jiho Yeo  
**Duration**: Aug-Dec 2024

##  Performance

- **Data Scale**: 12K stops, 944 routes, 693 bike stations, 50 PM stations
- **Coverage**: Seoul Gangnam District (37.46°-37.55°N, 127.00°-127.14°E)
- **Query Time**: 7-8 seconds with complete route reconstruction
- **Accuracy**: OSM-based real walking/riding distances

##  Technical Stack

- **Core**: Python 3.8+, NetworkX, NumPy, SciPy
- **Spatial**: OSMnx, KDTree (spatial indexing)
- **Data**: GTFS (Korea Transport DB), Seoul Bike API, Swing PM data
- **Optional**: Folium (visualization)

##  Usage

### Python API
```python
from mmraptor import MultimodalRouter

# Initialize router
router = MultimodalRouter(algorithm="osm_dijkstra")

# Find routes
routes = router.route(
    origin=(37.4979, 127.0276),    # Gangnam Station
    destination=(37.5007, 127.0363) # Yeoksam Station
)

# Display results
for i, route in enumerate(routes, 1):
    print(f"Route {i}: {route['total_time_min']:.1f}min, {route['total_cost_won']:,}₩")
    for segment in route['segments']:
        print(f"  - {segment['description']}")
```

### Command Line Interface
```bash
# Basic routing
mmraptor --origin "37.4979,127.0276" --dest "37.5007,127.0363"

# Algorithm selection
mmraptor --origin "37.4979,127.0276" --dest "37.5007,127.0363" --algorithm wave_expansion

# Departure time
mmraptor --origin "37.4979,127.0276" --dest "37.5007,127.0363" --time 8.5
```

### Example Output
```
=== Gangnam Station - Yeoksam Station (Short Distance) ===
✅ 3 journeys found:

💡 Journey 1: 5.5min, 1500₩ (→08:05 arrival, 0 rounds)
   1. 🛴 PM Direct: 1077m (5.5min)
      💰 Cost: 1500₩ | ⏱️ Wait: 1.0min, Ride: 4.5min

💡 Journey 2: 15.0min, 0₩ (→08:14 arrival, 0 rounds)
   1. 🚶 Walk Direct: 1077m (15.0min)

💡 Journey 3: 10.2min, 2760₩ (→08:10 arrival, 3 rounds)
   1. 🚶 Access (walk): 08:00 → 08:00 (0.5min)
   2. 🚌 Bus Seocho09: Gangnam Station Exit 9 → Gangnam.Samsung Electronics
   3. 🚶 Transfer: Gangnam.Samsung Electronics → Gangnam.Yeoksam Tax Office (2.0min)
   4. 🚌 Bus 360: Gangnam.Yeoksam Tax Office → Yeoksam Station.POSCO Tower
   5. 🚶 Egress (bike): 2.2min
```

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **Korea Transport Database (KTDB)** - High-quality GTFS data
- **Seoul Metropolitan Government** - Public bike station data  
- **Swing** - Anonymized PM usage data
- **OpenStreetMap Contributors** - Detailed road network data
- **Gachon University Smart City Department** - Research environment support

---

## 📞 Contact

**Author**: Taewoo Kim  
**University**: Gachon University  
**Department**: Smart City Department, Undergraduate Researcher  
**Email**: twdaniel@gachon.ac.kr  

**Supervisor**: Jiho Yeo

---

**Project Duration**: August-December 2024  
**Research Areas**: Multimodal Transportation, Urban Mobility, Algorithm Innovation  

**Keywords**: RAPTOR, Wave-Expansion, OSM Dijkstra, Multimodal Routing, Public Transit, Shared Mobility, Seoul