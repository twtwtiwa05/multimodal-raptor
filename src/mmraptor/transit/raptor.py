"""
RAPTOR algorithm implementations for multimodal routing

This module contains the core RAPTOR (Round-based Public Transit Optimized Router)
implementations including OSM Dijkstra-based and Wave-Expansion variants.
"""

from typing import List, Dict, Tuple, Any, Optional
import sys
import os

# Import existing implementations  
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../scripts'))

try:
    from PART3_OSM_DIJKSTRA import OSMDijkstraRAPTOR as _OSMDijkstraRAPTOR
    from PART3_WAVE_EXPANSION_V2 import DualGridWaveExpansionRAPTOR as _WaveExpansionRAPTOR
except ImportError:
    # Fallback if not available
    _OSMDijkstraRAPTOR = None
    _WaveExpansionRAPTOR = None


class OSMDijkstraRAPTOR:
    """
    OSM Dijkstra-based RAPTOR for multimodal routing
    
    Features:
    - Real road network distances via OSM
    - 4km buffer expansion for access/egress
    - PM density optimization
    - Realistic walking/riding time calculation
    """
    
    def __init__(self, 
                 raptor_data_path: str = None,
                 osm_graph_path: str = None,
                 bike_stations_path: str = None,
                 pm_density_path: str = None,
                 **kwargs):
        """
        Initialize OSM Dijkstra RAPTOR
        
        Args:
            raptor_data_path: Path to RAPTOR data (default: auto-detect)
            osm_graph_path: Path to OSM graph (default: auto-detect)
            bike_stations_path: Path to bike stations (default: auto-detect)
            pm_density_path: Path to PM density map (default: auto-detect)
        """
        if _OSMDijkstraRAPTOR is None:
            raise ImportError("PART3_OSM_DIJKSTRA not available")
        
        # Auto-detect data paths if not provided
        base_dir = os.path.join(os.path.dirname(__file__), '../../../data/processed')
        
        default_paths = {
            'raptor_data_path': raptor_data_path or os.path.join(base_dir, 'gangnam_raptor_data/raptor_data.pkl'),
            'osm_graph_path': osm_graph_path or os.path.join(base_dir, 'gangnam_road_network.pkl'),
            'bike_stations_path': bike_stations_path or os.path.join(base_dir, '../raw/bike_stations_simple/ttareungee_stations.csv'),
            'pm_density_path': pm_density_path or os.path.join(base_dir, 'grid_pm_data/pm_density_map.json')
        }
        
        # Update with any additional kwargs
        default_paths.update(kwargs)
        
        self._raptor = _OSMDijkstraRAPTOR(**default_paths)
    
    def route(self, 
              origin_lat: float, 
              origin_lon: float,
              dest_lat: float, 
              dest_lon: float,
              dep_time: float = 8.0) -> List[Dict[str, Any]]:
        """
        Find multimodal routes between origin and destination
        
        Args:
            origin_lat: Origin latitude
            origin_lon: Origin longitude  
            dest_lat: Destination latitude
            dest_lon: Destination longitude
            dep_time: Departure time (hours, e.g., 8.5 for 08:30)
            
        Returns:
            List of journey dictionaries with segments, times, costs
        """
        return self._raptor.route(origin_lat, origin_lon, dest_lat, dest_lon, dep_time)


class WaveExpansionRAPTOR:
    """
    Wave-Expansion RAPTOR for multimodal routing
    
    Features:
    - Dual-grid system (50m + 300m)
    - Wave expansion up to 4.2km radius
    - PM continuous ride modeling
    - Dynamic density-based availability
    """
    
    def __init__(self, **kwargs):
        if _WaveExpansionRAPTOR is None:
            raise ImportError("PART3_WAVE_EXPANSION_V2 not available")
        self._raptor = _WaveExpansionRAPTOR(**kwargs)
    
    def find_routes(self,
                   origin: Tuple[float, float],
                   destination: Tuple[float, float], 
                   departure_time: float = 8.0,
                   max_expansion_km: float = 4.2) -> List[Dict[str, Any]]:
        """
        Find multimodal routes using wave expansion
        
        Args:
            origin: (lat, lon) tuple
            destination: (lat, lon) tuple
            departure_time: Departure time (hours)
            max_expansion_km: Maximum expansion radius
            
        Returns:
            List of journey dictionaries
        """
        return self._raptor.find_multimodal_routes(
            origin, destination, departure_time, max_expansion_km
        )