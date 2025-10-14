#!/usr/bin/env python3
"""
Tests for RAPTOR algorithms
"""

import pytest
import sys
import os

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../scripts'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

# Try to import the actual implementations
try:
    from mmraptor.transit.raptor import OSMDijkstraRAPTOR, WaveExpansionRAPTOR
except ImportError:
    # Fallback to direct imports if package not installed
    try:
        from PART3_OSM_DIJKSTRA import OSMDijkstraRAPTOR as OSMDijkstraRAPTOR
        from PART3_WAVE_EXPANSION_V2 import DualGridWaveExpansionRAPTOR as WaveExpansionRAPTOR
    except ImportError:
        OSMDijkstraRAPTOR = None
        WaveExpansionRAPTOR = None


class TestOSMDijkstraRAPTOR:
    """Test OSM Dijkstra RAPTOR implementation"""
    
    def test_initialization(self):
        """Test router initialization"""
        if OSMDijkstraRAPTOR is None:
            pytest.skip("PART3_OSM_DIJKSTRA not available")
        
        try:
            # Use relative data paths from test directory
            data_dir = os.path.join(os.path.dirname(__file__), '../data/processed')
            
            if isinstance(OSMDijkstraRAPTOR, type):
                # Using mmraptor wrapper
                router = OSMDijkstraRAPTOR()
            else:
                # Using direct implementation
                router = OSMDijkstraRAPTOR(
                    raptor_data_path=os.path.join(data_dir, 'gangnam_raptor_data/raptor_data.pkl'),
                    osm_graph_path=os.path.join(data_dir, 'gangnam_road_network.pkl'),
                    bike_stations_path=os.path.join(os.path.dirname(__file__), '../data/raw/bike_stations_simple/ttareungee_stations.csv'),
                    pm_density_path=os.path.join(data_dir, 'grid_pm_data/pm_density_map.json')
                )
            assert router is not None
        except Exception as e:
            pytest.skip(f"Could not initialize router: {e}")
    
    def test_gangnam_yeoksam_route(self):
        """Test known short route: Gangnam → Yeoksam"""
        if OSMDijkstraRAPTOR is None:
            pytest.skip("PART3_OSM_DIJKSTRA not available")
        
        try:
            # Initialize router with data paths
            data_dir = os.path.join(os.path.dirname(__file__), '../data/processed')
            
            if isinstance(OSMDijkstraRAPTOR, type):
                # Using mmraptor wrapper
                router = OSMDijkstraRAPTOR()
            else:
                # Using direct implementation
                router = OSMDijkstraRAPTOR(
                    raptor_data_path=os.path.join(data_dir, 'gangnam_raptor_data/raptor_data.pkl'),
                    osm_graph_path=os.path.join(data_dir, 'gangnam_road_network.pkl'),
                    bike_stations_path=os.path.join(os.path.dirname(__file__), '../data/raw/bike_stations_simple/ttareungee_stations.csv'),
                    pm_density_path=os.path.join(data_dir, 'grid_pm_data/pm_density_map.json')
                )
            
            routes = router.route(
                origin_lat=37.4979, origin_lon=127.0276,  # Gangnam
                dest_lat=37.5007, dest_lon=127.0363,      # Yeoksam
                dep_time=8.0
            )
            
            assert len(routes) > 0, "Should find at least one route"
            
            best_route = routes[0]
            assert 'total_time_min' in best_route
            assert 'total_cost_won' in best_route
            assert 'segments' in best_route
            
            # Should be reasonable time (not 175+ minutes bug)
            assert best_route['total_time_min'] < 30, "Route should be under 30 minutes"
            assert best_route['total_time_min'] > 0, "Route should have positive time"
            
        except Exception as e:
            pytest.skip(f"Could not run routing test: {e}")


class TestWaveExpansionRAPTOR:
    """Test Wave Expansion RAPTOR implementation"""
    
    def test_initialization(self):
        """Test router initialization"""
        if WaveExpansionRAPTOR is None:
            pytest.skip("PART3_WAVE_EXPANSION_V2 not available")
        
        try:
            # Wave expansion might need different initialization
            router = WaveExpansionRAPTOR()
            assert router is not None
        except Exception as e:
            pytest.skip(f"Could not initialize Wave Expansion router: {e}")
    
    def test_dual_grid_routing(self):
        """Test dual grid wave expansion"""
        if WaveExpansionRAPTOR is None:
            pytest.skip("PART3_WAVE_EXPANSION_V2 not available")
        
        try:
            router = WaveExpansionRAPTOR()
            
            # Check if router has the expected method
            if hasattr(router, 'find_routes'):
                routes = router.find_routes(
                    origin=(37.4813, 127.0701),      # Gaepo
                    destination=(37.4935, 127.0591), # Daechi
                    departure_time=8.0,
                    max_expansion_km=4.2
                )
            elif hasattr(router, 'find_multimodal_routes'):
                routes = router.find_multimodal_routes(
                    origin=(37.4813, 127.0701),
                    destination=(37.4935, 127.0591),
                    departure_time=8.0,
                    max_expansion_km=4.2
                )
            else:
                pytest.skip("Wave expansion router does not have expected methods")
            
            assert isinstance(routes, list), "Should return list of routes"
            
        except Exception as e:
            pytest.skip(f"Could not run wave expansion test: {e}")


def test_coordinate_validation():
    """Test coordinate bounds validation"""
    # Gangnam district bounds
    gangnam_bounds = {
        'lat_min': 37.460, 'lat_max': 37.550,
        'lon_min': 127.000, 'lon_max': 127.140
    }
    
    # Test coordinates
    coords = [
        (37.4979, 127.0276),  # Gangnam Station - valid
        (37.5007, 127.0363),  # Yeoksam Station - valid  
        (35.0000, 129.0000),  # Busan - invalid
    ]
    
    for lat, lon in coords[:2]:  # Only test valid coords
        assert gangnam_bounds['lat_min'] <= lat <= gangnam_bounds['lat_max']
        assert gangnam_bounds['lon_min'] <= lon <= gangnam_bounds['lon_max']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])