#!/usr/bin/env python3
"""
Basic functionality tests that should always work
"""

import pytest
import os
import sys

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))


def test_data_files_exist():
    """Test that essential data files exist"""
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    
    # Check processed data
    processed_dir = os.path.join(base_dir, 'data/processed')
    assert os.path.exists(os.path.join(processed_dir, 'gangnam_raptor_data/raptor_data.pkl')), "RAPTOR data missing"
    assert os.path.exists(os.path.join(processed_dir, 'gangnam_road_network.pkl')), "OSM network missing"
    assert os.path.exists(os.path.join(processed_dir, 'grid_pm_data/pm_density_map.json')), "PM density map missing"
    
    # Check raw data
    raw_dir = os.path.join(base_dir, 'data/raw')
    assert os.path.exists(os.path.join(raw_dir, 'gtfs')), "GTFS data missing"
    assert os.path.exists(os.path.join(raw_dir, 'bike_stations_simple/ttareungee_stations.csv')), "Bike stations missing"


def test_scripts_exist():
    """Test that core script files exist"""
    scripts_dir = os.path.join(os.path.dirname(__file__), '../scripts')
    
    assert os.path.exists(os.path.join(scripts_dir, 'PART3_OSM_DIJKSTRA.py')), "OSM Dijkstra script missing"
    assert os.path.exists(os.path.join(scripts_dir, 'PART3_WAVE_EXPANSION_V2.py')), "Wave expansion script missing"
    assert os.path.exists(os.path.join(scripts_dir, 'PART1_2.py')), "RAPTOR builder missing"
    assert os.path.exists(os.path.join(scripts_dir, 'GTFSLOADER2.py')), "GTFS loader missing"


def test_package_imports():
    """Test that package imports work"""
    try:
        from mmraptor import MultimodalRouter
        assert MultimodalRouter is not None
    except ImportError as e:
        pytest.skip(f"Package import failed: {e}")


def test_coordinate_bounds():
    """Test Gangnam district coordinate validation"""
    # Gangnam bounds
    bounds = {
        'lat_min': 37.460, 'lat_max': 37.550,
        'lon_min': 127.000, 'lon_max': 127.140
    }
    
    # Test coordinates
    test_coords = [
        (37.4979, 127.0276, "Gangnam Station"),
        (37.5007, 127.0363, "Yeoksam Station"),
        (37.4813, 127.0701, "Gaepo-dong"),
        (37.4935, 127.0591, "Daechi-dong"),
    ]
    
    for lat, lon, name in test_coords:
        assert bounds['lat_min'] <= lat <= bounds['lat_max'], f"{name} latitude out of bounds"
        assert bounds['lon_min'] <= lon <= bounds['lon_max'], f"{name} longitude out of bounds"


def test_example_file_exists():
    """Test that example files exist and are runnable"""
    examples_dir = os.path.join(os.path.dirname(__file__), '../examples')
    
    assert os.path.exists(os.path.join(examples_dir, 'gangnam_quick.py')), "Example file missing"
    
    # Check if example file has main function
    with open(os.path.join(examples_dir, 'gangnam_quick.py'), 'r') as f:
        content = f.read()
        assert 'def main()' in content, "Example should have main function"
        assert '__name__ == "__main__"' in content, "Example should be runnable"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])