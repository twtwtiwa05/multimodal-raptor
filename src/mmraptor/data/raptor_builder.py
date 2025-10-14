"""
RAPTOR data structure builder
Based on PART1_2.py
"""

import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
import logging
import os

logger = logging.getLogger(__name__)


@dataclass
class Stop:
    """Transit stop/station"""
    stop_id: str
    stop_name: str
    stop_lat: float
    stop_lon: float
    stop_type: int = 0  # 0: bus, 1: subway, 2: bike, 3: PM
    zone_id: str = 'gangnam'


@dataclass  
class Route:
    """Transit route"""
    route_id: str
    route_short_name: str
    route_long_name: str
    route_type: int
    route_color: str = None
    n_trips: int = 0


@dataclass
class Trip:
    """Individual transit trip"""
    trip_id: str
    route_id: str
    service_id: str
    direction_id: int = 0
    stop_times: List = field(default_factory=list)


class RAPTORBuilder:
    """
    Build RAPTOR data structures from GTFS data
    
    Creates optimized data structures for RAPTOR algorithm including
    stops, routes, trips, timetables, and transfers.
    """
    
    def __init__(self, gtfs_dir: str, output_path: str = None):
        """
        Initialize RAPTOR builder
        
        Args:
            gtfs_dir: Path to cleaned GTFS data
            output_path: Path to save RAPTOR data (optional)
        """
        self.gtfs_dir = gtfs_dir
        self.output_path = output_path or os.path.join(
            os.path.dirname(gtfs_dir), 'raptor_data.pkl'
        )
        
        self.stops = []
        self.routes = []
        self.trips = []
        self.stop_times_dict = {}
        self.transfers = []
        
    def build(self) -> Dict[str, Any]:
        """
        Build complete RAPTOR data structure
        
        Returns:
            Dictionary containing all RAPTOR data
        """
        logger.info("🏗️ Building RAPTOR data structures...")
        
        # Load GTFS data
        self._load_gtfs_data()
        
        # Build core structures
        self._build_stops()
        self._build_routes_and_trips() 
        self._build_timetables()
        self._build_transfers()
        
        # Create final data package
        raptor_data = {
            'stops': self.stops,
            'routes': self.routes, 
            'trips': self.trips,
            'stop_times_dict': self.stop_times_dict,
            'transfers': self.transfers,
            'metadata': {
                'n_stops': len(self.stops),
                'n_routes': len(self.routes),
                'n_trips': len(self.trips),
                'build_date': pd.Timestamp.now().isoformat()
            }
        }
        
        if self.output_path:
            self._save_raptor_data(raptor_data)
            
        logger.info(f"✅ RAPTOR build complete: {len(self.stops)} stops, {len(self.routes)} routes")
        return raptor_data
    
    def _load_gtfs_data(self) -> None:
        """Load cleaned GTFS CSV files"""
        self.gtfs_data = {}
        
        files = ['stops', 'routes', 'trips', 'stop_times', 'calendar']
        for filename in files:
            filepath = os.path.join(self.gtfs_dir, f"{filename}.csv")
            if os.path.exists(filepath):
                self.gtfs_data[filename] = pd.read_csv(filepath)
                logger.debug(f"Loaded {filename}: {len(self.gtfs_data[filename])} records")
    
    def _build_stops(self) -> None:
        """Build stop objects from GTFS stops"""
        stops_df = self.gtfs_data['stops']
        
        for _, row in stops_df.iterrows():
            stop = Stop(
                stop_id=str(row['stop_id']),
                stop_name=row['stop_name'],
                stop_lat=float(row['stop_lat']),
                stop_lon=float(row['stop_lon']),
                stop_type=0  # Default to bus stop
            )
            self.stops.append(stop)
    
    def _build_routes_and_trips(self) -> None:
        """Build route and trip objects"""
        routes_df = self.gtfs_data['routes']
        trips_df = self.gtfs_data['trips']
        
        # Build routes
        for _, row in routes_df.iterrows():
            route = Route(
                route_id=str(row['route_id']),
                route_short_name=str(row.get('route_short_name', '')),
                route_long_name=str(row.get('route_long_name', '')),
                route_type=int(row['route_type'])
            )
            self.routes.append(route)
        
        # Build trips
        for _, row in trips_df.iterrows():
            trip = Trip(
                trip_id=str(row['trip_id']),
                route_id=str(row['route_id']),
                service_id=str(row['service_id']),
                direction_id=int(row.get('direction_id', 0))
            )
            self.trips.append(trip)
    
    def _build_timetables(self) -> None:
        """Build timetable structures"""
        stop_times_df = self.gtfs_data['stop_times']
        
        # Group by trip for timetable building
        for trip_id, group in stop_times_df.groupby('trip_id'):
            stop_times = []
            for _, row in group.sort_values('stop_sequence').iterrows():
                # Convert time to minutes since midnight
                arrival_time = self._time_to_minutes(row['arrival_time'])
                departure_time = self._time_to_minutes(row['departure_time'])
                
                stop_times.append({
                    'stop_id': str(row['stop_id']),
                    'arrival_time': arrival_time,
                    'departure_time': departure_time,
                    'stop_sequence': int(row['stop_sequence'])
                })
            
            self.stop_times_dict[str(trip_id)] = stop_times
    
    def _build_transfers(self) -> None:
        """Build transfer connections between nearby stops"""
        # Simple distance-based transfers for now
        # In production, this would use OSM walking distances
        
        stops_array = np.array([[s.stop_lat, s.stop_lon] for s in self.stops])
        
        # Find stops within 300m
        for i, stop1 in enumerate(self.stops):
            for j, stop2 in enumerate(self.stops[i+1:], i+1):
                distance = self._haversine_distance(
                    stop1.stop_lat, stop1.stop_lon,
                    stop2.stop_lat, stop2.stop_lon
                )
                
                if distance <= 300:  # 300m transfer radius
                    transfer_time = max(60, distance / 1.2)  # 1.2 m/s walking speed
                    
                    self.transfers.extend([
                        {
                            'from_stop_id': stop1.stop_id,
                            'to_stop_id': stop2.stop_id,
                            'transfer_time': transfer_time,
                            'distance': distance
                        },
                        {
                            'from_stop_id': stop2.stop_id,
                            'to_stop_id': stop1.stop_id,
                            'transfer_time': transfer_time,
                            'distance': distance
                        }
                    ])
    
    def _time_to_minutes(self, time_str: str) -> int:
        """Convert HH:MM:SS to minutes since midnight"""
        try:
            parts = time_str.split(':')
            hours = int(parts[0])
            minutes = int(parts[1])
            return hours * 60 + minutes
        except:
            return 0
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance in meters"""
        from math import radians, cos, sin, asin, sqrt
        
        R = 6371000  # Earth radius in meters
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        return 2 * R * asin(sqrt(a))
    
    def _save_raptor_data(self, data: Dict[str, Any]) -> None:
        """Save RAPTOR data to pickle file"""
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        with open(self.output_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"💾 RAPTOR data saved: {self.output_path}")


def main():
    """CLI entry point"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m mmraptor.data.raptor_builder <gtfs_dir> [output_path]")
        return
    
    gtfs_dir = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    builder = RAPTORBuilder(gtfs_dir, output_path)
    data = builder.build()
    
    print(f"✅ Built RAPTOR data:")
    print(f"  - Stops: {data['metadata']['n_stops']}")
    print(f"  - Routes: {data['metadata']['n_routes']}")
    print(f"  - Trips: {data['metadata']['n_trips']}")


if __name__ == "__main__":
    main()