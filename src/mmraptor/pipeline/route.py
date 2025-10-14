"""
High-level routing interface
"""

from typing import List, Dict, Tuple, Any, Optional
from ..transit.raptor import OSMDijkstraRAPTOR, WaveExpansionRAPTOR


class MultimodalRouter:
    """
    High-level interface for multimodal routing
    
    Provides a unified API for different RAPTOR variants with
    automatic algorithm selection based on query characteristics.
    """
    
    def __init__(self, 
                 algorithm: str = "osm_dijkstra",
                 **kwargs):
        """
        Initialize router with specified algorithm
        
        Args:
            algorithm: "osm_dijkstra" or "wave_expansion"
            **kwargs: Algorithm-specific parameters
        """
        self.algorithm = algorithm
        
        if algorithm == "osm_dijkstra":
            self.raptor = OSMDijkstraRAPTOR(**kwargs)
        elif algorithm == "wave_expansion":
            self.raptor = WaveExpansionRAPTOR(**kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def route(self,
              origin: Tuple[float, float],
              destination: Tuple[float, float],
              departure_time: float = 8.0,
              **kwargs) -> List[Dict[str, Any]]:
        """
        Find optimal multimodal routes
        
        Args:
            origin: (lat, lon) tuple
            destination: (lat, lon) tuple  
            departure_time: Departure time (hours)
            **kwargs: Algorithm-specific parameters
            
        Returns:
            List of journey dictionaries sorted by optimality
        """
        if self.algorithm == "osm_dijkstra":
            return self.raptor.route(
                origin[0], origin[1], 
                destination[0], destination[1],
                departure_time
            )
        elif self.algorithm == "wave_expansion":
            return self.raptor.find_routes(
                origin, destination, departure_time, **kwargs
            )
    
    def quick_route(self, 
                   origin_str: str, 
                   dest_str: str,
                   departure_time: float = 8.0) -> Dict[str, Any]:
        """
        Quick routing with string coordinates
        
        Args:
            origin_str: "lat,lon" format
            dest_str: "lat,lon" format
            departure_time: Departure time (hours)
            
        Returns:
            Best route dictionary
        """
        origin = tuple(map(float, origin_str.split(',')))
        dest = tuple(map(float, dest_str.split(',')))
        
        routes = self.route(origin, dest, departure_time)
        return routes[0] if routes else {}


def cli_main():
    """Command line interface entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Multimodal RAPTOR Routing')
    parser.add_argument('--origin', required=True, help='Origin lat,lon')
    parser.add_argument('--dest', required=True, help='Destination lat,lon')
    parser.add_argument('--time', type=float, default=8.0, help='Departure time (hours)')
    parser.add_argument('--algorithm', default='osm_dijkstra', 
                       choices=['osm_dijkstra', 'wave_expansion'])
    
    args = parser.parse_args()
    
    router = MultimodalRouter(algorithm=args.algorithm)
    result = router.quick_route(args.origin, args.dest, args.time)
    
    if result:
        print(f"🎯 Best route: {result.get('total_time_min', 0):.1f}min, "
              f"{result.get('total_cost_won', 0)}₩")
        
        for i, segment in enumerate(result.get('segments', []), 1):
            print(f"  {i}. {segment.get('description', 'Unknown segment')}")
    else:
        print("❌ No routes found")


if __name__ == "__main__":
    cli_main()