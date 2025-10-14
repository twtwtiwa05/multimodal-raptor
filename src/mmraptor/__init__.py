"""
Multimodal RAPTOR: High-performance multimodal transportation routing

This package implements Wave-Expansion and OSM Dijkstra RAPTOR algorithms
for integrated public transit and shared mobility routing in urban areas.
"""

__version__ = "1.0.0"
__author__ = "Taewoo Kim"
__email__ = "twdaniel@gachon.ac.kr"

from .pipeline.route import MultimodalRouter
from .transit.raptor import OSMDijkstraRAPTOR, WaveExpansionRAPTOR

__all__ = [
    "MultimodalRouter", 
    "OSMDijkstraRAPTOR", 
    "WaveExpansionRAPTOR"
]