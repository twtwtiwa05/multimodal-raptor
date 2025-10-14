#!/usr/bin/env python3
"""
PART3: OSM 도로망 기반 다익스트라 액세스/이그레스 RAPTOR v3.0
- OSM 그래프에서 다익스트라로 시간 등고선 생성
- PM(킥보드) + 따릉이(도킹형 자전거) 액세스/이그레스 지원
- 격자 방식 대신 실제 도로망 기반 정확한 시간 계산
"""

import pickle
import json
import math
import time
import logging
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import heapq
from scipy.spatial import KDTree

# 기존 모듈 import
try:
    from PART1_2 import Stop, Route, Trip
except ImportError:
    print("PART1_2.py가 필요합니다.")
    exit(1)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ============================================================================
# 상수 및 파라미터
# ============================================================================

# 속도 (m/s)
SPEED_WALK_MPS = 1.2        # 보행속도: 1.2 m/s (4.3 km/h)
SPEED_PM_MPS = 4.0          # PM(킥보드) 속도: 4.0 m/s (14.4 km/h)  
SPEED_BIKE_MPS = 3.5        # 따릉이 속도: 3.5 m/s (12.6 km/h)

# 시간 한계 (초)
MAX_ACCESS_TIME_SEC = 15 * 60    # 최대 액세스 시간: 15분
MAX_EGRESS_TIME_SEC = 30 * 60    # 최대 이그레스 시간: 30분 (늘려서 테스트)

# PM 대기시간 (초) - 밀도 기반
PM_BASE_WAIT_SEC = 60       # PM 기본 대기시간: 1분
PM_DENSITY_THRESHOLD = 0.1  # PM 밀도 임계값

# 따릉이 파라미터
BIKE_PICKUP_TIME_SEC = 30   # 자전거 픽업/반납 시간: 30초
BIKE_DOCK_RADIUS_M = 200    # 대여소 탐색 반경: 200m

# 최소 시간 임계값
MIN_SEGMENT_TIME_MIN = 0.5  # 최소 세그먼트 시간: 0.5분

# ============================================================================
# 데이터 구조
# ============================================================================

@dataclass
class AccessResult:
    """액세스/이그레스 결과"""
    stop_id: str
    stop_coords: Tuple[float, float]
    access_time_sec: float
    mode: str  # 'walk', 'pm', 'bike'
    access_path: List[int]  # OSM 노드 경로
    mode_details: Dict[str, Any] = field(default_factory=dict)  # 모드별 세부정보
    # 추가 상세 정보
    station_name: str = None  # 따릉이 대여소 이름
    station_id: str = None    # 따릉이 대여소 ID
    station_lat: float = None # 따릉이 대여소 위도
    station_lon: float = None # 따릉이 대여소 경도
    grid_info: dict = field(default_factory=dict)  # PM 격자 정보

@dataclass
class BikeStation:
    """따릉이 대여소"""
    station_id: str
    station_name: str
    lat: float
    lon: float
    n_bikes: int
    osm_node_id: int

# ============================================================================
# 메인 클래스
# ============================================================================

class OSMDijkstraRAPTOR:
    """OSM 기반 다익스트라 액세스/이그레스 RAPTOR"""
    
    def __init__(self, 
                 raptor_data_path: str = "gangnam_raptor_data/raptor_data.pkl",
                 osm_graph_path: str = "gangnam_road_network.pkl",
                 bike_stations_path: str = "bike_stations_simple/ttareungee_stations.csv",
                 pm_density_path: str = "grid_pm_data/pm_density_map.json"):
        
        logger.info("=== OSM 다익스트라 RAPTOR v3.0 초기화 ===")
        
        # 데이터 로드
        self._load_raptor_data(raptor_data_path)
        self._load_osm_graph(osm_graph_path)
        self._load_bike_stations(bike_stations_path)
        self._load_pm_density(pm_density_path)
        
        # OSM 엣지에 시간 가중치 추가
        self._preprocess_osm_edges()
        
        logger.info("초기화 완료")
    
    def _load_raptor_data(self, path: str):
        """RAPTOR 데이터 로드"""
        logger.info(f"RAPTOR 데이터 로드: {path}")
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.stops = data['stops']
        self.routes = data['routes']
        self.timetables = data['timetables']
        self.transfers = data['transfers']
        self.route_stops = data.get('route_stop_sequences', {})
        
        # 정류장 인덱싱
        self.stop_index_map = {stop_id: idx for idx, stop_id in enumerate(self.stops.keys())}
        self.index_to_stop = {idx: stop_id for stop_id, idx in self.stop_index_map.items()}
        self.stop_names = {stop_id: stop.stop_name for stop_id, stop in self.stops.items()}
        
        # 정류장 좌표 배열 (KDTree용)
        self.stop_coords = []
        self.stop_ids_list = []
        for stop_id, stop in self.stops.items():
            self.stop_coords.append([stop.stop_lat, stop.stop_lon])
            self.stop_ids_list.append(stop_id)
        
        self.stop_kdtree = KDTree(np.array(self.stop_coords))
        
        logger.info(f"정류장 {len(self.stops)}개, 노선 {len(self.routes)}개 로드")
    
    def _load_osm_graph(self, path: str):
        """OSM 그래프 로드"""
        logger.info(f"OSM 그래프 로드: {path}")
        
        with open(path, 'rb') as f:
            self.osm_graph = pickle.load(f)
        
        logger.info(f"OSM 노드 {len(self.osm_graph.nodes)}개, 엣지 {len(self.osm_graph.edges)}개 로드")
        
        # 정류장을 OSM 노드에 스냅
        self._snap_stops_to_osm()
    
    def _load_bike_stations(self, path: str):
        """따릉이 대여소 데이터 로드"""
        logger.info(f"따릉이 대여소 로드: {path}")
        
        import pandas as pd
        df = pd.read_csv(path)
        
        self.bike_stations = {}
        bike_coords = []
        
        for _, row in df.iterrows():
            station = BikeStation(
                station_id=row['station_id'],
                station_name=row['station_name'],
                lat=row['lat'],
                lon=row['lon'],
                n_bikes=row['n_bikes'],
                osm_node_id=None  # 나중에 스냅
            )
            
            self.bike_stations[station.station_id] = station
            bike_coords.append([station.lat, station.lon])
        
        # 대여소 KDTree
        if bike_coords:
            self.bike_coords = np.array(bike_coords)
            self.bike_kdtree = KDTree(self.bike_coords)
            self.bike_ids_list = list(self.bike_stations.keys())
            
            # 대여소를 OSM 노드에 스냅
            self._snap_bike_stations_to_osm()
        
        logger.info(f"따릉이 대여소 {len(self.bike_stations)}개 로드")
    
    def _load_pm_density(self, path: str):
        """PM 밀도 데이터 로드"""
        logger.info(f"PM 밀도 데이터 로드: {path}")
        
        with open(path, 'r') as f:
            pm_data = json.load(f)
        
        self.pm_density_map = pm_data['density_map']
        logger.info(f"PM 밀도 격자 {len(self.pm_density_map)}개 로드")
    
    def _preprocess_osm_edges(self):
        """OSM 엣지에 시간 가중치 추가"""
        logger.info("OSM 엣지 시간 가중치 전처리...")
        
        edges_processed = 0
        # 엣지 속성을 올바르게 설정하는 방법
        for u, v, key in self.osm_graph.edges(keys=True):
            edge_data = self.osm_graph.edges[u, v, key]
            length = edge_data.get('length', 100)  # 기본 100m
            
            # 각 모드별 이동시간 계산 (초)
            edge_data['travel_time_walk'] = length / SPEED_WALK_MPS
            edge_data['travel_time_pm'] = length / SPEED_PM_MPS
            edge_data['travel_time_bike'] = length / SPEED_BIKE_MPS
            edges_processed += 1
        
        logger.info(f"엣지 전처리 완료: {edges_processed}개 엣지 처리")
    
    def _snap_stops_to_osm(self):
        """정류장을 OSM 노드에 스냅"""
        logger.info("정류장 OSM 스냅...")
        
        self.stop_osm_nodes = {}
        
        for stop_id, stop in self.stops.items():
            # 가장 가까운 OSM 노드 찾기
            nearest_node = self._find_nearest_osm_node(stop.stop_lat, stop.stop_lon)
            self.stop_osm_nodes[stop_id] = nearest_node
        
        logger.info(f"정류장 {len(self.stop_osm_nodes)}개 OSM 스냅 완료")
    
    def _snap_bike_stations_to_osm(self):
        """따릉이 대여소를 OSM 노드에 스냅"""
        logger.info("따릉이 대여소 OSM 스냅...")
        
        for station_id, station in self.bike_stations.items():
            nearest_node = self._find_nearest_osm_node(station.lat, station.lon)
            station.osm_node_id = nearest_node
        
        logger.info(f"따릉이 대여소 {len(self.bike_stations)}개 OSM 스냅 완료")
    
    def _find_nearest_osm_node(self, lat: float, lon: float) -> int:
        """좌표에서 가장 가까운 OSM 노드 찾기 (50m 허용)"""
        min_dist = float('inf')
        nearest_node = None
        
        for node_id, data in self.osm_graph.nodes(data=True):
            node_lat = data.get('y', 0)
            node_lon = data.get('x', 0)
            
            # 대략적인 미터 거리 계산 (1도 ≈ 111km)
            lat_diff = (lat - node_lat) * 111000
            lon_diff = (lon - node_lon) * 111000 * 0.7  # 서울 위도 보정
            dist_m = (lat_diff ** 2 + lon_diff ** 2) ** 0.5
            
            # 50m 허용범위 내에서 가장 가까운 노드
            if dist_m < 50 and dist_m < min_dist:
                min_dist = dist_m
                nearest_node = node_id
        
        # 50m 내에 노드가 없으면 가장 가까운 노드 선택
        if nearest_node is None:
            for node_id, data in self.osm_graph.nodes(data=True):
                node_lat = data.get('y', 0)
                node_lon = data.get('x', 0)
                
                lat_diff = (lat - node_lat) * 111000
                lon_diff = (lon - node_lon) * 111000 * 0.7
                dist_m = (lat_diff ** 2 + lon_diff ** 2) ** 0.5
                
                if dist_m < min_dist:
                    min_dist = dist_m
                    nearest_node = node_id
        
        return nearest_node
    
    def find_access_options(self, origin_lat: float, origin_lon: float, 
                          max_time_sec: int = MAX_ACCESS_TIME_SEC,
                          top_n: int = 20) -> List[AccessResult]:
        """출발지에서 정류장들로의 액세스 옵션 탐색"""
        
        logger.info(f"액세스 옵션 탐색: ({origin_lat:.4f}, {origin_lon:.4f})")
        
        # 출발지 OSM 노드 찾기
        origin_node = self._find_nearest_osm_node(origin_lat, origin_lon)
        
        access_results = []
        
        # 1. 도보 액세스
        walk_results = self._dijkstra_access(
            origin_node, 'walk', max_time_sec
        )
        access_results.extend(walk_results)
        
        # 2. PM 액세스
        pm_results = self._dijkstra_access(
            origin_node, 'pm', max_time_sec,
            origin_coords=(origin_lat, origin_lon)
        )
        access_results.extend(pm_results)
        
        # 3. 따릉이 액세스 
        bike_results = self._bike_access(
            origin_lat, origin_lon, max_time_sec
        )
        access_results.extend(bike_results)
        
        # 시간순 정렬 후 상위 N개 반환
        access_results.sort(key=lambda x: x.access_time_sec)
        
        logger.info(f"액세스 옵션 {len(access_results[:top_n])}개 발견")
        
        return access_results[:top_n]
    
    def find_egress_options(self, dest_lat: float, dest_lon: float,
                          max_time_sec: int = MAX_EGRESS_TIME_SEC,
                          top_n: int = 20) -> List[AccessResult]:
        """정류장들에서 목적지로의 이그레스 옵션 탐색"""
        
        logger.info(f"이그레스 옵션 탐색: ({dest_lat:.4f}, {dest_lon:.4f})")
        
        # 목적지 OSM 노드 찾기
        dest_node = self._find_nearest_osm_node(dest_lat, dest_lon)
        
        egress_results = []
        
        # 1. 도보 이그레스  
        walk_results = self._dijkstra_egress(
            dest_node, 'walk', max_time_sec,
            dest_coords=(dest_lat, dest_lon)
        )
        egress_results.extend(walk_results)
        
        # 2. PM 이그레스
        pm_results = self._dijkstra_egress(
            dest_node, 'pm', max_time_sec,
            dest_coords=(dest_lat, dest_lon)
        )
        egress_results.extend(pm_results)
        
        # 3. 따릉이 이그레스
        bike_results = self._bike_egress(
            dest_lat, dest_lon, max_time_sec
        )
        egress_results.extend(bike_results)
        
        # 시간순 정렬 후 상위 N개 반환
        egress_results.sort(key=lambda x: x.access_time_sec)
        
        logger.info(f"이그레스 옵션 {len(egress_results[:top_n])}개 발견")
        
        return egress_results[:top_n]
    
    def _dijkstra_access(self, origin_node: int, mode: str, max_time_sec: int,
                        origin_coords: Optional[Tuple[float, float]] = None) -> List[AccessResult]:
        """다익스트라로 액세스 옵션 탐색"""
        
        # 모드별 시간 속성
        time_attr = f'travel_time_{mode}'
        
        # 다익스트라 실행
        distances = self._single_source_dijkstra(
            self.osm_graph, origin_node, max_time_sec, time_attr
        )
        
        results = []
        
        # 도달 가능한 정류장들 확인
        for stop_id, stop_osm_node in self.stop_osm_nodes.items():
            if stop_osm_node in distances:
                travel_time = distances[stop_osm_node]
                
                # PM 모드인 경우 대기시간 추가
                total_time = travel_time
                mode_details = {}
                
                if mode == 'pm' and origin_coords:
                    wait_time = self._get_pm_wait_time(origin_coords[0], origin_coords[1])
                    total_time += wait_time
                    mode_details['wait_time_sec'] = wait_time
                    mode_details['ride_time_sec'] = travel_time
                
                # 액세스 경로 구성 (단순화: 직선 경로)
                access_path = [origin_node, stop_osm_node]
                
                stop = self.stops[stop_id]
                result = AccessResult(
                    stop_id=stop_id,
                    stop_coords=(stop.stop_lat, stop.stop_lon),
                    access_time_sec=total_time,
                    mode=mode,
                    access_path=access_path,
                    mode_details=mode_details
                )
                
                results.append(result)
        
        return results
    
    def _dijkstra_egress(self, dest_node: int, mode: str, max_time_sec: int,
                        dest_coords: Optional[Tuple[float, float]] = None) -> List[AccessResult]:
        """다익스트라로 이그레스 옵션 탐색 (역방향)"""
        
        time_attr = f'travel_time_{mode}'
        results = []
        
        # 가까운 정류장들만 체크 (성능 최적화)
        dest_coords_array = np.array([[dest_coords[0], dest_coords[1]] if dest_coords else [0, 0]])
        if dest_coords:
            # 목적지 반경 2km 내 정류장들만 체크
            distances_deg, indices = self.stop_kdtree.query(
                dest_coords_array, 
                k=min(100, len(self.stop_ids_list))  # 최대 100개 정류장만
            )
            nearby_stops = [self.stop_ids_list[idx] for idx in indices[0] if distances_deg[0][0] < 0.018]  # 약 2km
        else:
            nearby_stops = list(self.stops.keys())[:50]  # 처음 50개만
        
        # logger.debug(f"이그레스 체크 대상 정류장: {len(nearby_stops)}개")
        # logger.debug(f"목적지 노드: {dest_node}")
        
        # 각 정류장에서 목적지까지 개별 다익스트라 실행
        valid_stops = 0
        tested_stops = 0
        for stop_id in nearby_stops:
            if stop_id not in self.stop_osm_nodes:
                continue
                
            tested_stops += 1
            stop_osm_node = self.stop_osm_nodes[stop_id]
            
            # 간단한 연결성 테스트 (첫 번째 정류장만) - 주석 처리
            # if tested_stops == 1:
            #     logger.debug(f"첫 번째 테스트 정류장: {stop_id}, OSM노드: {stop_osm_node}")
            #     simple_path = None
            #     try:
            #         simple_path = nx.shortest_path(self.osm_graph.to_undirected(), stop_osm_node, dest_node)
            #         logger.debug(f"NetworkX 경로 발견: {len(simple_path)}개 노드, 거리 확인 중...")
            #     except:
            #         logger.debug(f"NetworkX 경로 없음: {stop_osm_node} -> {dest_node}")
            
            # 정류장에서 목적지까지 다익스트라
            distances = self._single_source_dijkstra(
                self.osm_graph, stop_osm_node, max_time_sec, time_attr
            )
            
            # 첫 번째 정류장의 다익스트라 결과 자세히 확인 - 주석 처리
            # if tested_stops == 1:
            #     logger.debug(f"다익스트라 탐색 결과: {len(distances)}개 노드 도달")
            #     if dest_node in distances:
            #         logger.debug(f"목적지까지 시간: {distances[dest_node]:.1f}초")
            #     else:
            #         logger.debug(f"다익스트라로 목적지 미도달, 최대 시간 제한: {max_time_sec}초")
            
            # 목적지 노드에 도달 가능한지 확인
            if dest_node in distances:
                valid_stops += 1
                travel_time = distances[dest_node]
                
                # PM 모드인 경우 대기시간 추가
                total_time = travel_time
                mode_details = {}
                
                if mode == 'pm' and dest_coords:
                    # 정류장에서의 PM 대기시간
                    stop = self.stops[stop_id]
                    wait_time = self._get_pm_wait_time(stop.stop_lat, stop.stop_lon)
                    total_time += wait_time
                    mode_details['wait_time_sec'] = wait_time
                    mode_details['ride_time_sec'] = travel_time
                
                # 이그레스 경로 구성
                access_path = [stop_osm_node, dest_node]
                
                stop = self.stops[stop_id]
                result = AccessResult(
                    stop_id=stop_id,
                    stop_coords=(stop.stop_lat, stop.stop_lon),
                    access_time_sec=total_time,
                    mode=mode,
                    access_path=access_path,
                    mode_details=mode_details
                )
                
                results.append(result)
        
        # logger.debug(f"이그레스 {mode} 모드: {valid_stops}개 정류장에서 목적지 도달 가능, {len(results)}개 결과")
        return results
    
    def _bike_access(self, origin_lat: float, origin_lon: float, 
                    max_time_sec: int) -> List[AccessResult]:
        """따릉이 액세스 옵션 탐색"""
        
        results = []
        
        # 반경 내 대여소 찾기
        origin_coords = np.array([[origin_lat, origin_lon]])
        distances, indices = self.bike_kdtree.query(
            origin_coords, 
            k=min(10, len(self.bike_stations))
        )
        
        origin_node = self._find_nearest_osm_node(origin_lat, origin_lon)
        
        for dist_deg, station_idx in zip(distances[0], indices[0]):
            dist_m = dist_deg * 111000  # 대략적인 거리 변환
            
            if dist_m > BIKE_DOCK_RADIUS_M:
                continue
            
            station_id = self.bike_ids_list[station_idx]
            station = self.bike_stations[station_id]
            
            # 자전거 이용 가능 여부 확인
            if station.n_bikes <= 0:
                continue
            
            # 출발지 → 대여소 도보시간
            walk_to_dock_time = dist_m / SPEED_WALK_MPS
            
            # 대여소에서 정류장들까지 자전거 이동시간
            dock_distances = self._single_source_dijkstra(
                self.osm_graph, station.osm_node_id, 
                max_time_sec - walk_to_dock_time - BIKE_PICKUP_TIME_SEC,
                'travel_time_bike'
            )
            
            for stop_id, stop_osm_node in self.stop_osm_nodes.items():
                if stop_osm_node in dock_distances:
                    bike_time = dock_distances[stop_osm_node]
                    total_time = walk_to_dock_time + BIKE_PICKUP_TIME_SEC + bike_time
                    
                    if total_time <= max_time_sec:
                        stop = self.stops[stop_id]
                        result = AccessResult(
                            stop_id=stop_id,
                            stop_coords=(stop.stop_lat, stop.stop_lon),
                            access_time_sec=total_time,
                            mode='bike',
                            access_path=[origin_node, station.osm_node_id, stop_osm_node],
                            mode_details={
                                'dock_station_id': station_id,
                                'dock_name': station.station_name,
                                'walk_to_dock_sec': walk_to_dock_time,
                                'pickup_time_sec': BIKE_PICKUP_TIME_SEC,
                                'bike_ride_sec': bike_time
                            }
                        )
                        results.append(result)
        
        return results
    
    def _bike_egress(self, dest_lat: float, dest_lon: float,
                    max_time_sec: int) -> List[AccessResult]:
        """따릉이 이그레스 옵션 탐색"""
        
        results = []
        
        # 반경 내 대여소 찾기
        dest_coords = np.array([[dest_lat, dest_lon]])
        distances, indices = self.bike_kdtree.query(
            dest_coords,
            k=min(10, len(self.bike_stations))
        )
        
        dest_node = self._find_nearest_osm_node(dest_lat, dest_lon)
        
        for dist_deg, station_idx in zip(distances[0], indices[0]):
            dist_m = dist_deg * 111000
            
            if dist_m > BIKE_DOCK_RADIUS_M:
                continue
            
            station_id = self.bike_ids_list[station_idx]
            station = self.bike_stations[station_id]
            
            # 대여소 → 목적지 도보시간
            walk_from_dock_time = dist_m / SPEED_WALK_MPS
            
            # 정류장들에서 대여소까지 자전거 이동시간 (역방향)
            reverse_graph = self.osm_graph.reverse(copy=True)
            dock_distances = self._single_source_dijkstra(
                reverse_graph, station.osm_node_id,
                max_time_sec - walk_from_dock_time - BIKE_PICKUP_TIME_SEC,
                'travel_time_bike'
            )
            
            for stop_id, stop_osm_node in self.stop_osm_nodes.items():
                if stop_osm_node in dock_distances:
                    bike_time = dock_distances[stop_osm_node]
                    total_time = bike_time + BIKE_PICKUP_TIME_SEC + walk_from_dock_time
                    
                    if total_time <= max_time_sec:
                        stop = self.stops[stop_id]
                        result = AccessResult(
                            stop_id=stop_id,
                            stop_coords=(stop.stop_lat, stop.stop_lon),
                            access_time_sec=total_time,
                            mode='bike',
                            access_path=[stop_osm_node, station.osm_node_id, dest_node],
                            mode_details={
                                'dock_station_id': station_id,
                                'dock_name': station.station_name,
                                'bike_ride_sec': bike_time,
                                'pickup_time_sec': BIKE_PICKUP_TIME_SEC,
                                'walk_from_dock_sec': walk_from_dock_time
                            }
                        )
                        results.append(result)
        
        return results
    
    def _single_source_dijkstra(self, graph: nx.Graph, source: int, 
                               max_time: float, time_attr: str) -> Dict[int, float]:
        """단일 소스 다익스트라 알고리즘"""
        
        distances = {source: 0}
        heap = [(0, source)]
        visited = set()
        
        while heap:
            current_dist, current_node = heapq.heappop(heap)
            
            if current_node in visited:
                continue
            
            visited.add(current_node)
            
            if current_dist > max_time:
                continue
            
            # 이웃 노드들 탐색
            for neighbor in graph.neighbors(current_node):
                if neighbor in visited:
                    continue
                
                # MultiDiGraph에서 첫 번째 엣지의 데이터 가져오기
                if hasattr(graph, 'edges'):
                    # 멀티그래프인 경우 첫 번째 엣지 키 사용
                    edge_keys = list(graph[current_node][neighbor].keys())
                    if edge_keys:
                        edge_data = graph[current_node][neighbor][edge_keys[0]]
                    else:
                        edge_data = {}
                else:
                    edge_data = graph[current_node][neighbor]
                
                edge_time = edge_data.get(time_attr, float('inf'))
                
                # 첫 번째 엣지에서 디버깅 (한 번만) - 주석 처리
                # if len(visited) == 1 and time_attr == 'travel_time_walk':
                #     logger.debug(f"첫 엣지 확인: {current_node}->{neighbor}, time_attr={time_attr}, edge_time={edge_time}")
                #     logger.debug(f"엣지 데이터 키들: {list(edge_data.keys())}")
                
                new_dist = current_dist + edge_time
                
                if new_dist <= max_time and (neighbor not in distances or new_dist < distances[neighbor]):
                    distances[neighbor] = new_dist
                    heapq.heappush(heap, (new_dist, neighbor))
        
        return distances
    
    def _get_pm_wait_time(self, lat: float, lon: float) -> float:
        """PM 대기시간 계산 (밀도 기반)"""
        
        # 격자 ID 계산 (단순화)
        grid_row = int((lat - 37.46) / 0.0005)  # 50m 격자 크기
        grid_col = int((lon - 127.0) / 0.0007)
        grid_id = f"G_{grid_row:03d}_{grid_col:03d}"
        
        if grid_id in self.pm_density_map:
            density = self.pm_density_map[grid_id].get('adjusted_density', PM_DENSITY_THRESHOLD)
            if density > PM_DENSITY_THRESHOLD:
                # 밀도가 높을수록 대기시간 짧음
                wait_time = PM_BASE_WAIT_SEC / max(density, 0.1)
                return min(wait_time, 5 * 60)  # 최대 5분
        
        return PM_BASE_WAIT_SEC  # 기본 1분
    
    def _check_direct_routes(self, origin_lat: float, origin_lon: float,
                           dest_lat: float, dest_lon: float, dep_time: float) -> List[Dict[str, Any]]:
        """직접 경로 확인 (PM, 자전거, 도보)"""
        
        direct_journeys = []
        dep_time_min = dep_time * 60
        
        # 직선거리 계산
        import math
        dlat = math.radians(dest_lat - origin_lat)
        dlon = math.radians(dest_lon - origin_lon)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(origin_lat)) * math.cos(math.radians(dest_lat)) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        distance_m = 6371000 * c
        
        # 실제 도로 거리 (직선거리 × 1.3 근사)
        road_distance_m = distance_m * 1.3
        
        # 1. PM 직접 경로
        pm_wait_time = self._get_pm_wait_time(origin_lat, origin_lon)
        pm_ride_time = road_distance_m / SPEED_PM_MPS
        pm_total_time = (pm_wait_time + pm_ride_time) / 60  # 분 단위
        pm_cost = max(1000, int(road_distance_m / 100) * 100)  # 100m당 100원, 최소 1000원
        
        if pm_total_time <= 30:  # 30분 이내만
            direct_journeys.append({
                'total_time_min': pm_total_time,
                'final_arrival_time': dep_time_min + pm_total_time,
                'total_cost_won': pm_cost,
                'egress_stop': 'direct',
                'rounds': 0,
                'type': 'direct_pm',
                'segments': [{
                    'type': 'direct',
                    'mode': 'pm',
                    'duration_min': pm_total_time,
                    'distance_m': road_distance_m,
                    'cost_won': pm_cost,
                    'details': {
                        'wait_time_min': pm_wait_time / 60,
                        'ride_time_min': pm_ride_time / 60
                    }
                }]
            })
        
        # 2. 도보 직접 경로  
        walk_time = road_distance_m / SPEED_WALK_MPS / 60  # 분 단위
        
        if walk_time <= 25:  # 25분 이내만
            direct_journeys.append({
                'total_time_min': walk_time,
                'final_arrival_time': dep_time_min + walk_time,
                'total_cost_won': 0,
                'egress_stop': 'direct',
                'rounds': 0, 
                'type': 'direct_walk',
                'segments': [{
                    'type': 'direct',
                    'mode': 'walk',
                    'duration_min': walk_time,
                    'distance_m': road_distance_m,
                    'cost_won': 0
                }]
            })
        
        # 3. 자전거 직접 경로 (가까운 대여소 있는 경우)
        bike_coords = np.array([[origin_lat, origin_lon]])
        if hasattr(self, 'bike_kdtree'):
            distances, indices = self.bike_kdtree.query(bike_coords, k=1)
            
            if distances[0] * 111000 <= BIKE_DOCK_RADIUS_M:  # 200m 이내 대여소
                station_idx = indices[0]
                station_id = self.bike_ids_list[station_idx]
                station = self.bike_stations[station_id]
                
                # 출발지→대여소 도보
                walk_to_dock_time = (distances[0] * 111000) / SPEED_WALK_MPS / 60
                # 자전거 라이딩
                bike_ride_time = road_distance_m / SPEED_BIKE_MPS / 60
                # 대여소→목적지 도보 (목적지 근처 대여소 찾기)
                dest_coords = np.array([[dest_lat, dest_lon]])
                dest_distances, dest_indices = self.bike_kdtree.query(dest_coords, k=1)
                walk_from_dock_time = (dest_distances[0] * 111000) / SPEED_WALK_MPS / 60
                
                bike_total_time = walk_to_dock_time + (BIKE_PICKUP_TIME_SEC/60) + bike_ride_time + (BIKE_PICKUP_TIME_SEC/60) + walk_from_dock_time
                
                if bike_total_time <= 20:  # 20분 이내만
                    direct_journeys.append({
                        'total_time_min': bike_total_time,
                        'final_arrival_time': dep_time_min + bike_total_time,
                        'total_cost_won': 1000,
                        'egress_stop': 'direct',
                        'rounds': 0,
                        'type': 'direct_bike',
                        'segments': [{
                            'type': 'direct',
                            'mode': 'bike',
                            'duration_min': bike_total_time,
                            'distance_m': road_distance_m,
                            'cost_won': 1000,  # 따릉이 기본요금
                            'details': {
                                'origin_station': station.station_name,
                                'walk_to_dock_min': walk_to_dock_time,
                                'bike_ride_min': bike_ride_time,
                                'walk_from_dock_min': walk_from_dock_time
                            }
                        }]
                    })
        
        logger.info(f"직접 경로 {len(direct_journeys)}개 발견")
        return direct_journeys
    
    def _short_distance_routing(self, origin_lat: float, origin_lon: float,
                              dest_lat: float, dest_lon: float, 
                              dep_time: float, distance_km: float) -> List[Dict[str, Any]]:
        """근거리(<1.2km) 전용 최적 라우팅"""
        
        dep_time_min = dep_time * 60
        road_distance_m = distance_km * 1000 * 1.3  # 도로거리 = 직선거리 × 1.3
        
        short_journeys = []
        
        # 1. PM 직접 (최우선)
        pm_wait_time = self._get_pm_wait_time(origin_lat, origin_lon)
        pm_ride_time = road_distance_m / SPEED_PM_MPS
        pm_total_time = (pm_wait_time + pm_ride_time) / 60
        pm_cost = max(1500, int(road_distance_m / 100) * 150)
        
        short_journeys.append({
            'total_time_min': pm_total_time,
            'final_arrival_time': dep_time_min + pm_total_time,
            'total_cost_won': pm_cost,
            'egress_stop': 'direct',
            'rounds': 0,
            'type': 'short_pm',
            'priority': 1,
            'segments': [{
                'type': 'direct',
                'mode': 'pm',
                'description': f'PM 직접: {road_distance_m:.0f}m ({pm_total_time:.1f}분)',
                'duration_min': pm_total_time,
                'distance_m': road_distance_m,
                'cost_won': pm_cost,
                'details': {
                    'wait_time_min': pm_wait_time / 60,
                    'ride_time_min': pm_ride_time / 60
                }
            }]
        })
        
        # 2. 도보 직접
        walk_time = road_distance_m / SPEED_WALK_MPS / 60
        
        short_journeys.append({
            'total_time_min': walk_time,
            'final_arrival_time': dep_time_min + walk_time,
            'total_cost_won': 0,
            'egress_stop': 'direct',
            'rounds': 0,
            'type': 'short_walk',
            'priority': 2,
            'segments': [{
                'type': 'direct',
                'mode': 'walk',
                'description': f'도보 직접: {road_distance_m:.0f}m ({walk_time:.1f}분)',
                'duration_min': walk_time,
                'distance_m': road_distance_m,
                'cost_won': 0
            }]
        })
        
        # 3. 따릉이 직접 (가까운 대여소 있는 경우만)
        if hasattr(self, 'bike_kdtree'):
            bike_coords = np.array([[origin_lat, origin_lon]])
            distances, indices = self.bike_kdtree.query(bike_coords, k=1)
            
            if distances[0] * 111000 <= BIKE_DOCK_RADIUS_M:
                station_idx = indices[0]
                station = self.bike_stations[self.bike_ids_list[station_idx]]
                
                # 목적지 근처 대여소도 확인
                dest_coords = np.array([[dest_lat, dest_lon]])
                dest_distances, dest_indices = self.bike_kdtree.query(dest_coords, k=1)
                
                if dest_distances[0] * 111000 <= BIKE_DOCK_RADIUS_M:
                    dest_station = self.bike_stations[self.bike_ids_list[dest_indices[0]]]
                    
                    walk_to_dock = (distances[0] * 111000) / SPEED_WALK_MPS / 60
                    bike_ride = road_distance_m / SPEED_BIKE_MPS / 60  
                    walk_from_dock = (dest_distances[0] * 111000) / SPEED_WALK_MPS / 60
                    bike_total = walk_to_dock + 1 + bike_ride + 1 + walk_from_dock  # +1분씩 대여/반납
                    
                    short_journeys.append({
                        'total_time_min': bike_total,
                        'final_arrival_time': dep_time_min + bike_total,
                        'total_cost_won': 1000,
                        'egress_stop': 'direct',
                        'rounds': 0,
                        'type': 'short_bike',
                        'priority': 3,
                        'segments': [{
                            'type': 'direct',
                            'mode': 'bike',
                            'description': f'따릉이 직접: {station.station_name} → {dest_station.station_name} ({bike_total:.1f}분)',
                            'duration_min': bike_total,
                            'distance_m': road_distance_m,
                            'cost_won': 1000,
                            'details': {
                                'origin_station': station.station_name,
                                'dest_station': dest_station.station_name,
                                'walk_to_dock_min': walk_to_dock,
                                'bike_ride_min': bike_ride,
                                'walk_from_dock_min': walk_from_dock
                            }
                        }]
                    })
        
        # 4. 근거리 대중교통 (참고용만 - 직접 경로가 30% 이내면 제외)
        if distance_km > 0.5:  # 0.5km 이상일 때만 대중교통 고려
            logger.info("참고용 대중교통 경로도 확인...")
            try:
                access_options = self.find_access_options(origin_lat, origin_lon, top_n=5)
                egress_options = self.find_egress_options(dest_lat, dest_lon, top_n=5)
                
                if access_options and egress_options:
                    transit_journeys = self._run_raptor(access_options, egress_options, dep_time)[:2]  # 상위 2개만
                    
                    # 직접 경로와 비교: 50% 이내는 추가, 아니면 최단 1개는 참고용으로 강제 추가
                    min_direct_time = min(j['total_time_min'] for j in short_journeys)
                    added_transit_count = 0
                    
                    for i, journey in enumerate(transit_journeys):
                        # 50% 이내면 추가
                        if journey['total_time_min'] <= min_direct_time * 1.5:  # 50% 이내
                            journey['type'] = 'short_transit'
                            journey['priority'] = 9  # 낮은 우선순위
                            short_journeys.append(journey)
                            added_transit_count += 1
                        # 아무것도 안 추가됐으면 최단 1개는 참고용으로 강제 추가
                        elif i == 0 and added_transit_count == 0:
                            journey['type'] = 'short_transit_ref'
                            journey['priority'] = 10  # 더 낮은 우선순위 (참고용)
                            journey['description'] = f"[참고] {journey.get('description', '대중교통')}"
                            short_journeys.append(journey)
                            added_transit_count += 1
                            logger.info(f"참고용 대중교통({journey['total_time_min']:.1f}분) 추가 (직접경로 {min_direct_time:.1f}분 대비 느리지만 참고용)")
                        else:
                            logger.info(f"대중교통({journey['total_time_min']:.1f}분)이 직접경로({min_direct_time:.1f}분) 대비 너무 느려 제외")
            except:
                logger.info("대중교통 경로 확인 실패")
        
        # 중복 여정 제거
        deduplicated_short = self._deduplicate_journeys(short_journeys)
        
        # 우선순위 + 시간순 정렬
        deduplicated_short.sort(key=lambda x: (x.get('priority', 5), x['total_time_min']))
        
        logger.info(f"🎯 근거리 최적 경로 {len(short_journeys)}개 → 중복제거 후: {len(deduplicated_short)}개 (직접 경로 우선)")
        return deduplicated_short
    
    def route(self, origin_lat: float, origin_lon: float,
              dest_lat: float, dest_lon: float,
              dep_time: float = 8.0) -> List[Dict[str, Any]]:
        """완전한 멀티모달 RAPTOR 라우팅"""
        
        logger.info(f"=== OSM 다익스트라 멀티모달 RAPTOR ===")
        logger.info(f"출발: ({origin_lat:.4f}, {origin_lon:.4f})")
        logger.info(f"도착: ({dest_lat:.4f}, {dest_lon:.4f})")
        logger.info(f"출발시간: {dep_time:.1f}시")
        
        start_time = time.time()
        
        # 거리 기반 쇼트서킷 체크
        import math
        dlat = math.radians(dest_lat - origin_lat)
        dlon = math.radians(dest_lon - origin_lon)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(origin_lat)) * math.cos(math.radians(dest_lat)) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        distance_km = 6371 * c
        
        logger.info(f"직선거리: {distance_km:.2f}km")
        
        # 1.2km 이하 근거리 쇼트서킷
        if distance_km < 1.2:
            logger.info("🔥 근거리 감지 - 직접 경로 우선 모드")
            return self._short_distance_routing(origin_lat, origin_lon, dest_lat, dest_lon, dep_time, distance_km)
        
        # 1. 액세스 옵션 탐색 (출발지→정류장)
        logger.info("1단계: 액세스 옵션 탐색...")
        access_options = self.find_access_options(origin_lat, origin_lon)
        
        if not access_options:
            logger.warning("액세스 옵션이 없습니다.")
            return []
        
        # 2. 이그레스 옵션 탐색 (정류장→목적지)  
        logger.info("2단계: 이그레스 옵션 탐색...")
        egress_options = self.find_egress_options(dest_lat, dest_lon)
        
        if not egress_options:
            logger.warning("이그레스 옵션이 없습니다.")
            return []
        
        # 3. 직접 경로 확인 (PM, 자전거, 도보)
        logger.info("3단계: 직접 경로 확인...")
        direct_journeys = self._check_direct_routes(origin_lat, origin_lon, dest_lat, dest_lon, dep_time)
        
        # 4. RAPTOR 알고리즘 실행
        logger.info("4단계: RAPTOR 알고리즘 실행...")
        raptor_journeys = self._run_raptor(access_options, egress_options, dep_time)
        
        # 직접 경로 + RAPTOR 경로 결합
        all_journeys = direct_journeys + raptor_journeys
        
        # 중복 여정 제거
        deduplicated_journeys = self._deduplicate_journeys(all_journeys)
        
        elapsed = time.time() - start_time
        
        logger.info(f"=== 라우팅 완료 ({elapsed:.2f}초) ===")
        logger.info(f"액세스 옵션: {len(access_options)}개")
        logger.info(f"이그레스 옵션: {len(egress_options)}개")
        logger.info(f"발견된 여정: {len(all_journeys)}개 → 중복제거 후: {len(deduplicated_journeys)}개")
        
        return deduplicated_journeys
    
    def _run_raptor(self, access_options: List[AccessResult], 
                   egress_options: List[AccessResult], 
                   dep_time: float) -> List[Dict[str, Any]]:
        """RAPTOR 알고리즘 실행"""
        
        logger.info(f"RAPTOR 시작: {len(access_options)}개 액세스, {len(egress_options)}개 이그레스")
        
        # 시간 변환 (시 → 분)
        dep_time_min = dep_time * 60
        
        # RAPTOR 초기화
        n_stops = len(self.stops)
        max_rounds = 5  # 최대 5번 환승
        
        # tau[round][stop_idx] = 정류장에 round번째 라운드에서 도착하는 최소시간
        tau = [[float('inf')] * n_stops for _ in range(max_rounds + 1)]
        
        # parent[round][stop_idx] = 부모 정보 (경로 재구성용)
        parent = [{} for _ in range(max_rounds + 1)]
        
        # 1라운드: 액세스로 도달 가능한 정류장들 초기화
        marked_stops = set()
        
        for access in access_options:
            if access.stop_id in self.stop_index_map:
                stop_idx = self.stop_index_map[access.stop_id]
                arrival_time = dep_time_min + (access.access_time_sec / 60)
                
                if arrival_time < tau[1][stop_idx]:
                    tau[1][stop_idx] = arrival_time
                    # 액세스 상세 정보 생성
                    access_info = {
                        'type': 'access',
                        'mode': access.mode,
                        'access_time': access.access_time_sec / 60,
                        'details': access.mode_details
                    }
                    
                    # 모드별 상세 정보 추가
                    if access.mode == 'bike' and hasattr(access, 'station_name'):
                        access_info['station_info'] = {
                            'stationName': getattr(access, 'station_name', '정보없음'),
                            'stationId': getattr(access, 'station_id', ''),
                            'lat': getattr(access, 'station_lat', 0),
                            'lon': getattr(access, 'station_lon', 0)
                        }
                    elif access.mode == 'pm':
                        # PM 격자 정보 (mode_details에서 추출)
                        if access.mode_details and 'density' in str(access.mode_details):
                            access_info['grid_info'] = access.mode_details
                        else:
                            access_info['grid_info'] = {'density': 10.0}  # 기본값
                    
                    parent[1][stop_idx] = access_info
                    marked_stops.add(stop_idx)
        
        logger.info(f"1라운드 초기화: {len(marked_stops)}개 정류장 마킹")
        
        # RAPTOR 라운드 실행 (PART2_NEW 검증된 로직 적용)
        for k in range(2, max_rounds + 1):  # 2라운드부터 시작 (1라운드는 액세스 초기화)
            logger.info(f"RAPTOR {k}라운드 시작...")
            
            if not marked_stops:
                logger.info(f"마킹된 정류장이 없어 {k}라운드 종료")
                break
            
            # PART2_NEW 스타일 route-based propagation
            new_marked_stops = self._route_based_propagation_osm(k, tau, parent, marked_stops)
            
            # 환승 처리 (PART2_NEW 스타일)
            transfer_marked = self._transfer_propagation_osm(k, tau, parent, new_marked_stops)
            new_marked_stops.update(transfer_marked)
            
            marked_stops = new_marked_stops
            logger.info(f"{k}라운드 완료: {len(marked_stops)}개 정류장 마킹")
        
        # 여정 구성
        journeys = self._build_journeys(tau, parent, egress_options, max_rounds, dep_time_min)
        
        return journeys
    
    def _route_based_propagation_osm(self, k: int, tau: List[List[float]], 
                                   parent: List[Dict], marked_stops: set) -> set:
        """대중교통 노선 기반 전파 - PART2_NEW 검증된 로직 적용"""
        marked = set()
        routes_to_scan = set()
        
        # 1단계: k-1 라운드에 도달한 정류장에서 탑승 가능한 노선들 수집
        for stop_idx in range(len(tau[k-1])):
            if tau[k-1][stop_idx] < float('inf'):
                stop_id = self.index_to_stop.get(stop_idx)
                if stop_id:
                    # 이 정류장을 지나는 노선들 추가  
                    for route_id in self.timetables.keys():
                        if route_id in self.route_stops:
                            stop_sequence = self.route_stops[route_id]
                            if stop_id in stop_sequence:
                                routes_to_scan.add(route_id)
        
        # 2단계: 각 노선별로 처리
        for route_id in routes_to_scan:
            timetable = self.timetables.get(route_id)
            stop_sequence = self.route_stops.get(route_id, [])
            
            if not timetable or len(stop_sequence) < 2:
                continue
            
            # 시간표가 정상적인 구조인지 확인: timetable[stop_seq_idx][trip_idx]
            if not isinstance(timetable[0], list):
                continue
            
            # 이 노선의 각 trip별로 처리
            n_trips = len(timetable[0]) if timetable else 0
            
            for trip_idx in range(n_trips):
                # 이 trip에서 탑승할 정류장 찾기
                board_stop_idx = -1
                board_time = float('inf')
                
                for i, stop_id in enumerate(stop_sequence):
                    if stop_id not in self.stop_index_map:
                        continue
                    
                    stop_idx = self.stop_index_map[stop_id]
                    arrival_time = tau[k-1][stop_idx]
                    
                    if arrival_time < float('inf') and i < len(timetable):
                        if trip_idx < len(timetable[i]):
                            dep_time = timetable[i][trip_idx]
                            
                            # 도착시간 이후에 출발하는 경우만 탑승 가능
                            if dep_time >= arrival_time:
                                board_stop_idx = i
                                board_time = dep_time
                                break
                
                # 탑승 가능하면 이후 정류장들 업데이트
                if board_stop_idx >= 0:
                    board_stop_id = stop_sequence[board_stop_idx]
                    
                    # 하차 가능한 정류장들 업데이트
                    for j in range(board_stop_idx + 1, len(stop_sequence)):
                        alight_stop_id = stop_sequence[j]
                        if alight_stop_id not in self.stop_index_map:
                            continue
                        
                        alight_stop_idx = self.stop_index_map[alight_stop_id]
                        
                        # 같은 trip의 도착 시간
                        if j < len(timetable) and trip_idx < len(timetable[j]):
                            alight_time = timetable[j][trip_idx]
                            
                            # 시간 유효성 검사: 도착시간이 탑승시간보다 늦어야 함
                            if alight_time <= board_time:
                                continue  # 잘못된 데이터 건너뛰기 (같은 시간도 제외)
                            
                            # 개선된 경우만 업데이트
                            if alight_time < tau[k][alight_stop_idx]:
                                tau[k][alight_stop_idx] = alight_time
                                
                                parent[k][alight_stop_idx] = {
                                    'type': 'route',
                                    'route_id': route_id,
                                    'board_stop': board_stop_id,
                                    'alight_stop': alight_stop_id,
                                    'board_time': board_time,
                                    'alight_time': alight_time,
                                    'trip_idx': trip_idx
                                }
                                
                                marked.add(alight_stop_idx)
        
        return marked
    
    def _transfer_propagation_osm(self, k: int, tau: List[List[float]], 
                                parent: List[Dict], new_marked_stops: set) -> set:
        """환승 전파 - PART2_NEW 검증된 로직 적용"""
        transfer_marked = set()
        
        for stop_idx in new_marked_stops:
            if tau[k][stop_idx] == float('inf'):
                continue
                
            stop_id = self.index_to_stop.get(stop_idx)
            if not stop_id or stop_id not in self.transfers:
                continue
            
            current_time = tau[k][stop_idx]
            
            # 기존 도보 환승
            for transfer_stop_id, transfer_time in self.transfers[stop_id]:
                if transfer_stop_id not in self.stop_index_map:
                    continue
                
                # 가짜 환승 필터링
                from_name = self.stop_names.get(stop_id, stop_id)
                to_name = self.stop_names.get(transfer_stop_id, transfer_stop_id)
                
                # 같은 이름 정류장 간 0분 환승 제거
                if from_name == to_name and transfer_time == 0:
                    continue
                
                # 1분 미만 환승도 제거 (실제 의미 없음)
                if transfer_time < 1.0:
                    continue
                
                transfer_idx = self.stop_index_map[transfer_stop_id]
                arrival_time = current_time + transfer_time
                
                if arrival_time < tau[k][transfer_idx]:
                    tau[k][transfer_idx] = arrival_time
                    
                    parent[k][transfer_idx] = {
                        'type': 'transfer',
                        'transfer_type': 'walk',
                        'from_stop': stop_id,
                        'to_stop': transfer_stop_id,
                        'transfer_time': transfer_time
                    }
                    
                    transfer_marked.add(transfer_idx)
        
        return transfer_marked
    
    def _find_best_departure(self, route_id: str, stop_id: str, passenger_arrival: float):
        """승객 도착 시간 이후 가장 가까운 출발시간 찾기"""
        
        if route_id not in self.timetables:
            return None
        
        route_stops = self.route_stops[route_id]
        if stop_id not in route_stops:
            return None
        
        stop_seq_idx = route_stops.index(stop_id)
        route_timetable = self.timetables[route_id]
        
        best_trip = None
        best_departure = float('inf')
        
        for trip_idx, trip_times in enumerate(route_timetable):
            if stop_seq_idx >= len(trip_times):
                continue
            
            departure_time = trip_times[stop_seq_idx]
            
            if departure_time >= passenger_arrival and departure_time < best_departure:
                best_departure = departure_time
                best_trip = (trip_idx, departure_time)
        
        return best_trip
    
    def _get_arrival_time(self, route_id: str, trip_idx: int, alight_stop_id: str, board_time: float):
        """특정 trip에서 하차 정류장 도착시간 계산"""
        
        route_stops = self.route_stops[route_id]
        if alight_stop_id not in route_stops:
            return None
        
        stop_seq_idx = route_stops.index(alight_stop_id)
        route_timetable = self.timetables[route_id]
        
        if trip_idx >= len(route_timetable) or stop_seq_idx >= len(route_timetable[trip_idx]):
            return None
        
        return route_timetable[trip_idx][stop_seq_idx]
    
    def _build_journeys(self, tau, parent, egress_options: List[AccessResult], max_rounds: int, dep_time_min: float):
        """RAPTOR 결과에서 상세 여정 구성"""
        
        journeys = []
        
        # 이그레스 정류장들에서 최종 도달 시간 확인
        for egress in egress_options:
            if egress.stop_id not in self.stop_index_map:
                continue
            
            stop_idx = self.stop_index_map[egress.stop_id]
            
            # 각 라운드에서 이 정류장에 도달한 최선의 경로 찾기
            for k in range(1, max_rounds + 1):
                if tau[k][stop_idx] == float('inf'):
                    continue
                
                # 총 여행시간 = (도착시간 - 출발시간) + 이그레스 시간  
                raptor_arrival_time = tau[k][stop_idx]
                total_travel_time = (raptor_arrival_time - dep_time_min) + (egress.access_time_sec / 60)
                
                # 상세 경로 재구성
                detailed_segments = self._reconstruct_detailed_path(parent, k, stop_idx, egress, dep_time_min)
                
                if detailed_segments:
                    journey = {
                        'total_time_min': total_travel_time,
                        'total_cost_won': self._calculate_total_cost(detailed_segments),
                        'final_arrival_time': raptor_arrival_time + (egress.access_time_sec / 60),
                        'rounds': k,
                        'segments': detailed_segments
                    }
                    
                    journeys.append(journey)
        
        # 시간순 정렬
        journeys.sort(key=lambda x: x['total_time_min'])
        
        return journeys[:5]  # 상위 5개만 반환
    
    def _reconstruct_detailed_path(self, parent, round_k: int, stop_idx: int, egress: AccessResult, dep_time_min: float):
        """상세한 경로 재구성 - 시간, 역명, 노선 정보 포함"""
        
        segments = []
        current_round = round_k
        current_stop_idx = stop_idx
        
        # 역순으로 경로 추적
        path = []
        while current_round > 0 and current_stop_idx in parent[current_round]:
            parent_info = parent[current_round][current_stop_idx]
            path.append((current_round, current_stop_idx, parent_info))
            
            if parent_info['type'] == 'access':
                break
            elif parent_info['type'] == 'route':
                # 대중교통 구간
                current_stop_idx = self.stop_index_map[parent_info['board_stop']]
                current_round -= 1
            elif parent_info['type'] == 'transfer':
                # 환승 구간  
                current_stop_idx = self.stop_index_map[parent_info['from_stop']]
                # 라운드는 그대로
        
        # 경로를 정방향으로 변환하고 세그먼트 생성
        path.reverse()
        
        for i, (round_num, stop_idx, parent_info) in enumerate(path):
            if parent_info['type'] == 'access':
                # 액세스 구간 - 상세 정보 포함
                access_time = self._clean_segment_time(parent_info['access_time'])
                mode = parent_info['mode']
                
                # 모드별 상세 정보 생성
                if mode == 'bike':
                    # 따릉이 정보
                    bike_station_info = parent_info.get('station_info', {})
                    station_name = bike_station_info.get('stationName', '정보없음')
                    description = f"🚲 따릉이: 출발지 → {station_name} 대여소 ({access_time:.1f}분)"
                    cost = 1000  # 따릉이 기본 요금
                elif mode == 'pm':
                    # PM(킥보드) 정보
                    grid_info = parent_info.get('grid_info', {})
                    density = grid_info.get('density', 0)
                    description = f"🛴 PM(킥보드): 출발지에서 탑승 ({access_time:.1f}분, 밀도: {density:.1f}대/km²)"
                    cost = 1500  # PM 기본 요금
                else:  # walk
                    description = f"🚶 도보: 출발지 → 정류장 ({access_time:.1f}분)"
                    cost = 0
                
                segments.append({
                    'type': 'access',
                    'mode': mode,
                    'description': description,
                    'duration_min': access_time,
                    'departure_time': self._format_time(dep_time_min),
                    'arrival_time': self._format_time(dep_time_min + access_time),
                    'cost_won': cost,
                    'detail_info': parent_info.get('station_info', parent_info.get('grid_info', {}))
                })
            
            elif parent_info['type'] == 'route':
                # 대중교통 구간
                route_id = parent_info['route_id']
                route = self.routes.get(route_id)
                
                board_stop_name = self.stop_names.get(parent_info['board_stop'], parent_info['board_stop'])
                alight_stop_name = self.stop_names.get(parent_info['alight_stop'], parent_info['alight_stop'])
                
                # 노선 타입 확인
                route_name = f"노선 {route_id}"
                if route and hasattr(route, 'route_type'):
                    if route.route_type == 1:  # 지하철
                        route_name = f"지하철 {route.route_short_name or route_id}"
                    else:  # 버스
                        route_name = f"버스 {route.route_short_name or route_id}"
                
                duration = parent_info['alight_time'] - parent_info['board_time']
                duration = self._clean_segment_time(duration)  # 시간 정리 적용
                
                cost = 1370 if route and route.route_type == 1 else 1380  # 지하철/버스 요금
                
                segments.append({
                    'type': 'transit',
                    'mode': 'subway' if route and route.route_type == 1 else 'bus',
                    'route_name': route_name,
                    'description': f"{route_name}: {board_stop_name} → {alight_stop_name}",
                    'duration_min': duration,
                    'departure_time': self._format_time(parent_info['board_time']),
                    'arrival_time': self._format_time(parent_info['alight_time']),
                    'from_stop': board_stop_name,
                    'to_stop': alight_stop_name,
                    'cost_won': cost
                })
            
            elif parent_info['type'] == 'transfer':
                # 환승 구간
                from_stop_name = self.stop_names.get(parent_info['from_stop'], parent_info['from_stop'])
                to_stop_name = self.stop_names.get(parent_info['to_stop'], parent_info['to_stop'])
                transfer_time = self._clean_segment_time(parent_info['transfer_time'])
                
                segments.append({
                    'type': 'transfer',
                    'mode': 'walk',
                    'description': f"도보 환승: {from_stop_name} → {to_stop_name}",
                    'duration_min': transfer_time,
                    'from_stop': from_stop_name,
                    'to_stop': to_stop_name,
                    'cost_won': 0
                })
        
        # 이그레스 구간 추가 - 상세 정보 포함
        if egress:
            egress_time = self._clean_segment_time(egress.access_time_sec / 60)
            mode = egress.mode
            
            # 모드별 상세 정보 생성
            if mode == 'bike':
                # 따릉이 반납 정보
                bike_station_info = getattr(egress, 'station_info', {})
                station_name = bike_station_info.get('stationName', '정보없음') if bike_station_info else '정보없음'
                description = f"🚲 따릉이: {station_name} 대여소 → 도착지 ({egress_time:.1f}분, 반납)"
                cost = 0  # 이그레스는 추가 요금 없음
            elif mode == 'pm':
                # PM(킥보드) 하차 정보
                grid_info = getattr(egress, 'grid_info', {})
                density = grid_info.get('density', 0) if grid_info else 0
                description = f"🛴 PM(킥보드): 정류장 → 도착지 ({egress_time:.1f}분, 밀도: {density:.1f}대/km²)"
                cost = 0  # 이그레스는 추가 요금 없음
            else:  # walk
                description = f"🚶 도보: 정류장 → 도착지 ({egress_time:.1f}분)"
                cost = 0
            
            segments.append({
                'type': 'egress',
                'mode': mode,
                'description': description,
                'duration_min': egress_time,
                'cost_won': cost,
                'detail_info': getattr(egress, 'station_info', getattr(egress, 'grid_info', {}))
            })
        
        return segments
    
    def _calculate_total_cost(self, segments):
        """세그먼트들의 총 비용 계산"""
        return sum(seg.get('cost_won', 0) for seg in segments)
    
    def _format_time(self, minutes):
        """분을 HH:MM 형식으로 변환"""
        hours = int(minutes // 60)
        mins = int(minutes % 60)
        return f"{hours:02d}:{mins:02d}"
    
    def _clean_segment_time(self, duration_min: float) -> float:
        """세그먼트 시간 정리: 최소값 적용 및 반올림"""
        if duration_min <= 0:
            return MIN_SEGMENT_TIME_MIN
        return round(max(duration_min, MIN_SEGMENT_TIME_MIN), 1)
    
    def _create_journey_signature(self, journey: Dict[str, Any]) -> str:
        """여정 시그니처 생성 (중복 제거용)"""
        segments = journey.get('segments', [])
        
        # 대중교통 세그먼트만 추출하여 시그니처 생성
        transit_parts = []
        for seg in segments:
            if seg.get('type') == 'transit':
                route_name = seg.get('route_name', '')
                from_stop = seg.get('from_stop', '')
                to_stop = seg.get('to_stop', '')
                departure_time = seg.get('departure_time', '')
                arrival_time = seg.get('arrival_time', '')
                transit_parts.append(f"{route_name}|{from_stop}|{to_stop}|{departure_time}|{arrival_time}")
        
        # 대중교통이 없으면 여정 타입으로 구분
        if not transit_parts:
            return journey.get('type', 'direct')
        
        return "||".join(transit_parts)
    
    def _deduplicate_journeys(self, journeys: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """중복 여정 제거: 동일 경로에서 도보 거리가 짧은 것만 유지"""
        
        signature_groups = {}
        
        for journey in journeys:
            signature = self._create_journey_signature(journey)
            
            if signature not in signature_groups:
                signature_groups[signature] = []
            signature_groups[signature].append(journey)
        
        # 각 그룹에서 최적 여정 선택
        deduplicated = []
        for signature, group in signature_groups.items():
            if len(group) == 1:
                deduplicated.extend(group)
            else:
                # 동일 시그니처 그룹에서 도보 거리가 가장 짧은 것 선택
                best_journey = min(group, key=lambda j: self._calculate_total_walk_distance(j))
                deduplicated.append(best_journey)
        
        return deduplicated
    
    def _calculate_total_walk_distance(self, journey: Dict[str, Any]) -> float:
        """여정의 총 도보 거리 계산"""
        total_walk = 0.0
        segments = journey.get('segments', [])
        
        for seg in segments:
            if seg.get('mode') == 'walk' or seg.get('type') == 'transfer':
                # 거리가 있으면 사용, 없으면 시간으로 추정
                distance_m = seg.get('distance_m', 0)
                if distance_m == 0:
                    duration_min = seg.get('duration_min', 0)
                    distance_m = duration_min * 60 * SPEED_WALK_MPS  # 시간 → 거리 변환
                total_walk += distance_m
        
        return total_walk
    
    def _reconstruct_path(self, parent, round_k: int, stop_idx: int, egress: AccessResult):
        """경로 재구성"""
        
        segments = []
        current_round = round_k
        current_stop_idx = stop_idx
        
        # 역순으로 경로 추적
        path = []
        while current_round > 0:
            if current_stop_idx not in parent[current_round]:
                break
            
            parent_info = parent[current_round][current_stop_idx]
            path.append((current_round, current_stop_idx, parent_info))
            
            if parent_info['type'] == 'access':
                break
            elif parent_info['type'] == 'transit':
                # 탑승 정류장으로 이동
                board_stop_id = parent_info['board_stop_id']
                current_stop_idx = self.stop_index_map[board_stop_id]
                current_round -= 1
            elif parent_info['type'] == 'transfer':
                # 환승 출발 정류장으로 이동  
                from_stop_id = parent_info['from_stop_id']
                current_stop_idx = self.stop_index_map[from_stop_id]
        
        # 정방향으로 세그먼트 구성
        path.reverse()
        
        for i, (round_num, stop_idx, parent_info) in enumerate(path):
            if parent_info['type'] == 'access':
                segments.append({
                    'type': 'access',
                    'mode': parent_info['mode'],
                    'duration_min': parent_info['access_time'],
                    'details': parent_info.get('details', {})
                })
            elif parent_info['type'] == 'transit':
                route_id = parent_info['route_id']
                route_name = self.routes[route_id].route_short_name if route_id in self.routes else route_id
                
                segments.append({
                    'type': 'transit',
                    'route_name': route_name,
                    'board_stop': self.stop_names.get(parent_info['board_stop_id'], parent_info['board_stop_id']),
                    'alight_stop': self.stop_names.get(self.index_to_stop[stop_idx], ''),
                    'board_time': parent_info['board_time'],
                    'alight_time': parent_info['alight_time'],
                    'board_time_str': self._format_time(parent_info['board_time']),
                    'alight_time_str': self._format_time(parent_info['alight_time']),
                    'duration_min': parent_info['alight_time'] - parent_info['board_time']
                })
            elif parent_info['type'] == 'transfer':
                segments.append({
                    'type': 'transfer',
                    'duration_min': parent_info['walk_time'],
                    'from_stop': self.stop_names.get(parent_info['from_stop_id'], parent_info['from_stop_id']),
                    'to_stop': self.stop_names.get(self.index_to_stop[stop_idx], '')
                })
        
        # 이그레스 추가
        segments.append({
            'type': 'egress',
            'mode': egress.mode,
            'duration_min': egress.access_time_sec / 60,
            'details': egress.mode_details
        })
        
        return segments
    
    def _format_time(self, minutes: float) -> str:
        """분을 시:분 형식으로 변환"""
        hours = int(minutes // 60)
        mins = int(minutes % 60)
        return f"{hours:02d}:{mins:02d}"

# ============================================================================
# 테스트 함수
# ============================================================================

def test_osm_dijkstra():
    """OSM 다익스트라 RAPTOR 테스트"""
    
    try:
        # 라우터 초기화
        router = OSMDijkstraRAPTOR()
        
        # 테스트 케이스 - 근거리, 중거리, 장거리, 주택가 포함
        test_cases = [
            {
                'name': '강남역 - 역삼역 (근거리)',
                'origin': (37.4979, 127.0276),  # 강남역
                'dest': (37.5007, 127.0363),    # 역삼역
                'time': 8.0
            },
            {
                'name': '개포동 주공아파트 - 대치동 은마아파트 (중거리 주택가)',
                'origin': (37.4813, 127.0701),  # 개포동 주공아파트
                'dest': (37.4935, 127.0591),    # 대치동 은마아파트
                'time': 8.0
            },
            {
                'name': '세곡동 래미안 - 도곡동 타워팰리스 (장거리 주택가)',
                'origin': (37.4680, 127.1035),  # 세곡동 래미안
                'dest': (37.4803, 127.0395),    # 도곡동 타워팰리스
                'time': 8.0
            },
            {
                'name': '일원동 현대아파트 - 삼성동 코엑스몰 (장거리)',
                'origin': (37.4847, 127.0828),  # 일원동 현대아파트
                'dest': (37.5115, 127.0595),    # 삼성동 코엑스몰
                'time': 8.0
            }
        ]
        
        # PM 직접 경로 테스트 추가
        print("\n=== PM 직접 경로 가능성 확인 ===")
        origin_lat, origin_lon = 37.4979, 127.0276
        dest_lat, dest_lon = 37.5007, 127.0363
        
        # PM 직접 시간 계산
        import math
        dlat = math.radians(dest_lat - origin_lat)
        dlon = math.radians(dest_lon - origin_lon)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(origin_lat)) * math.cos(math.radians(dest_lat)) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        distance_m = 6371000 * c
        
        pm_wait_time = router._get_pm_wait_time(origin_lat, origin_lon)
        pm_ride_time = distance_m / SPEED_PM_MPS
        pm_total_time = pm_wait_time + pm_ride_time
        
        print(f"직선거리: {distance_m:.0f}m")
        print(f"PM 대기시간: {pm_wait_time:.1f}초")
        print(f"PM 주행시간: {pm_ride_time:.1f}초") 
        print(f"PM 총 시간: {pm_total_time/60:.1f}분")
        print("→ 이 시간이면 PM으로 바로 갈 수 있어야 함!")
        
        for case in test_cases:
            print(f"\n=== {case['name']} ===")
            
            journeys = router.route(
                case['origin'][0], case['origin'][1],
                case['dest'][0], case['dest'][1],
                case['time']
            )
            
            if not journeys:
                print("❌ 경로를 찾을 수 없습니다.")
                continue
            
            print(f"✅ {len(journeys)}개 여정 발견:")
            
            for i, journey in enumerate(journeys, 1):
                total_time = journey.get('total_time_min', 0)
                total_cost = journey.get('total_cost_won', 0)
                rounds = journey.get('rounds', 0)
                
                if journey.get('final_arrival_time'):
                    arrival_str = router._format_time(journey['final_arrival_time'])
                    print(f"\n💡 여정 {i}: {total_time:.1f}분, {total_cost}원 (→{arrival_str} 도착, {rounds}라운드)")
                else:
                    print(f"\n💡 여정 {i}: {total_time:.1f}분, {total_cost}원 ({rounds}라운드)")
                
                for j, segment in enumerate(journey.get('segments', []), 1):
                    mode = segment.get('mode', 'unknown')
                    seg_type = segment.get('type', 'unknown')
                    duration = segment.get('duration_min', 0)
                    cost = segment.get('cost_won', 0)
                    
                    # 아이콘 설정
                    if mode == 'pm':
                        icon = '🛴'
                    elif mode == 'walk':
                        icon = '🚶'
                    elif mode == 'bike':
                        icon = '🚲'
                    elif mode == 'subway':
                        icon = '🚇'
                    elif mode == 'bus':
                        icon = '🚌'
                    else:
                        icon = '🔄'
                    
                    # 세그먼트별 정보 출력
                    if seg_type == 'direct':
                        # 직접 경로
                        description = segment.get('description', f"{mode} 직접")
                        print(f"   {j}. {icon} {description}")
                        
                        if segment.get('details'):
                            details = segment['details']
                            if mode == 'pm':
                                wait_time = details.get('wait_time_min', 0)
                                ride_time = details.get('ride_time_min', 0)
                                print(f"      💰 비용: {cost}원 | ⏱️ 대기: {wait_time:.1f}분, 주행: {ride_time:.1f}분")
                            elif mode == 'bike':
                                print(f"      💰 비용: {cost}원 | 🚲 따릉이 대여/반납 포함")
                            else:
                                print(f"      💰 비용: {cost}원")
                    
                    elif seg_type == 'access':
                        # 액세스
                        departure_time = segment.get('departure_time', '08:00')
                        arrival_time = segment.get('arrival_time', '08:00')
                        print(f"   {j}. 🚶 액세스 ({mode}): {departure_time} → {arrival_time} ({duration:.1f}분)")
                    
                    elif seg_type == 'transit':
                        # 대중교통
                        route_name = segment.get('route_name', f'{mode} 노선')
                        departure_time = segment.get('departure_time', '08:00')
                        arrival_time = segment.get('arrival_time', '08:00')
                        from_stop = segment.get('from_stop', '출발역')
                        to_stop = segment.get('to_stop', '도착역')
                        
                        print(f"   {j}. {icon} {route_name}")
                        print(f"      📍 {from_stop} ({departure_time}) → {to_stop} ({arrival_time})")
                        print(f"      💰 비용: {cost}원 | ⏱️ 소요시간: {duration:.1f}분")
                    
                    elif seg_type == 'transfer':
                        # 환승
                        from_stop = segment.get('from_stop', '환승역1')
                        to_stop = segment.get('to_stop', '환승역2')
                        print(f"   {j}. 🚶 도보 환승: {from_stop} → {to_stop} ({duration:.1f}분)")
                    
                    elif seg_type == 'egress':
                        # 이그레스
                        print(f"   {j}. 🚶 이그레스 ({mode}): {duration:.1f}분")
                    
                    else:
                        # 기타
                        description = segment.get('description', f"{mode}: {duration:.1f}분")
                        print(f"   {j}. {icon} {description}")
                        if cost > 0:
                            print(f"      💰 비용: {cost}원")
    
    except Exception as e:
        logger.error(f"테스트 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_osm_dijkstra()