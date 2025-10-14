#!/usr/bin/env python3
"""
PART3: 이중 격자 파동-확산 멀티모달 RAPTOR v2.0
- 출발지: 50m 정밀격자 → PM 대기시간 정확 계산
- 확산: 300m 거대격자 → 빠른 탐색 (최대 4.2km)
- 도착지: 정류장별 PM 접근성 분석 → 이중 라스트마일
- PM 연속주행: 대기는 처음만, 이후 순수 주행
"""

import pickle
import json
import math
import time
import logging
import numpy as np
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
# 데이터 구조
# ============================================================================

@dataclass
class FineGrid:
    """50m 정밀 격자 (출발지/도착지 밀도 계산용)"""
    grid_id: str
    lat: float
    lon: float
    density: float
    availability_score: float
    grid_type: str

@dataclass
class CoarseGrid:
    """300m 거대 격자 (빠른 확산용)"""
    grid_id: str
    center_lat: float
    center_lon: float
    fine_grids: List[str] = field(default_factory=list)  # 포함된 50m 격자들

@dataclass
class PMRoute:
    """PM 전용 경로"""
    origin: Tuple[float, float]
    destination: Tuple[float, float]
    total_distance_m: float
    total_ride_time_min: float
    initial_wait_time_min: float
    total_cost_won: int
    grid_path: List[str]

@dataclass
class PMEgressOption:
    """정류장→도착지 PM 옵션"""
    station_id: str
    station_coords: Tuple[float, float]
    pm_wait_time_min: float
    pm_ride_time_min: float
    pm_cost_won: int
    dest_distance_m: float
    viability: str  # 'excellent', 'good', 'poor'

@dataclass
class MultimodeJourney:
    """완전한 멀티모달 여정"""
    segments: List[Dict[str, Any]]
    total_time_min: float
    total_cost_won: int
    n_transfers: int
    total_walk_m: float
    journey_type: str  # 'pm_only', 'transit_only', 'multimodal'
    pm_portion_min: float = 0.0
    
    def get_pareto_vector(self) -> Tuple[float, int, float, int]:
        """파레토 최적화용 벡터"""
        return (self.total_time_min, self.n_transfers, self.total_walk_m, self.total_cost_won)

# ============================================================================
# 메인 클래스
# ============================================================================

class DualGridWaveExpansionRAPTOR:
    """이중 격자 파동-확산 멀티모달 RAPTOR"""
    
    def __init__(self, 
                 raptor_data_path: str = "gangnam_raptor_data/raptor_data.pkl",
                 pm_density_path: str = "grid_pm_data/pm_density_map.json",
                 bike_stations_path: str = "bike_stations_simple/ttareungee_stations.csv"):
        
        logger.info("=== 이중 격자 파동-확산 RAPTOR v2.0 초기화 ===")
        
        # 핵심 파라미터
        self.params = {
            'PM_SPEED_KMH': 18.0,       # PM 속도 (시속)
            'PM_SPEED_MPS': 5.0,        # PM 속도 (m/s)
            'PM_SPEED_MPM': 300.0,      # PM 속도 (m/분)
            'WALK_SPEED_MPS': 1.4,      # 도보 속도 (m/s)
            'WALK_SPEED_MPM': 84.0,     # 도보 속도 (m/분)
            
            'FINE_GRID_SIZE_M': 50,     # 정밀 격자 크기
            'COARSE_GRID_SIZE_M': 300,  # 거대 격자 크기
            'MAX_WAVES': 14,            # 최대 파동 수 (4.2km)
            
            'ORIGIN_SEARCH_RADIUS_M': 500,    # 출발지 격자 탐색 반경
            'DEST_STATION_RADIUS_M': 2000,    # 도착지 정류장 수집 반경
            'GRID_TO_STATION_MAX_M': 600,     # 격자→정류장 최대거리
            
            'PM_BASE_FARE': 1000,       # PM 기본요금
            'PM_PER_MINUTE_FARE': 150,  # PM 분당요금
            
            'MIN_PM_DENSITY': 0.05,     # 최소 PM 밀도
            'MAX_PM_WAIT_MIN': 8.0,     # 최대 PM 대기시간
            'MIN_PM_WAIT_MIN': 0.5,     # 최소 PM 대기시간
        }
        
        # 1. RAPTOR 데이터 로드
        self._load_raptor_data(raptor_data_path)
        
        # 2. PM 밀도 맵 로드 (50m 격자)
        self._load_pm_density_map(pm_density_path)
        
        # 3. 300m 거대 격자 생성
        self._build_coarse_grid_system()
        
        # 4. 따릉이 정거장 로드
        self._load_bike_stations(bike_stations_path)
        
        # 5. KDTree 구축
        self._build_spatial_indices()
        
        logger.info(f"초기화 완료: {len(self.fine_grids)}개 50m격자, {len(self.coarse_grids)}개 300m격자")
        logger.info(f"RAPTOR: {len(self.stops)}개 정류장, {len(self.routes)}개 노선")
    
    def _load_raptor_data(self, path: str):
        """RAPTOR 데이터 로드"""
        logger.info(f"RAPTOR 데이터 로드: {path}")
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.stops = data['stops']
        self.routes = data['routes']
        self.trips = data.get('trips', {})
        
        # 인덱스 매핑
        self.stop_index_map = data['stop_index_map']
        self.index_to_stop = data['index_to_stop']
        
        # 노선 정보
        self.route_stop_sequences = data['route_stop_sequences']
        self.route_stop_indices = data['route_stop_indices']
        
        # PART2_NEW에서는 stop_routes를 사용하므로 호환성을 위해 둘 다 체크
        self.routes_by_stop = data.get('routes_by_stop', {})
        self.stop_routes = data.get('stop_routes', {})
        
        # routes_by_stop가 비어있으면 stop_routes 사용
        if not self.routes_by_stop and self.stop_routes:
            self.routes_by_stop = self.stop_routes
            logger.info("stop_routes로 routes_by_stop 매핑 복사")
        
        # 정류장명 매핑 (한글 표시용)
        self.stop_names = {}
        for stop_id, stop in self.stops.items():
            # 정류장명 정리
            if hasattr(stop, 'stop_name') and stop.stop_name:
                self.stop_names[stop_id] = stop.stop_name
            else:
                self.stop_names[stop_id] = stop_id
        
        # 환승 및 시간표
        self.transfers = data['transfers']
        self.timetables = data['timetables']
        self.trip_ids_by_route = data.get('trip_ids_by_route', {})
        
        logger.info(f"Timetables loaded: {len(self.timetables)}")
        if self.timetables:
            sample_route = list(self.timetables.keys())[0]
            sample_tt = self.timetables[sample_route]
            logger.info(f"Sample timetable {sample_route}: {len(sample_tt)} trips")
        
        # 노선 인덱스
        self.route_idx_map = {rid: i for i, rid in enumerate(self.routes.keys())}
        
        logger.info(f"RAPTOR 로드 완료: {len(self.stops)}개 정류장, {len(self.routes)}개 노선")
        logger.info(f"Routes-by-stop mappings: {len(self.routes_by_stop)}개")
        
        # 샘플 확인
        if self.routes_by_stop:
            sample_stop = list(self.routes_by_stop.keys())[0]
            sample_routes = self.routes_by_stop[sample_stop]
            logger.info(f"Sample stop {sample_stop}: {len(sample_routes) if isinstance(sample_routes, (list, set, tuple)) else 1} routes")
    
    def _load_pm_density_map(self, path: str):
        """50m PM 밀도 맵 로드"""
        logger.info(f"PM 밀도 맵 로드: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            pm_data = json.load(f)
        
        self.pm_density_raw = pm_data['density_map']
        self.grid_metadata = pm_data['metadata']
        self.grid_bounds = self.grid_metadata['bounds']
        
        # 50m 정밀 격자 객체 생성
        self.fine_grids = {}
        
        for grid_id, grid_data in self.pm_density_raw.items():
            # 격자 중심 좌표 계산
            row, col = int(grid_id.split('_')[1]), int(grid_id.split('_')[2])
            lat_step = 50 / 111000  # 50m를 도 단위로
            lon_step = 50 / (111000 * math.cos(math.radians(37.5)))
            
            lat = self.grid_bounds['min_lat'] + (row + 0.5) * lat_step
            lon = self.grid_bounds['min_lon'] + (col + 0.5) * lon_step
            
            fine_grid = FineGrid(
                grid_id=grid_id,
                lat=lat,\
                lon=lon,
                density=grid_data['adjusted_density'],
                availability_score=grid_data['availability_score'],
                grid_type=grid_data['grid_type']
            )
            
            self.fine_grids[grid_id] = fine_grid
        
        logger.info(f"50m 격자 생성: {len(self.fine_grids)}개")
    
    def _build_coarse_grid_system(self):
        """300m 거대 격자 시스템 구축"""
        logger.info("300m 거대 격자 시스템 구축...")
        
        self.coarse_grids = {}
        self.fine_to_coarse_map = {}  # 50m격자 → 300m격자 매핑
        
        # 300m 격자 크기 계산
        coarse_lat_step = 300 / 111000
        coarse_lon_step = 300 / (111000 * math.cos(math.radians(37.5)))
        
        # 강남구 범위를 300m 격자로 분할
        lat_range = self.grid_bounds['max_lat'] - self.grid_bounds['min_lat']
        lon_range = self.grid_bounds['max_lon'] - self.grid_bounds['min_lon']
        
        coarse_lat_count = int(lat_range / coarse_lat_step) + 1
        coarse_lon_count = int(lon_range / coarse_lon_step) + 1
        
        logger.info(f"300m 격자 크기: {coarse_lat_count} × {coarse_lon_count} = {coarse_lat_count * coarse_lon_count}개")
        
        # 300m 격자 생성
        for row in range(coarse_lat_count):
            for col in range(coarse_lon_count):
                coarse_grid_id = f"C_{row:03d}_{col:03d}"
                
                # 300m 격자 중심 좌표
                center_lat = self.grid_bounds['min_lat'] + (row + 0.5) * coarse_lat_step
                center_lon = self.grid_bounds['min_lon'] + (col + 0.5) * coarse_lon_step
                
                coarse_grid = CoarseGrid(
                    grid_id=coarse_grid_id,
                    center_lat=center_lat,
                    center_lon=center_lon
                )
                
                self.coarse_grids[coarse_grid_id] = coarse_grid
        
        # 50m 격자들을 300m 격자에 매핑
        for fine_grid_id, fine_grid in self.fine_grids.items():
            # 해당하는 300m 격자 찾기
            coarse_row = int((fine_grid.lat - self.grid_bounds['min_lat']) / coarse_lat_step)
            coarse_col = int((fine_grid.lon - self.grid_bounds['min_lon']) / coarse_lon_step)
            
            coarse_row = min(coarse_row, coarse_lat_count - 1)
            coarse_col = min(coarse_col, coarse_lon_count - 1)
            
            coarse_grid_id = f"C_{coarse_row:03d}_{coarse_col:03d}"
            
            if coarse_grid_id in self.coarse_grids:
                self.coarse_grids[coarse_grid_id].fine_grids.append(fine_grid_id)
                self.fine_to_coarse_map[fine_grid_id] = coarse_grid_id
        
        logger.info(f"300m 격자 생성 완료: {len(self.coarse_grids)}개")
    
    def _load_bike_stations(self, path: str):
        """따릉이 정거장 로드"""
        import pandas as pd
        
        logger.info(f"따릉이 정거장 로드: {path}")
        
        try:
            bike_df = pd.read_csv(path)
            self.bike_stations = []
            
            for _, row in bike_df.iterrows():
                if (self.grid_bounds['min_lat'] <= row['lat'] <= self.grid_bounds['max_lat'] and
                    self.grid_bounds['min_lon'] <= row['lon'] <= self.grid_bounds['max_lon']):
                    self.bike_stations.append({
                        'station_id': row['station_id'],
                        'lat': row['lat'],
                        'lon': row['lon']
                    })
            
            logger.info(f"따릉이 정거장: {len(self.bike_stations)}개")
        
        except Exception as e:
            logger.warning(f"따릉이 데이터 로드 실패: {e}")
            self.bike_stations = []
    
    def _build_spatial_indices(self):
        """공간 인덱스 (KDTree) 구축"""
        logger.info("공간 인덱스 구축...")
        
        # 50m 격자 KDTree
        fine_coords = []
        self.fine_grid_ids_list = []
        
        for grid_id, grid in self.fine_grids.items():
            fine_coords.append([grid.lat, grid.lon])
            self.fine_grid_ids_list.append(grid_id)
        
        self.fine_grid_kdtree = KDTree(np.array(fine_coords))
        
        # 300m 격자 KDTree  
        coarse_coords = []
        self.coarse_grid_ids_list = []
        
        for grid_id, grid in self.coarse_grids.items():
            coarse_coords.append([grid.center_lat, grid.center_lon])
            self.coarse_grid_ids_list.append(grid_id)
        
        self.coarse_grid_kdtree = KDTree(np.array(coarse_coords))
        
        # 정류장 KDTree
        stop_coords = []
        self.stop_ids_list = []
        
        for stop_id, stop in self.stops.items():
            stop_coords.append([stop.stop_lat, stop.stop_lon])
            self.stop_ids_list.append(stop_id)
        
        self.stop_kdtree = KDTree(np.array(stop_coords))
        
        logger.info("공간 인덱스 구축 완료")
    
    def _haversine_distance_km(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Haversine 거리 계산 (km)"""
        R = 6371  # 지구 반지름 (km)
        
        lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
        lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    def _haversine_distance_m(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Haversine 거리 계산 (m)"""
        return self._haversine_distance_km(lat1, lon1, lat2, lon2) * 1000
    
    def _find_origin_grid(self, origin: Tuple[float, float]) -> Optional[FineGrid]:
        """출발지 최적 50m 격자 찾기"""
        origin_coords = np.array([[origin[0], origin[1]]])
        
        # 500m 반경 내 격자들 찾기
        distances, indices = self.fine_grid_kdtree.query(
            origin_coords, 
            k=min(20, len(self.fine_grid_ids_list))
        )
        
        best_grid = None
        best_score = -1
        
        for dist_deg, idx in zip(distances[0], indices[0]):
            dist_m = dist_deg * 111000
            
            if dist_m > self.params['ORIGIN_SEARCH_RADIUS_M']:
                continue
            
            grid_id = self.fine_grid_ids_list[idx]
            grid = self.fine_grids[grid_id]
            
            if grid.density < self.params['MIN_PM_DENSITY']:
                continue
            
            # 점수: 밀도 높고 거리 가까운 것 우선
            score = grid.density / (dist_m + 10)  # 10m 기본 추가
            
            if score > best_score:
                best_score = score
                best_grid = grid
        
        return best_grid
    
    def _get_coarse_neighbors(self, coarse_grid_id: str) -> List[str]:
        """300m 격자의 이웃들 찾기 (8방향)"""
        row, col = int(coarse_grid_id.split('_')[1]), int(coarse_grid_id.split('_')[2])
        
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                
                neighbor_id = f"C_{row + dr:03d}_{col + dc:03d}"
                if neighbor_id in self.coarse_grids:
                    neighbors.append(neighbor_id)
        
        return neighbors
    
    def _analyze_destination_pm_accessibility(self, dest_coords: Tuple[float, float]) -> Dict[str, PMEgressOption]:
        """도착지 PM 접근성 분석"""
        logger.info("도착지 PM 접근성 분석...")
        
        # 도착지 반경 2km 내 정류장들 찾기
        dest_query = np.array([[dest_coords[0], dest_coords[1]]])
        distances, indices = self.stop_kdtree.query(
            dest_query, 
            k=min(50, len(self.stop_ids_list))
        )
        
        pm_egress_options = {}
        
        for dist_deg, stop_idx in zip(distances[0], indices[0]):
            dist_m = dist_deg * 111000
            
            if dist_m > self.params['DEST_STATION_RADIUS_M']:
                continue
            
            stop_id = self.stop_ids_list[stop_idx]
            stop = self.stops[stop_id]
            station_coords = (stop.stop_lat, stop.stop_lon)
            
            # 정류장 위치의 50m 격자 밀도 확인
            station_query = np.array([[stop.stop_lat, stop.stop_lon]])
            fine_distances, fine_indices = self.fine_grid_kdtree.query(station_query, k=1)
            
            # numpy.int64 처리
            if isinstance(fine_indices[0], np.int64):
                nearest_fine_idx = int(fine_indices[0])
            else:
                if len(fine_indices[0]) == 0:
                    continue
                nearest_fine_idx = fine_indices[0][0]
            nearest_fine_grid_id = self.fine_grid_ids_list[nearest_fine_idx]
            station_grid = self.fine_grids[nearest_fine_grid_id]
            
            # 6️⃣ 밀도 클램핑 + PM 대기시간 계산 안정화
            clamped_density = max(self.params['MIN_PM_DENSITY'], 
                                min(station_grid.density, 0.95))
            
            if clamped_density < self.params['MIN_PM_DENSITY']:
                pm_wait_time = float('inf')
                viability = 'impossible'
            else:
                raw_wait = 1.5 / clamped_density
                pm_wait_time = round(max(
                    self.params['MIN_PM_WAIT_MIN'],
                    min(raw_wait, self.params['MAX_PM_WAIT_MIN'])
                ), 1)  # 0.1분 단위 라운딩
                
                # 생존 가능성 평가
                if pm_wait_time <= 3.0:
                    viability = 'excellent'
                elif pm_wait_time <= 5.0:
                    viability = 'good'
                else:
                    viability = 'poor'
            
            # 정류장 → 도착지 PM 주행시간
            dest_distance_m = self._haversine_distance_m(
                stop.stop_lat, stop.stop_lon,
                dest_coords[0], dest_coords[1]
            )
            
            pm_ride_time = dest_distance_m / self.params['PM_SPEED_MPM']
            pm_cost = self.params['PM_BASE_FARE'] + int(pm_ride_time * self.params['PM_PER_MINUTE_FARE'])
            
            pm_egress_options[stop_id] = PMEgressOption(
                station_id=stop_id,
                station_coords=station_coords,
                pm_wait_time_min=pm_wait_time if pm_wait_time != float('inf') else 999,
                pm_ride_time_min=pm_ride_time,
                pm_cost_won=pm_cost,
                dest_distance_m=dest_distance_m,
                viability=viability
            )
        
        # 통계 로그
        viable_count = sum(1 for opt in pm_egress_options.values() if opt.viability in ['excellent', 'good'])
        logger.info(f"도착지 PM 접근성: 총 {len(pm_egress_options)}개 정류장, {viable_count}개 실용적")
        
        return pm_egress_options
    
    def _wave_expansion_search(self, origin: Tuple[float, float], dest_coords: Tuple[float, float],
                              dep_time_min: float, mode: str = "EXPLAIN") -> List[MultimodeJourney]:
        """이중 격자 파동 확산 검색"""
        
        logger.info("=== 파동 확산 검색 시작 ===")
        
        # 1. 출발지 50m 격자 찾기 + PM 대기시간 계산
        origin_fine_grid = self._find_origin_grid(origin)
        
        if not origin_fine_grid:
            logger.warning("출발지 근처에 적절한 PM 격자를 찾을 수 없음")
            return []
        
        # 6️⃣ PM 초기 대기시간 안정화 (한 번만!)
        clamped_density = max(self.params['MIN_PM_DENSITY'], 
                            min(origin_fine_grid.density, 0.95))
        raw_wait = 1.5 / clamped_density
        pm_initial_wait = round(max(
            self.params['MIN_PM_WAIT_MIN'],
            min(raw_wait, self.params['MAX_PM_WAIT_MIN'])
        ), 1)  # 0.1분 단위 라운딩
        
        if mode == "EXPLAIN":
            logger.info(f"출발 격자: {origin_fine_grid.grid_id}, 밀도: {origin_fine_grid.density:.3f}")
            logger.info(f"PM 초기 대기: {pm_initial_wait:.1f}분")
        
        # 2. 출발지 근처 300m 격자로 변환
        origin_coarse_grid_id = self.fine_to_coarse_map.get(origin_fine_grid.grid_id)
        
        if not origin_coarse_grid_id:
            logger.warning("출발지를 300m 격자로 매핑할 수 없음")
            return []
        
        # 3. 300m 격자 파동 확산
        coarse_labels = {}  # coarse_grid_id -> (arrival_time, total_distance, path)
        frontier = [(dep_time_min + pm_initial_wait, origin_coarse_grid_id, 0.0, [origin_coarse_grid_id])]
        
        coarse_labels[origin_coarse_grid_id] = (dep_time_min + pm_initial_wait, 0.0, [origin_coarse_grid_id])
        
        wave_stats = []
        
        for wave in range(self.params['MAX_WAVES']):
            if not frontier:
                break
            
            current_frontier = list(frontier)
            frontier = []
            
            expanded_count = 0
            
            for arrival_time, current_grid_id, total_distance, path in current_frontier:
                neighbors = self._get_coarse_neighbors(current_grid_id)
                
                for neighbor_id in neighbors:
                    # 6️⃣ 웨이브 시간 1분 단위 스냅 (안정화)
                    ride_time = 1.0  # 300m = 1분 고정
                    new_arrival = round(arrival_time + ride_time, 0)  # 정수 분으로 스냅
                    new_distance = total_distance + 300.0
                    new_path = path + [neighbor_id]
                    
                    # 더 빠른 경로인 경우만 업데이트
                    if (neighbor_id not in coarse_labels or 
                        new_arrival < coarse_labels[neighbor_id][0]):
                        
                        coarse_labels[neighbor_id] = (new_arrival, new_distance, new_path)
                        frontier.append((new_arrival, neighbor_id, new_distance, new_path))
                        expanded_count += 1
            
            wave_stats.append(expanded_count)
            
            if mode == "EXPLAIN":
                logger.info(f"WAVE {wave + 1}: {expanded_count}개 격자 확장")
        
        if mode == "EXPLAIN":
            total_reached = len(coarse_labels)
            max_distance = max((data[1] for data in coarse_labels.values()), default=0)
            logger.info(f"총 도달 격자: {total_reached}개, 최대거리: {max_distance:.0f}m")
        
        # 3️⃣ 결정론적 프루닝으로 시드 생성 (수천 개 → ~100개)
        coarse_grid_best = {}  # 300m 격자당 가장 빠른 도착만 유지
        
        for coarse_grid_id, (arrival_time, total_distance, path) in coarse_labels.items():
            if (coarse_grid_id not in coarse_grid_best or 
                arrival_time < coarse_grid_best[coarse_grid_id][0]):
                coarse_grid_best[coarse_grid_id] = (arrival_time, total_distance, path, pm_initial_wait)
        
        # 시드 후보 생성
        seed_candidates = []
        
        for coarse_grid_id, (arrival_time, total_distance, path, initial_wait) in coarse_grid_best.items():
            coarse_grid = self.coarse_grids[coarse_grid_id]
            
            # 300m 격자 근처 정류장들 찾기
            grid_query = np.array([[coarse_grid.center_lat, coarse_grid.center_lon]])
            distances, indices = self.stop_kdtree.query(grid_query, k=5)  # 5개로 축소
            
            for dist_deg, stop_idx in zip(distances[0], indices[0]):
                dist_m = dist_deg * 111000
                
                if dist_m > self.params['GRID_TO_STATION_MAX_M']:
                    continue
                
                stop_id = self.stop_ids_list[stop_idx]
                
                # 격자 → 정류장 도보시간
                walk_time = dist_m / self.params['WALK_SPEED_MPM']
                seed_arrival = round(arrival_time + walk_time, 1)
                
                seed_candidates.append((stop_id, seed_arrival, total_distance, initial_wait))
        
        # 3️⃣ 시드 프루닝: 공간 분산 + arrival_time
        seed_candidates.sort(key=lambda x: x[1])  # arrival_time 기준 정렬
        
        # 300m 셀당 최대 1개 유지 (공간 분산)
        coarse_cell_seeds = {}
        for stop_id, seed_arrival, total_distance, initial_wait in seed_candidates:
            if stop_id in self.stops:
                stop = self.stops[stop_id]
                # 300m 셀 계산
                coarse_lat_step = 300 / 111000
                coarse_lon_step = 300 / (111000 * math.cos(math.radians(37.5)))
                coarse_row = int((stop.stop_lat - self.grid_bounds['min_lat']) / coarse_lat_step)
                coarse_col = int((stop.stop_lon - self.grid_bounds['min_lon']) / coarse_lon_step)
                coarse_cell = f"C_{coarse_row:03d}_{coarse_col:03d}"
                
                # 셀당 가장 빠른 것만 유지
                if coarse_cell not in coarse_cell_seeds or seed_arrival < coarse_cell_seeds[coarse_cell][1]:
                    coarse_cell_seeds[coarse_cell] = (stop_id, seed_arrival, total_distance, initial_wait)
        
        # 상위 120개로 확장 (공간 분산 후)
        spatial_filtered = list(coarse_cell_seeds.values())[:120]
        
        # 동일 정류장 중복 제거 (가장 빠른 것만)
        raptor_seeds = {}
        for stop_id, seed_arrival, total_distance, initial_wait in spatial_filtered:
            if stop_id not in raptor_seeds or seed_arrival < raptor_seeds[stop_id][0]:
                raptor_seeds[stop_id] = (seed_arrival, total_distance, initial_wait)
        
        if mode == "EXPLAIN":
            logger.info(f"RAPTOR 시드 프루닝: {len(coarse_grid_best)}개 격자 → {len(raptor_seeds)}개 시드")
        
        # 5. 도착지 PM 접근성 분석
        pm_egress_options = self._analyze_destination_pm_accessibility(dest_coords)
        
        # 6. RAPTOR 실행 + 도착지 옵션 확장
        journeys = self._run_raptor_with_pm_egress(
            raptor_seeds, dest_coords, pm_egress_options, dep_time_min
        )
        
        return journeys
    
    def _run_raptor_with_pm_egress(self, seeds: Dict[str, Tuple[float, float, float]], 
                                  dest_coords: Tuple[float, float],
                                  pm_egress_options: Dict[str, PMEgressOption],
                                  dep_time_min: float) -> List[MultimodeJourney]:
        """RAPTOR 실행 + PM 이그레스 옵션"""
        
        logger.info("RAPTOR + PM 이그레스 실행...")
        
        # 간단한 RAPTOR 구현 (실제로는 더 복잡해야 함)
        journeys = []
        
        # PART2_NEW의 실제 RAPTOR 구현 사용
        journeys = self._run_actual_raptor(seeds, dest_coords, pm_egress_options, dep_time_min)
        
        logger.info(f"실제 RAPTOR 완료: {len(journeys)}개")
        return journeys
    
    def _run_actual_raptor(self, seeds: Dict[str, Tuple[float, float, float]], 
                          dest_coords: Tuple[float, float],
                          pm_egress_options: Dict[str, PMEgressOption],
                          dep_time_min: float) -> List[MultimodeJourney]:
        """실제 RAPTOR 알고리즘 (PART2_NEW 기반)"""
        
        # 목적지 근처 정류장들 (확장: k=30, 반경 2.0km)
        dest_query = np.array([[dest_coords[0], dest_coords[1]]])
        distances, indices = self.stop_kdtree.query(dest_query, k=30)
        
        dest_stop_indices = []
        for dist_deg, stop_idx in zip(distances[0], indices[0]):
            dist_m = dist_deg * 111000
            if dist_m <= 2000:  # 1.5km → 2.0km
                dest_stop_indices.append(stop_idx)
        
        if not dest_stop_indices:
            return []
        
        # PART2_NEW 스타일 RAPTOR 초기화
        INF = float('inf')
        MAX_ROUNDS = 4
        tau = [[INF] * len(self.stops) for _ in range(MAX_ROUNDS + 1)]
        parent = [{} for _ in range(MAX_ROUNDS + 1)]
        
        # 시드 설정 및 디버깅
        seed_count = 0
        for stop_id, (seed_arrival, pm_distance, initial_wait) in seeds.items():
            if stop_id in self.stop_index_map:
                stop_idx = self.stop_index_map[stop_id]
                # 이 stop에 routes가 있는지 확인
                if stop_idx in self.routes_by_stop:
                    routes_serving = self.routes_by_stop[stop_idx]
                    route_count = len(routes_serving) if isinstance(routes_serving, (list, set, tuple)) else (1 if routes_serving else 0)
                    if route_count > 0:  # 노선이 있는 시드만 설정
                        tau[0][stop_idx] = seed_arrival
                        parent[0][stop_idx] = {
                            'type': 'seed',
                            'pm_distance': pm_distance,
                            'initial_wait': initial_wait
                        }
                        seed_count += 1
                        
                        # 디버그: 시드 정보 (인덱스 사용)
                        if seed_count <= 3:
                            if isinstance(routes_serving, (list, set, tuple)):
                                print(f"DEBUG: seed {stop_id} (idx={stop_idx}), routes_serving={len(routes_serving)}")
                                if len(routes_serving) > 0:
                                    print(f"DEBUG: first_route={list(routes_serving)[:3]}")
                            else:
                                print(f"DEBUG: seed {stop_id} (idx={stop_idx}), single_route={routes_serving}")
        
        print(f"DEBUG: total seeds set = {seed_count}")
        if seed_count == 0:
            print("ERROR: No seeds were set! Check coordinate conversion and stop proximity.")
        
        # RAPTOR 라운드 실행 (PART2_NEW 방식)
        for k in range(1, MAX_ROUNDS + 1):
            # 1. 노선 기반 전파 (실제 timetable 사용)
            marked_stops = self._route_based_propagation_v3(k, tau, parent, dep_time_min)
            
            # 2. 환승 전파 (도보)
            self._transfer_propagation_v3(k, tau, parent, marked_stops)
        
        # 결과 수집 및 여정 구성
        journeys = []
        print(f"DEBUG: Checking {len(dest_stop_indices)} destination stops...")
        
        for dest_idx in dest_stop_indices:
            dest_stop_id = self.index_to_stop.get(dest_idx, f"idx_{dest_idx}")
            best_time = INF
            best_round = -1
            
            for k in range(MAX_ROUNDS + 1):
                if tau[k][dest_idx] < best_time:
                    best_time = tau[k][dest_idx]
                    best_round = k
            
            print(f"DEBUG: dest {dest_stop_id} (idx={dest_idx}): best_time={best_time}, round={best_round}")
            
            if best_round >= 0 and best_time < INF:
                # round=0은 시드 자체가 목적지라는 의미가 아니라
                # 시드에서 바로 PM으로 목적지에 도달했다는 의미
                # 하지만 실제로는 RAPTOR를 통해 도달한 것이므로 일반 경로로 구성
                journey = self._build_journey_from_raptor(
                    dest_idx, best_round, tau, parent, dest_coords, dep_time_min
                )
                
                if journey:
                    journeys.append(journey)
                    print(f"DEBUG: Built journey to {dest_stop_id} (round={best_round})")
        
        # 추가: 도달한 모든 정류장 확인
        all_reached = []
        for k in range(MAX_ROUNDS + 1):
            for stop_idx, time_val in enumerate(tau[k]):
                if time_val < INF:
                    stop_id = self.index_to_stop.get(stop_idx, f"idx_{stop_idx}")
                    all_reached.append((stop_id, stop_idx, k, time_val))
        
        print(f"DEBUG: Total reached stops: {len(all_reached)}")
        if len(all_reached) > 0:
            print(f"DEBUG: Sample reached: {all_reached[:10]}")
        
        return journeys
    
    def _build_journey_from_raptor(self, dest_idx: int, final_round: int,
                                  tau, parent, dest_coords: Tuple[float, float],
                                  dep_time_min: float) -> Optional[MultimodeJourney]:
        """RAPTOR 결과에서 여정 구성"""
        
        segments = []
        path = []
        
        # 역추적
        current_idx = dest_idx
        current_round = final_round
        
        while current_round >= 0:
            path.append((current_idx, current_round))
            if current_round == 0:
                break
            
            parent_info = parent[current_round].get(current_idx)
            if not parent_info:
                print(f"DEBUG: No parent info for idx={current_idx}, round={current_round}")
                break
            
            if parent_info['type'] == 'transit':
                board_stop_id = parent_info['board_stop']
                current_idx = self.stop_index_map[board_stop_id]
                current_round = parent_info['from_round']
                print(f"DEBUG: Transit path: {parent_info['alight_stop']} <- {board_stop_id}")
            elif parent_info['type'] == 'transfer':
                # 환승도 처리
                from_stop_id = parent_info['from_stop']
                current_idx = self.stop_index_map[from_stop_id]
                current_round = parent_info['from_round']
                print(f"DEBUG: Transfer path: {parent_info['to_stop']} <- {from_stop_id}")
            else:
                print(f"DEBUG: Unknown parent type: {parent_info['type']}")
                break
        
        # path가 비어있거나 너무 짧은 경우만 제외
        # 직접 연결된 경로도 허용 (단일 대중교통 구간)
        if len(path) < 1:
            return None
        
        path.reverse()
        
        # 시드 정보
        seed_info = parent[0].get(path[0][0], {})
        pm_distance = seed_info.get('pm_distance', 0)
        initial_wait = seed_info.get('initial_wait', 0)
        
        # PM 구간
        pm_ride_time = pm_distance / self.params['PM_SPEED_MPM']
        pm_cost = self.params['PM_BASE_FARE'] + int(pm_ride_time * self.params['PM_PER_MINUTE_FARE'])
        
        if initial_wait > 0:
            segments.append({
                'mode': 'pm_wait',
                'duration_min': initial_wait,
                'cost_won': 0,
                'description': 'PM 대기 (출발)'
            })
        
        if pm_ride_time > 0:
            segments.append({
                'mode': 'kickboard',
                'duration_min': pm_ride_time,
                'distance_m': pm_distance,
                'cost_won': pm_cost,
                'description': f'킥보드 {pm_distance:.0f}m'
            })
        
        # 대중교통 구간 (개선된 추적)
        total_transit_cost = 0
        total_pm_cost = 0
        for i in range(1, len(path)):
            current_idx, current_round = path[i]
            parent_info = parent[current_round][current_idx]
            
            if parent_info['type'] == 'transit':
                route_name = parent_info.get('route_name', '대중교통')
                board_name = self.stop_names.get(parent_info['board_stop'], parent_info['board_stop'])
                alight_name = self.stop_names.get(parent_info['alight_stop'], parent_info['alight_stop'])
                
                # 정확한 대중교통 소요시간 계산 (시간표 기반)
                board_time = parent_info['board_time']
                alight_time = parent_info['alight_time'] 
                transit_time = alight_time - board_time
                
                # 비현실적인 짧은 대중교통 시간 체크 및 보정
                if transit_time < 1.0:  # 1분 미만은 명확히 오류
                    print(f"WARNING: 비정상적 대중교통 시간 - {route_name}: {board_name}→{alight_name}, "
                          f"board_time={board_time:.1f}, alight_time={alight_time:.1f}, duration={transit_time:.1f}분")
                    transit_time = max(transit_time, 2.0)  # 최소 2분 보정
                
                segments.append({
                    'mode': 'transit',
                    'duration_min': transit_time,
                    'cost_won': 1370,
                    'from': board_name,
                    'to': alight_name,
                    'route': route_name,
                    'description': f'{route_name}: {board_name} → {alight_name}'
                })
                total_transit_cost += 1370
            
            elif parent_info['type'] == 'transfer':
                from_name = self.stop_names.get(parent_info['from_stop'], parent_info['from_stop'])
                to_name = self.stop_names.get(parent_info['to_stop'], parent_info['to_stop'])
                walk_time = parent_info['walk_time']
                
                # 불필요한 환승 필터링
                # 1. 같은 이름의 정류장 간 환승
                if from_name == to_name:
                    continue  # 중복 환승 제거
                
                # 2. 매우 짧은 환승 시간 (0.1분 미만)
                if walk_time < 0.1:
                    continue  # 실질적 0분 환승 제거
                
                # 3. 비슷한 이름 정류장 간 환승 (역 접미사만 다른 경우)
                from_base = from_name.replace('역', '').replace('station', '').strip()
                to_base = to_name.replace('역', '').replace('station', '').strip()
                if from_base == to_base and walk_time < 0.5:
                    continue  # 예: "신논현역" ↔ "신논현"
                
                segments.append({
                    'mode': 'walk',
                    'duration_min': walk_time,
                    'cost_won': 0,
                    'from': from_name,
                    'to': to_name,
                    'description': f'도보 환승: {from_name} → {to_name} ({walk_time:.1f}분)'
                })
        
        # 마지막 이동: 도보 vs PM 선택 (이그레스 통합)
        final_stop_id = self.index_to_stop[dest_idx]
        final_stop = self.stops[final_stop_id]
        final_walk_dist = self._haversine_distance_m(
            final_stop.stop_lat, final_stop.stop_lon,
            dest_coords[0], dest_coords[1]
        )
        final_walk_time = final_walk_dist / self.params['WALK_SPEED_MPM']
        
        # 이그레스 PM 치환: 도보 > 350m이고 PM 가용성이 good 이상이면 PM 사용
        use_pm_egress = False
        if final_walk_dist > 350:  # 350m 기준
            # 정류장의 PM 가용성 확인
            station_query = np.array([[final_stop.stop_lat, final_stop.stop_lon]])
            fine_distances, fine_indices = self.fine_grid_kdtree.query(station_query, k=1)
            
            if isinstance(fine_indices[0], np.int64):
                nearest_fine_idx = int(fine_indices[0])
            else:
                nearest_fine_idx = fine_indices[0][0] if len(fine_indices[0]) > 0 else 0
            
            if nearest_fine_idx < len(self.fine_grid_ids_list):
                nearest_fine_grid_id = self.fine_grid_ids_list[nearest_fine_idx]
                station_grid = self.fine_grids[nearest_fine_grid_id]
                
                # PM 가용성 평가
                clamped_density = max(self.params['MIN_PM_DENSITY'], 
                                    min(station_grid.density, 0.95))
                raw_wait = 1.5 / clamped_density
                pm_wait_time = max(self.params['MIN_PM_WAIT_MIN'],
                                 min(raw_wait, self.params['MAX_PM_WAIT_MIN']))
                
                # excellent (≤3분) 또는 good (≤5분)이면 PM 사용
                if pm_wait_time <= 5.0:
                    use_pm_egress = True
        
        if use_pm_egress:
            # PM 이그레스 사용
            pm_ride_time = final_walk_dist / self.params['PM_SPEED_MPM']
            pm_cost = self.params['PM_BASE_FARE'] + int(pm_ride_time * self.params['PM_PER_MINUTE_FARE'])
            
            segments.append({
                'mode': 'pm_wait',
                'duration_min': pm_wait_time,
                'cost_won': 0,
                'description': f'PM 대기 (이그레스)'
            })
            
            segments.append({
                'mode': 'kickboard',
                'duration_min': pm_ride_time,
                'distance_m': final_walk_dist,
                'cost_won': pm_cost,
                'description': f'킥보드 {final_walk_dist:.0f}m (이그레스)'
            })
            
            total_pm_cost += pm_cost
            final_walk_time = pm_wait_time + pm_ride_time
            final_walk_dist = 0  # PM으로 대체했으므로 도보 거리는 0
        else:
            # 일반 도보
            segments.append({
                'mode': 'walk',
                'duration_min': final_walk_time,
                'distance_m': final_walk_dist,
                'cost_won': 0,
                'description': f'도보 {final_walk_dist:.0f}m'
            })
        
        # 총계 - 세그먼트별 실제 소요시간 합산
        total_time = sum(segment['duration_min'] for segment in segments)
        total_cost = pm_cost + total_transit_cost + total_pm_cost
        
        return MultimodeJourney(
            segments=segments,
            total_time_min=total_time,
            total_cost_won=total_cost,
            n_transfers=len([s for s in segments if s['mode'] == 'transit']) - 1 if len([s for s in segments if s['mode'] == 'transit']) > 0 else 0,
            total_walk_m=final_walk_dist + sum(s.get('distance_m', 0) for s in segments if s['mode'] == 'walk'),
            journey_type='multimodal',
            pm_portion_min=pm_ride_time
        )
    
    def _build_pm_only_journey_from_seed(self, dest_idx: int, tau, parent, 
                                        dest_coords: Tuple[float, float],
                                        dep_time_min: float) -> Optional[MultimodeJourney]:
        """시드에서 도착지까지 PM 전용 경로 구성"""
        
        # 시드 정보 추출
        seed_info = parent[0].get(dest_idx, {})
        if seed_info.get('type') != 'seed':
            return None
        
        pm_distance = seed_info.get('pm_distance', 0)
        initial_wait = seed_info.get('initial_wait', 0)
        
        if pm_distance == 0:
            return None
        
        segments = []
        
        # PM 대기
        if initial_wait > 0:
            segments.append({
                'mode': 'pm_wait',
                'duration_min': initial_wait,
                'cost_won': 0,
                'description': 'PM 대기 (출발)'
            })
        
        # PM 주행
        pm_ride_time = pm_distance / self.params['PM_SPEED_MPM']
        pm_cost = self.params['PM_BASE_FARE'] + int(pm_ride_time * self.params['PM_PER_MINUTE_FARE'])
        
        segments.append({
            'mode': 'kickboard',
            'duration_min': pm_ride_time,
            'distance_m': pm_distance,
            'cost_won': pm_cost,
            'description': f'킥보드 {pm_distance:.0f}m'
        })
        
        # 도착지 근처 정류장에서 도착지까지 도보
        dest_stop_id = self.index_to_stop.get(dest_idx)
        if dest_stop_id and dest_stop_id in self.stops:
            dest_stop = self.stops[dest_stop_id]
            final_walk_dist = self._haversine_distance_m(
                dest_stop.stop_lat, dest_stop.stop_lon,
                dest_coords[0], dest_coords[1]
            )
            final_walk_time = final_walk_dist / self.params['WALK_SPEED_MPM']
            
            segments.append({
                'mode': 'walk',
                'duration_min': final_walk_time,
                'distance_m': final_walk_dist,
                'cost_won': 0,
                'description': f'도보 {final_walk_dist:.0f}m'
            })
        else:
            final_walk_dist = 0
            final_walk_time = 0
        
        # 전체 시간 및 비용
        total_time = initial_wait + pm_ride_time + final_walk_time
        total_cost = pm_cost
        
        return MultimodeJourney(
            segments=segments,
            total_time_min=total_time,
            total_cost_won=total_cost,
            n_transfers=0,
            total_walk_m=final_walk_dist,
            journey_type='pm_only',
            pm_portion_min=pm_ride_time
        )
    
    def _format_route_name(self, route) -> str:
        """실제 노선 정보를 한글 노선명으로 포맷"""
        
        # 노선 타입 및 이름 추출
        route_type = getattr(route, 'route_type', 'unknown')
        route_short_name = getattr(route, 'route_short_name', '')
        route_long_name = getattr(route, 'route_long_name', '')
        route_color = getattr(route, 'route_color', '')
        
        # 지하철 노선 처리
        if route_type == 1 or 'subway' in str(route_short_name).lower() or 'metro' in str(route_short_name).lower():
            # 지하철 노선번호 추출
            line_number = None
            if route_short_name:
                import re
                line_match = re.search(r'(\d+)', str(route_short_name))
                if line_match:
                    line_number = line_match.group(1)
            
            # 색상 기반 노선 판별
            if route_color:
                color_to_line = {
                    '00A84D': '2',  # 초록색 - 2호선
                    'EF7C1C': '3',  # 주황색 - 3호선  
                    '00A5DE': '4',  # 파란색 - 4호선
                    '996CAC': '5',  # 보라색 - 5호선
                    'CD7C2F': '6',  # 갈색 - 6호선
                    '747F00': '7',  # 올리브 - 7호선
                    'E6186C': '8',  # 분홍색 - 8호선
                    'BDB092': '9',  # 황금색 - 9호선
                }
                line_number = color_to_line.get(route_color.upper(), line_number)
            
            if line_number:
                return f"지하철 {line_number}호선"
            else:
                return "지하철"
        
        # 버스 노선 처리
        elif route_type == 3 or 'bus' in str(route_short_name).lower():
            # 버스 번호 추출
            bus_number = route_short_name
            
            # 숫자만 추출
            import re
            number_match = re.search(r'(\d+)', str(bus_number))
            if number_match:
                return f"버스 {number_match.group(1)}"
            else:
                return f"버스 {bus_number}"
        
        # 기타 노선
        else:
            if route_short_name:
                return f"노선 {route_short_name}"
            elif route_long_name:
                return f"대중교통 ({route_long_name[:10]})"
            else:
                return "대중교통"
    
    def _route_based_propagation_v3(self, k: int, tau: List[List[float]], 
                                   parent: List[Dict], dep_time_min: float = 480.0) -> Set[int]:
        """노선 기반 전파 (PART2_NEW 스타일)"""
        
        marked_stops = set()
        
        # 라운드 k-1에서 도달 가능한 정류장들로부터 서비스되는 노선들
        routes_to_scan = set()
        reachable_stops = 0
        for stop_idx in range(len(tau[k-1])):
            if tau[k-1][stop_idx] < float('inf'):
                reachable_stops += 1
                stop_id = self.index_to_stop.get(stop_idx)
                # 버그 수정: routes_by_stop은 인덱스를 키로 사용, stop_id가 아님
                if stop_idx in self.routes_by_stop:
                    routes_serving = self.routes_by_stop[stop_idx]
                    # routes_serving이 리스트가 아니라 다른 형식일 수 있음
                    if isinstance(routes_serving, (list, set, tuple)):
                        routes_to_scan.update(routes_serving)
                    else:
                        # 단일 route_id인 경우
                        routes_to_scan.add(routes_serving)
                    
                    # 디버그: 첫 번째 reachable stop
                    if k == 1 and reachable_stops == 1:
                        if isinstance(routes_serving, (list, set, tuple)):
                            print(f"DEBUG: first_reachable_stop={stop_id} (idx={stop_idx}), routes={len(routes_serving)}")
                        else:
                            print(f"DEBUG: first_reachable_stop={stop_id} (idx={stop_idx}), routes=1 (single)")
        
        if k == 1:
            print(f"DEBUG: round {k-1} reachable stops = {reachable_stops}")
        
        # 각 노선에 대해 RAPTOR 전파
        for route_id in routes_to_scan:
            if route_id not in self.timetables:
                continue
                
            route_timetable = self.timetables[route_id]
            if not route_timetable or len(route_timetable) == 0:
                continue
            
            # 디버그: 첫 번째 노선에서 데이터 구조 확인
            if len(marked_stops) == 0 and k == 1:  # 첫 라운드 첫 노선만
                print(f"DEBUG: route_id={route_id}, timetable_entries={len(route_timetable)}")
                print(f"DEBUG: first_entry_type={type(route_timetable[0])}")
                if route_timetable and len(route_timetable) > 0:
                    sample_times = route_timetable[0][:3] if isinstance(route_timetable[0], list) and len(route_timetable[0]) > 2 else route_timetable[0]
                    print(f"DEBUG: first_entry_sample={sample_times}")
                    # 시간 값이 실제로 어떤 단위인지 확인
                    if isinstance(sample_times, list) and len(sample_times) > 0:
                        first_time = sample_times[0]
                        print(f"DEBUG: first_time_value={first_time}, in_hours={first_time/60:.1f}h, dep_time_min={dep_time_min/60:.1f}h")
            
            # 실제 노선 정보
            route = self.routes.get(route_id)
            stop_sequence = self.route_stop_sequences.get(route_id, [])
            
            if len(stop_sequence) < 2:
                continue
            
            # 🔥 PART2_NEW 방식으로 수정: trip_idx로 접근
            n_trips = len(route_timetable)
            if not isinstance(route_timetable[0], list):
                continue
                
            for trip_idx in range(n_trips):
                
                # 가장 빠른 탑승 지점 찾기
                earliest_board_idx = -1
                earliest_board_time = float('inf')
                
                # 디버그: 노선 처리 시작
                reachable_stops_in_route = []
                
                # 안전한 범위 확인
                max_seq_len = min(len(stop_sequence), len(route_timetable[trip_idx]))
                
                for seq_idx in range(max_seq_len):
                    stop_id = stop_sequence[seq_idx]
                    if stop_id not in self.stop_index_map:
                        continue
                    
                    stop_idx = self.stop_index_map[stop_id]
                    passenger_arrival = tau[k-1][stop_idx]
                    
                    if passenger_arrival < float('inf'):
                        reachable_stops_in_route.append((stop_id, passenger_arrival))
                        
                        # PART2_NEW 방식: 시간표 접근 [정류장][트립] 순서
                        if seq_idx >= len(route_timetable) or trip_idx >= len(route_timetable[seq_idx]):
                            continue
                        departure_time = route_timetable[seq_idx][trip_idx]
                        
                        # RAPTOR 핵심: 승객 도착 이후 출발하는 트립만 고려
                        if departure_time < passenger_arrival:
                            continue
                        
                        # 가장 빠른 탑승 가능 시간 찾기
                        if departure_time < earliest_board_time:
                            earliest_board_idx = seq_idx
                            earliest_board_time = departure_time
                
                # 디버그: 노선별 요약
                if k == 1 and len(marked_stops) < 2 and len(reachable_stops_in_route) > 0:
                    print(f"DEBUG: route={route_id}, reachable_stops={len(reachable_stops_in_route)}, earliest_board_idx={earliest_board_idx}")
                
                # 탑승 가능한 지점이 있으면 전파
                if earliest_board_idx >= 0:
                    board_stop_id = stop_sequence[earliest_board_idx]
                    board_stop_idx = self.stop_index_map[board_stop_id]
                    
                    # 탑승점 이후 모든 하차점으로 전파
                    # 안전한 인덱스 확인
                    max_seq_idx = min(len(stop_sequence), len(route_timetable[trip_idx]))
                    
                    for seq_idx in range(earliest_board_idx + 1, max_seq_idx):
                        alight_stop_id = stop_sequence[seq_idx]
                        if alight_stop_id not in self.stop_index_map:
                            continue
                        
                        alight_stop_idx = self.stop_index_map[alight_stop_id]
                        
                        # PART2_NEW 방식: 시간표 접근 [정류장][트립] 순서  
                        if seq_idx >= len(route_timetable) or trip_idx >= len(route_timetable[seq_idx]):
                            continue
                        arrival_time = route_timetable[seq_idx][trip_idx]
                        
                        # 도착시간이 출발시간보다 나중이어야 함
                        if arrival_time <= earliest_board_time:
                            continue
                        
                        # 더 빠른 경로인지 확인
                        if arrival_time < tau[k][alight_stop_idx]:
                            tau[k][alight_stop_idx] = arrival_time
                            
                            # 부모 정보 저장
                            parent[k][alight_stop_idx] = {
                                'type': 'transit',
                                'route_id': route_id,
                                'route_name': self._get_route_display_name(route),
                                'board_stop': board_stop_id,
                                'alight_stop': alight_stop_id,
                                'board_time': earliest_board_time,
                                'alight_time': arrival_time,
                                'from_round': k-1,
                                'board_stop_idx': board_stop_idx
                            }
                            
                            marked_stops.add(alight_stop_idx)
        
        # 디버그: 라운드 요약
        if k == 1:
            print(f"DEBUG: round {k} completed, marked {len(marked_stops)} stops")
            print(f"DEBUG: routes_scanned={len(routes_to_scan)}, valid_timetables={sum(1 for r in routes_to_scan if r in self.timetables and self.timetables[r])}")
        
        return marked_stops
    
    def _transfer_propagation_v3(self, k: int, tau: List[List[float]], 
                                parent: List[Dict], marked_stops: Set[int]):
        """환승 전파 (도보 연결)"""
        
        # 이번 라운드에서 도달한 정류장들로부터 도보 환승
        for stop_idx in marked_stops:
            stop_id = self.index_to_stop.get(stop_idx)
            if not stop_id or stop_id not in self.transfers:
                continue
            
            current_time = tau[k][stop_idx]
            if current_time >= float('inf'):
                continue
            
            # 환승 가능한 정류장들
            for transfer_stop_id, walk_time_min in self.transfers[stop_id]:
                if transfer_stop_id not in self.stop_index_map:
                    continue
                
                # 불필요한 환승 사전 필터링
                from_name = self.stop_names.get(stop_id, stop_id)
                to_name = self.stop_names.get(transfer_stop_id, transfer_stop_id)
                
                # 1. 같은 이름 정류장 환승 차단
                if from_name == to_name:
                    continue
                
                # 2. 실질적 0분 환승 차단
                if walk_time_min < 0.1:
                    continue
                
                # 3. 비슷한 이름 (역 접미사만 다른 경우) 짧은 환승 차단
                from_base = from_name.replace('역', '').replace('station', '').strip()
                to_base = to_name.replace('역', '').replace('station', '').strip()
                if from_base == to_base and walk_time_min < 0.5:
                    continue
                
                transfer_idx = self.stop_index_map[transfer_stop_id]
                new_arrival = current_time + walk_time_min
                
                if new_arrival < tau[k][transfer_idx]:
                    tau[k][transfer_idx] = new_arrival
                    parent[k][transfer_idx] = {
                        'type': 'transfer',
                        'from_stop': stop_id,
                        'to_stop': transfer_stop_id,
                        'walk_time': walk_time_min,
                        'from_round': k,
                        'from_stop_idx': stop_idx
                    }
    
    def _get_route_display_name(self, route) -> str:
        """노선 표시명 생성"""
        if not route:
            return "대중교통"
            
        route_type = getattr(route, 'route_type', 0)
        route_short_name = getattr(route, 'route_short_name', '')
        
        if route_type == 1:  # 지하철
            return f"지하철 {route_short_name}호선" if route_short_name else "지하철"
        elif route_type == 3:  # 버스
            return f"버스 {route_short_name}" if route_short_name else "버스"
        else:
            return f"노선 {route_short_name}" if route_short_name else "대중교통"
    
    def _epsilon_pareto_filter(self, journeys: List[MultimodeJourney], 
                              epsilon: float = 0.1) -> List[MultimodeJourney]:
        """ε-파레토 최적화"""
        if not journeys:
            return []
        
        pareto_optimal = []
        
        for journey in journeys:
            is_dominated = False
            
            for other in pareto_optimal:
                # 다른 경로가 모든 면에서 더 좋거나 같은가?
                other_vector = other.get_pareto_vector()
                current_vector = journey.get_pareto_vector()
                
                dominates = True
                for i in range(len(other_vector)):
                    if other_vector[i] > current_vector[i] * (1 + epsilon):
                        dominates = False
                        break
                
                if dominates:
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_optimal.append(journey)
        
        return pareto_optimal
    
    def route(self, origin_lat: float, origin_lon: float,
              dest_lat: float, dest_lon: float,
              dep_time: float = 8.0, mode: str = "EXPLAIN") -> List[MultimodeJourney]:
        """메인 라우팅 함수"""
        
        start_time = time.time()
        
        if mode == "EXPLAIN":
            logger.info(f"=== 이중 격자 파동-확산 라우팅 ===")
            logger.info(f"출발: ({origin_lat:.4f}, {origin_lon:.4f})")
            logger.info(f"도착: ({dest_lat:.4f}, {dest_lon:.4f})")
            logger.info(f"출발시각: {dep_time:.1f}시")
        
        dep_time_min = dep_time * 60  # 분 단위 변환
        
        # 파동 확산 검색
        all_journeys = self._wave_expansion_search(
            (origin_lat, origin_lon),
            (dest_lat, dest_lon),
            dep_time_min,
            mode
        )
        
        # ε-파레토 최적화
        optimized_journeys = self._epsilon_pareto_filter(all_journeys)
        
        # 상위 경로들 선택
        final_journeys = sorted(
            optimized_journeys, 
            key=lambda j: (j.total_time_min, j.n_transfers, j.total_cost_won)
        )[:5]
        
        elapsed = time.time() - start_time
        
        if mode == "EXPLAIN":
            logger.info(f"=== 라우팅 완료 ({elapsed:.2f}초) ===")
            logger.info(f"최종 경로: {len(final_journeys)}개")
            
            for i, journey in enumerate(final_journeys):
                logger.info(f"경로 {i+1}: {journey.total_time_min:.1f}분, "
                           f"{journey.total_cost_won:,}원, "
                           f"환승 {journey.n_transfers}회, "
                           f"PM {journey.pm_portion_min:.1f}분")
        
        return final_journeys
    
    def print_journey_details(self, journey: MultimodeJourney):
        """여정 상세 출력 (개선된 한글 표시)"""
        print(f"\n=== {journey.journey_type.upper()} 경로 ===")
        print(f"🕐 총 시간: {journey.total_time_min:.1f}분")
        print(f"💰 총 비용: {journey.total_cost_won:,}원")
        print(f"🔄 환승: {journey.n_transfers}회")
        print(f"🚶 도보: {journey.total_walk_m:.0f}m")
        print(f"🛴 PM 이용: {journey.pm_portion_min:.1f}분")
        
        print("\n📍 세그먼트:")
        current_time = 0
        for i, segment in enumerate(journey.segments):
            duration = segment['duration_min']
            cost = segment['cost_won']
            description = segment['description']
            
            # 시간 포매팅
            start_time = current_time
            end_time = current_time + duration
            current_time = end_time
            
            # 모드별 아이콘
            mode_icons = {
                'pm_wait': '⏳',
                'kickboard': '🛴',
                'transit': '🚌',
                'walk': '🚶'
            }
            
            icon = mode_icons.get(segment['mode'], '📍')
            
            print(f"  {i+1}. {icon} {description}")
            print(f"      시간: {start_time:.1f}→{end_time:.1f}분 ({duration:.1f}분)")
            print(f"      비용: {cost:,}원")
            
            if 'distance_m' in segment and segment['distance_m'] > 0:
                print(f"      거리: {segment['distance_m']:.0f}m")

# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 테스트 함수"""
    
    try:
        # 라우터 초기화
        router = DualGridWaveExpansionRAPTOR()
        
        # 테스트 케이스들
        test_cases = [
            {
                'name': '강남역 → 역삼역 (단거리)',
                'origin': (37.498, 127.028),
                'dest': (37.500, 127.036),
                'time': 8.0
            },
            {
                'name': '압구정 → 수서역 (장거리)',  
                'origin': (37.527, 127.028),
                'dest': (37.487, 127.100),
                'time': 9.0
            },
            {
                'name': '서초역 → 코엑스 (중거리)',
                'origin': (37.483, 127.010),
                'dest': (37.513, 127.058),
                'time': 8.5
            },
            {
                'name': '신논현 → 선릉역 (대각선)',
                'origin': (37.504, 127.025),
                'dest': (37.504, 127.049),
                'time': 9.5
            },
            {
                'name': '도곡역 → 압구정로데오 (남북)',
                'origin': (37.487, 127.032),
                'dest': (37.527, 127.040),
                'time': 10.0
            }
        ]
        
        for test_case in test_cases:
            print(f"\n{'='*60}")
            print(f"테스트: {test_case['name']}")
            print(f"{'='*60}")
            
            routes = router.route(
                test_case['origin'][0], test_case['origin'][1],
                test_case['dest'][0], test_case['dest'][1],
                test_case['time'],
                mode="EXPLAIN"
            )
            
            if routes:
                print(f"\n총 {len(routes)}개 경로 발견:")
                for route in routes:
                    router.print_journey_details(route)
            else:
                print("경로를 찾을 수 없습니다.")
    
    except Exception as e:
        logger.error(f"실행 중 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()