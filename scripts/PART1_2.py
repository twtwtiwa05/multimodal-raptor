"""
Part1: 강남구 Multi-modal RAPTOR 데이터 로더 v6.1 (최종 수정본)
- OSMnx 최신 버전 대응
- NetworkX 최신 버전 대응
- 모든 오류 해결
"""

import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field
import pickle
import json
import time
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# OSMnx는 선택적으로 import (없어도 작동)
try:
    import osmnx as ox
    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False
    print("⚠️ OSMnx 미설치 - 기본 그리드 네트워크 사용")

try:
    from scipy.spatial import KDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ SciPy 미설치 - 기본 환승 네트워크 사용")

# ============================================================================
# 데이터 구조 정의
# ============================================================================

@dataclass
class Stop:
    """정류장 정보"""
    stop_id: str
    stop_name: str
    stop_lat: float
    stop_lon: float
    stop_type: int  # 0: 버스, 1: 지하철, 2: 따릉이, 3: 킥보드
    zone_id: str = 'gangnam'  # gangnam/outside
    
@dataclass
class Route:
    """노선 정보"""
    route_id: str
    route_short_name: str
    route_long_name: str
    route_type: int  # 0: 트램, 1: 지하철, 2: 철도, 3: 버스
    stop_sequence: List[str] = field(default_factory=list)
    n_trips: int = 0
    
@dataclass
class Trip:
    """운행 정보"""
    trip_id: str
    route_id: str
    service_id: str
    direction_id: int = 0
    stop_times: List[Tuple[str, int, int]] = field(default_factory=list)

# ============================================================================
# 메인 로더 클래스
# ============================================================================

class GangnamMultiModalRAPTORLoader:
    """강남구 멀티모달 RAPTOR 데이터 로더 - 최종 수정본"""
    
    def __init__(self, 
                 gtfs_path: str = "cleaned_gtfs_data",
                 ttareungee_path: str = None,
                 shared_mobility_path: str = "shared_mobility",
                 transfer_config: Dict[str, int] = None):
        """초기화"""
        
        # 경로 설정
        self.gtfs_path = Path(gtfs_path)
        self.ttareungee_path = Path(ttareungee_path) if ttareungee_path else None
        self.shared_mobility_path = Path(shared_mobility_path) if shared_mobility_path else None
        
        # 환승 시간 설정 (분 단위)
        self.transfer_config = transfer_config or {
            'same_stop_transfer': 1,      # 동일 정류장 환승
            'walking_transfer': 5,        # 도보 환승
            'max_transfer_distance': 300  # 최대 환승 거리 (미터)
        }
        
        # 강남구 경계
        self.gangnam_bounds = {
            'min_lat': 37.460, 'max_lat': 37.550,
            'min_lon': 127.000, 'max_lon': 127.140
        }
        
        # GTFS 원본 데이터
        self.stops_df = None
        self.routes_df = None
        self.trips_df = None
        self.stop_times_df = None
        
        # === RAPTOR 핵심 데이터 구조 ===
        self.stops: Dict[str, Stop] = {}
        self.routes: Dict[str, Route] = {}
        self.trips: Dict[str, Trip] = {}
        
        # RAPTOR 인덱싱
        self.stop_index_map: Dict[str, int] = {}
        self.index_to_stop: Dict[int, str] = {}
        
        # 노선 패턴 (RAPTOR 핵심)
        self.route_stop_sequences: Dict[str, List[str]] = {}
        self.route_stop_indices: Dict[str, Dict[str, int]] = {}
        
        # 시간표 (RAPTOR 핵심)
        self.timetables: Dict[str, List[List[int]]] = {}
        self.trip_ids_by_route: Dict[str, List[str]] = {}
        
        # 환승 네트워크 (RAPTOR 핵심)
        self.transfers: Dict[str, List[Tuple[str, int]]] = {}
        self.stop_routes: Dict[str, Set[str]] = {}
        self.routes_by_stop: Dict[int, List[str]] = {}
        
        # 모빌리티
        self.bike_stations: Dict[str, Any] = {}
        self.shared_vehicles: List[Any] = []
        
        # 도로망
        self.road_graph: Optional[nx.Graph] = None
        
        # 통계
        self.stats = {
            'total_stops': 0,
            'gangnam_inside_stops': 0,
            'gangnam_outside_stops': 0,
            'total_routes': 0,
            'total_trips': 0,
            'transfers': 0
        }
        
        print("="*80)
        print("🚀 강남구 Multi-modal RAPTOR 데이터 로더 v6.1 (최종 수정본)")
        print("📚 강남을 지나는 모든 노선의 전체 정류장 포함")
        print("="*80)
    
    # ========================================================================
    # 1. GTFS 데이터 로딩
    # ========================================================================
    
    def load_gtfs_data(self) -> bool:
        """GTFS 데이터 로드"""
        print("\n📊 [1/6] GTFS 데이터 로딩...")
        start_time = time.time()
        
        try:
            # CSV 파일 로드
            self.stops_df = pd.read_csv(self.gtfs_path / 'stops.csv', dtype={'stop_id': str})
            self.routes_df = pd.read_csv(self.gtfs_path / 'routes.csv', dtype={'route_id': str})
            self.trips_df = pd.read_csv(self.gtfs_path / 'trips.csv', dtype={'trip_id': str, 'route_id': str})
            self.stop_times_df = pd.read_csv(self.gtfs_path / 'stop_times.csv', dtype={'trip_id': str, 'stop_id': str})
            
            print(f"   ✅ 전체 정류장: {len(self.stops_df):,}개")
            print(f"   ✅ 전체 노선: {len(self.routes_df):,}개")
            print(f"   ✅ 전체 운행: {len(self.trips_df):,}개")
            print(f"   ✅ 시간표 레코드: {len(self.stop_times_df):,}개")
            
            # 시간 파싱
            self._parse_times()
            
            # 메모리 최적화
            self._optimize_memory()
            
            elapsed = time.time() - start_time
            print(f"   ⏱️ 소요시간: {elapsed:.2f}초")
            
            return True
            
        except Exception as e:
            print(f"   ❌ GTFS 로딩 실패: {e}")
            return False
    
    def _parse_times(self):
        """시간 데이터 파싱 (25:30:00 형식 지원)"""
        print("   ⏰ 시간 파싱...")
        
        def parse_time(time_str):
            if pd.isna(time_str):
                return None
            try:
                parts = str(time_str).split(':')
                if len(parts) >= 2:
                    hours = int(parts[0])
                    minutes = int(parts[1])
                    return hours * 60 + minutes  # 분 단위
            except:
                return None
        
        self.stop_times_df['arrival_minutes'] = self.stop_times_df['arrival_time'].apply(parse_time)
        self.stop_times_df['departure_minutes'] = self.stop_times_df['departure_time'].apply(parse_time)
        
        # NULL 값 처리 - pandas 2.0+ 대응
        self.stop_times_df['arrival_minutes'] = self.stop_times_df['arrival_minutes'].fillna(method='ffill')
        self.stop_times_df['departure_minutes'] = self.stop_times_df['departure_minutes'].fillna(method='ffill')
    
    def _optimize_memory(self):
        """메모리 최적화"""
        # Category 타입 변환
        for col in ['stop_id', 'trip_id']:
            if col in self.stop_times_df.columns:
                unique_ratio = self.stop_times_df[col].nunique() / len(self.stop_times_df)
                if unique_ratio < 0.5:  # 50% 미만이면 category
                    self.stop_times_df[col] = self.stop_times_df[col].astype('category')
        
        # stop_sequence 최적화
        if 'stop_sequence' in self.stop_times_df.columns:
            self.stop_times_df['stop_sequence'] = self.stop_times_df['stop_sequence'].astype('uint16')
    
    # ========================================================================
    # 2. 강남 통과 노선 필터링 (핵심)
    # ========================================================================
    
    def filter_gangnam_complete(self) -> bool:
        """강남을 지나는 모든 노선과 그 노선의 전체 정류장 포함"""
        print("\n🎯 [2/6] 강남 통과 노선 필터링 (전체 정류장 포함)...")
        
        # 1. 강남구 내부 정류장 찾기
        gangnam_mask = (
            (self.stops_df['stop_lat'] >= self.gangnam_bounds['min_lat']) &
            (self.stops_df['stop_lat'] <= self.gangnam_bounds['max_lat']) &
            (self.stops_df['stop_lon'] >= self.gangnam_bounds['min_lon']) &
            (self.stops_df['stop_lon'] <= self.gangnam_bounds['max_lon'])
        )
        
        gangnam_stop_ids = set(self.stops_df[gangnam_mask]['stop_id'].astype(str))
        print(f"   📍 강남구 내부 정류장: {len(gangnam_stop_ids):,}개")
        
        # 2. 강남을 지나는 trip 찾기
        print("   🔍 강남 통과 운행 검색...")
        gangnam_trips = self.stop_times_df[
            self.stop_times_df['stop_id'].astype(str).isin(gangnam_stop_ids)
        ]['trip_id'].unique()
        print(f"   🚌 강남 통과 운행: {len(gangnam_trips):,}개")
        
        # 3. 해당 trip의 route 찾기
        gangnam_routes = self.trips_df[
            self.trips_df['trip_id'].isin(gangnam_trips)
        ]['route_id'].unique()
        print(f"   🚍 강남 통과 노선: {len(gangnam_routes):,}개")
        
        # 4. 강남 통과 노선의 모든 trip (강남 밖 구간 포함)
        print("   📊 노선 전체 구간 로드...")
        all_trips = self.trips_df[
            self.trips_df['route_id'].isin(gangnam_routes)
        ]['trip_id'].unique()
        
        # 5. 해당 trip들의 모든 정류장 (강남 밖 포함)
        all_stop_ids = self.stop_times_df[
            self.stop_times_df['trip_id'].isin(all_trips)
        ]['stop_id'].unique()
        
        print(f"   🚏 강남 통과 노선의 전체 정류장: {len(all_stop_ids):,}개")
        
        # 6. Stop 객체 생성 (강남 내외 구분)
        gangnam_inside = 0
        gangnam_outside = 0
        
        for stop_id in all_stop_ids:
            stop_data = self.stops_df[self.stops_df['stop_id'] == stop_id]
            if not stop_data.empty:
                row = stop_data.iloc[0]
                
                # 강남구 내외 판단
                is_inside = (
                    self.gangnam_bounds['min_lat'] <= row['stop_lat'] <= self.gangnam_bounds['max_lat'] and
                    self.gangnam_bounds['min_lon'] <= row['stop_lon'] <= self.gangnam_bounds['max_lon']
                )
                
                # stop_type 결정: RS_로 시작하면 지하철(1), 아니면 버스(0)
                stop_type = 1 if str(stop_id).startswith('RS_') else 0
                
                stop = Stop(
                    stop_id=str(stop_id),
                    stop_name=str(row['stop_name']),
                    stop_lat=float(row['stop_lat']),
                    stop_lon=float(row['stop_lon']),
                    stop_type=stop_type,
                    zone_id='gangnam' if is_inside else 'outside'
                )
                self.stops[stop.stop_id] = stop
                
                if is_inside:
                    gangnam_inside += 1
                else:
                    gangnam_outside += 1
        
        print(f"      강남구 내부: {gangnam_inside:,}개 정류장")
        print(f"      강남구 외부: {gangnam_outside:,}개 정류장")
        print(f"      총합: {len(self.stops):,}개 정류장")
        
        # 7. Route 객체 생성
        for route_id in gangnam_routes:
            route_data = self.routes_df[self.routes_df['route_id'] == route_id]
            if not route_data.empty:
                row = route_data.iloc[0]
                route = Route(
                    route_id=str(route_id),
                    route_short_name=str(row.get('route_short_name', '')),
                    route_long_name=str(row.get('route_long_name', '')),
                    route_type=int(row.get('route_type', 3))
                )
                self.routes[route.route_id] = route
        
        # 8. Trip 객체 생성 (메모리 고려 샘플링)
        max_trips = 50000  # 5만개까지
        if len(all_trips) > max_trips:
            print(f"   ⚠️ Trip 샘플링: {len(all_trips):,}개 중 {max_trips:,}개")
            sampled_trips = np.random.choice(all_trips, max_trips, replace=False)
        else:
            sampled_trips = all_trips
        
        for trip_id in sampled_trips:
            trip_data = self.trips_df[self.trips_df['trip_id'] == trip_id]
            if not trip_data.empty:
                row = trip_data.iloc[0]
                trip = Trip(
                    trip_id=str(trip_id),
                    route_id=str(row['route_id']),
                    service_id=str(row.get('service_id', 'default'))
                )
                self.trips[trip.trip_id] = trip
        
        # 통계 업데이트
        self.stats['total_stops'] = len(self.stops)
        self.stats['gangnam_inside_stops'] = gangnam_inside
        self.stats['gangnam_outside_stops'] = gangnam_outside
        self.stats['total_routes'] = len(self.routes)
        self.stats['total_trips'] = len(self.trips)
        
        print(f"\n   ✅ 강남 통과 {len(self.routes):,}개 노선의 전체 구간 로드 완료!")
        
        return True
    
    # ========================================================================
    # 3. RAPTOR 핵심 구조 생성
    # ========================================================================
    
    def build_raptor_structures(self) -> bool:
        """RAPTOR 알고리즘 핵심 구조 생성"""
        print("\n⚡ [3/6] RAPTOR 핵심 구조 생성...")
        
        # 1. Stop 인덱싱
        self._build_stop_indices()
        
        # 2. Route Patterns
        self._build_route_patterns()
        
        # 3. Timetables
        self._build_timetables()
        
        # 4. Stop-Route 매핑
        self._build_stop_route_mapping()
        
        # 5. Transfer Network
        self._build_transfers()
        
        return True
    
    def _build_stop_indices(self):
        """Stop 인덱싱"""
        print("   📍 Stop 인덱싱...")
        
        for idx, stop_id in enumerate(self.stops.keys()):
            self.stop_index_map[stop_id] = idx
            self.index_to_stop[idx] = stop_id
        
        print(f"      {len(self.stop_index_map):,}개 stop 인덱싱 완료")
    
    def _build_route_patterns(self):
        """Route Pattern 생성"""
        print("   🛣️ Route Pattern 생성...")
        
        pattern_count = 0
        
        for route_id in self.routes.keys():
            # 해당 route의 trip들
            route_trips = [t for t in self.trips.values() if t.route_id == route_id]
            
            if not route_trips:
                continue
            
            # 모든 trip에서 나타나는 모든 정류장 수집 (순환선 대응)
            all_stops_dict = {}  # stop_id -> [sequences]
            
            # 샘플을 늘려서 순환선도 커버
            sample_size = min(50, len(route_trips))  # 최대 50개까지 확인
            
            for trip in route_trips[:sample_size]:
                trip_stops = self.stop_times_df[
                    self.stop_times_df['trip_id'] == trip.trip_id
                ].sort_values('stop_sequence')
                
                for _, row in trip_stops.iterrows():
                    stop_id = str(row['stop_id'])
                    if stop_id in self.stops:  # 우리가 로드한 정류장만
                        seq = int(row['stop_sequence'])
                        if stop_id not in all_stops_dict:
                            all_stops_dict[stop_id] = []
                        all_stops_dict[stop_id].append(seq)
            
            # 순환선 감지 및 패턴 생성
            best_pattern = []
            if all_stops_dict:
                # 순환선 여부 확인: 한 정류장이 매우 다른 sequence 값을 가지는 경우
                is_circular = False
                for stop_id, sequences in all_stops_dict.items():
                    if len(sequences) > 1:
                        seq_range = max(sequences) - min(sequences)
                        if seq_range > 100:  # sequence 차이가 100 이상이면 순환선으로 판단
                            is_circular = True
                            break
                
                if is_circular:
                    # 순환선: 첫 번째 등장 sequence 사용
                    stop_first_seq = []
                    for stop_id, sequences in all_stops_dict.items():
                        first_seq = min(sequences)
                        stop_first_seq.append((first_seq, stop_id))
                    
                    # sequence 순서로 정렬
                    stop_first_seq.sort()
                    best_pattern = [stop_id for _, stop_id in stop_first_seq]
                else:
                    # 일반 노선: 평균 sequence 사용
                    stop_avg_seq = []
                    for stop_id, sequences in all_stops_dict.items():
                        avg_seq = sum(sequences) / len(sequences)
                        stop_avg_seq.append((avg_seq, stop_id))
                    
                    # sequence 순서로 정렬
                    stop_avg_seq.sort()
                    best_pattern = [stop_id for _, stop_id in stop_avg_seq]
            
            if len(best_pattern) >= 2:
                self.route_stop_sequences[route_id] = best_pattern
                self.route_stop_indices[route_id] = {
                    stop_id: idx for idx, stop_id in enumerate(best_pattern)
                }
                self.routes[route_id].stop_sequence = best_pattern
                self.routes[route_id].n_trips = len(route_trips)
                pattern_count += 1
                
                # 순환선 디버깅 정보
                if is_circular:
                    print(f"      🔄 순환선 감지: {route_id} ({len(best_pattern)}개 정류장)")
        
        print(f"      {pattern_count}개 route pattern 생성")
    
    def _build_timetables(self):
        """시간표 생성"""
        print("   📅 시간표 생성...")
        
        timetable_count = 0
        
        for route_id, stop_sequence in self.route_stop_sequences.items():
            route_trips = [t for t in self.trips.values() if t.route_id == route_id]
            
            if not route_trips:
                continue
            
            # 정류장별 출발시간 리스트
            stop_times_matrix = [[] for _ in range(len(stop_sequence))]
            trip_ids = []
            
            for trip in route_trips:
                trip_times = self.stop_times_df[
                    self.stop_times_df['trip_id'] == trip.trip_id
                ].sort_values('stop_sequence')
                
                trip_schedule = [None] * len(stop_sequence)
                
                for _, row in trip_times.iterrows():
                    stop_id = str(row['stop_id'])
                    if stop_id in self.route_stop_indices[route_id]:
                        idx = self.route_stop_indices[route_id][stop_id]
                        time = row.get('departure_minutes', row.get('arrival_minutes'))
                        if time is not None and not pd.isna(time):
                            trip_schedule[idx] = int(time)
                
                # 시간 보간
                trip_schedule = self._interpolate_schedule(trip_schedule)
                
                if any(t is not None for t in trip_schedule):
                    for idx, time_val in enumerate(trip_schedule):
                        if time_val is not None:
                            stop_times_matrix[idx].append(time_val)
                    trip_ids.append(trip.trip_id)
                    
                    # Trip에 저장
                    trip.stop_times = [
                        (stop_sequence[i], time_val, time_val)
                        for i, time_val in enumerate(trip_schedule)
                        if time_val is not None
                    ]
            
            # 각 정류장별 시간 정렬
            for times in stop_times_matrix:
                times.sort()
            
            if trip_ids:
                self.timetables[route_id] = stop_times_matrix
                self.trip_ids_by_route[route_id] = trip_ids
                timetable_count += 1
        
        print(f"      {timetable_count}개 route 시간표 생성")
        
        # 시간표 통계
        total_departures = sum(
            sum(len(times) for times in tt)
            for tt in self.timetables.values()
        )
        print(f"      총 {total_departures:,}개 출발시간")
    
    def _interpolate_schedule(self, times: List[Optional[int]]) -> List[Optional[int]]:
        """빈 시간 보간"""
        if not times or all(t is None for t in times):
            return times
        
        # 첫/마지막 유효시간
        first = next((i for i, t in enumerate(times) if t is not None), None)
        last = next((i for i in reversed(range(len(times))) if times[i] is not None), None)
        
        if first is None or last is None:
            return times
        
        # 중간값 보간
        for i in range(first + 1, last):
            if times[i] is None:
                # 앞뒤 찾기
                prev_idx = i - 1
                while prev_idx >= first and times[prev_idx] is None:
                    prev_idx -= 1
                
                next_idx = i + 1
                while next_idx <= last and times[next_idx] is None:
                    next_idx += 1
                
                if prev_idx >= first and next_idx <= last:
                    # 선형 보간
                    ratio = (i - prev_idx) / (next_idx - prev_idx)
                    times[i] = int(times[prev_idx] + ratio * (times[next_idx] - times[prev_idx]))
        
        return times
    
    def _build_stop_route_mapping(self):
        """Stop-Route 매핑"""
        print("   🔗 Stop-Route 매핑...")
        
        for route_id, stop_sequence in self.route_stop_sequences.items():
            for stop_id in stop_sequence:
                if stop_id not in self.stop_routes:
                    self.stop_routes[stop_id] = set()
                self.stop_routes[stop_id].add(route_id)
        
        # 인덱스 기반 매핑
        for stop_id, routes in self.stop_routes.items():
            if stop_id in self.stop_index_map:
                idx = self.stop_index_map[stop_id]
                self.routes_by_stop[idx] = list(routes)
        
        connected_stops = len([s for s in self.stop_routes if self.stop_routes[s]])
        print(f"      {connected_stops:,}개 stop에 route 연결")
    
    def _build_transfers(self):
        """환승 네트워크 구축 - 도시철도환승정보 포함"""
        print("   🔄 환승 네트워크 구축...")
        
        transfer_count = 0
        
        # 1. 지하철역 간 환승 (도시철도환승정보.xlsx 활용)
        print("      🚇 도시철도 환승정보 로드...")
        subway_transfers = 0
        
        try:
            # 도시철도환승정보 읽기
            subway_xfer_df = pd.read_excel('202303_GTFS_도시철도환승정보.xlsx', 
                                          sheet_name='Xfer', header=7)
            
            for idx, row in subway_xfer_df.iterrows():
                from_id = str(row['Fr_Stop_ID'])
                to_id = str(row['To_Stop_ID'])
                transfer_time = int(float(row['Time_Min']))  # 분 단위
                
                # 두 역이 모두 우리 데이터에 있는지 확인
                if from_id in self.stops and to_id in self.stops:
                    if from_id not in self.transfers:
                        self.transfers[from_id] = []
                    self.transfers[from_id].append((to_id, transfer_time))
                    subway_transfers += 1
            
            print(f"      지하철 환승: {subway_transfers:,}개")
        except Exception as e:
            print(f"      ⚠️ 도시철도환승정보 로드 실패: {e}")
        
        # 2. 지하철-버스 환승 (같은 역명 기준)
        print("      🚇🚌 지하철-버스 환승 연결...")
        subway_bus_transfers = 0
        
        # 지하철역과 버스정류장 분리
        subway_stops = {sid: s for sid, s in self.stops.items() if s.stop_type == 1}
        bus_stops = {sid: s for sid, s in self.stops.items() if s.stop_type == 0}
        
        # 주요 지하철역 이름과 매칭
        for sub_id, sub_stop in subway_stops.items():
            if sub_stop.zone_id != 'gangnam':
                continue
                
            # 지하철역 이름에서 핵심 부분 추출
            station_name = sub_stop.stop_name.replace('역', '').strip()
            if '선' in station_name:  # "2호선강남" → "강남"
                station_name = station_name.split('선')[-1]
            
            # 가까운 버스정류장 찾기
            for bus_id, bus_stop in bus_stops.items():
                if bus_stop.zone_id != 'gangnam':
                    continue
                    
                # 같은 역명이 포함된 경우
                if station_name in bus_stop.stop_name:
                    # 거리 계산
                    distance = self._calculate_distance(
                        sub_stop.stop_lat, sub_stop.stop_lon,
                        bus_stop.stop_lat, bus_stop.stop_lon
                    )
                    
                    # 200m 이내면 환승 가능
                    if distance <= 200:
                        # 거리에 따른 환승시간 (기본 2분 + 추가시간)
                        transfer_time = 2 + int(distance / 50)  # 50m당 1분 추가
                        
                        if sub_id not in self.transfers:
                            self.transfers[sub_id] = []
                        if bus_id not in self.transfers:
                            self.transfers[bus_id] = []
                            
                        self.transfers[sub_id].append((bus_id, transfer_time))
                        self.transfers[bus_id].append((sub_id, transfer_time))
                        subway_bus_transfers += 2
        
        print(f"      지하철-버스: {subway_bus_transfers:,}개")
        
        # 3. 같은 이름 환승 (버스-버스, 나머지 정류장)
        name_groups = defaultdict(list)
        for stop_id, stop in self.stops.items():
            base_name = stop.stop_name.replace('역', '').replace('.', '').strip()
            name_groups[base_name].append(stop_id)
        
        same_name_transfers = 0
        for group in name_groups.values():
            if len(group) > 1:
                for i in range(len(group)):
                    for j in range(i+1, len(group)):
                        # 이미 연결되어 있는지 확인
                        if group[i] not in self.transfers:
                            self.transfers[group[i]] = []
                        if group[j] not in self.transfers:
                            self.transfers[group[j]] = []
                        
                        # 중복 확인
                        existing_i = [s for s, _ in self.transfers[group[i]] if s == group[j]]
                        existing_j = [s for s, _ in self.transfers[group[j]] if s == group[i]]
                        
                        if not existing_i and not existing_j:
                            self.transfers[group[i]].append((group[j], 0))
                            self.transfers[group[j]].append((group[i], 0))
                            same_name_transfers += 2
        
        print(f"      같은역 환승: {same_name_transfers//2:,}개")
        
        # 2. 도보 환승 - SciPy 없이 구현
        walk_transfers = 0
        
        if SCIPY_AVAILABLE:
            # KDTree 사용 (빠름)
            gangnam_stops = [(sid, s) for sid, s in self.stops.items() if s.zone_id == 'gangnam']
            
            if len(gangnam_stops) > 0:
                coords = [(s.stop_lat, s.stop_lon) for _, s in gangnam_stops]
                stop_ids = [sid for sid, _ in gangnam_stops]
                
                tree = KDTree(coords)
                
                max_distance = self.transfer_config['max_transfer_distance']
                max_walk_time = self.transfer_config['walking_transfer']
                
                for idx, (lat, lon) in enumerate(coords):
                    # 거리를 도단위로 변환 (대략 1도 ≈ 111km)
                    radius_deg = max_distance / 111000
                    nearby = tree.query_ball_point([lat, lon], radius_deg)
                    
                    for j in nearby:
                        if idx != j:
                            dist = self._calculate_distance(
                                lat, lon, coords[j][0], coords[j][1]
                            )
                            
                            if dist <= max_distance:
                                walk_time = max(1, int(dist / 80))  # 80m/분 속도, 최소 1분
                                
                                from_stop = stop_ids[idx]
                                to_stop = stop_ids[j]
                                
                                if from_stop not in self.transfers:
                                    self.transfers[from_stop] = []
                                
                                # 중복 체크
                                existing = [s for s, _ in self.transfers[from_stop] if s == to_stop]
                                if not existing and walk_time <= max_walk_time:
                                    self.transfers[from_stop].append((to_stop, walk_time))
                                    walk_transfers += 1
        else:
            # KDTree 없이 간단한 환승만 (주요역)
            major_stations = ['강남', '역삼', '선릉', '삼성', '교대']
            
            for station_name in major_stations:
                station_stops = [
                    (sid, s) for sid, s in self.stops.items() 
                    if station_name in s.stop_name and s.zone_id == 'gangnam'
                ]
                
                # 같은 역 내 환승
                for i in range(len(station_stops)):
                    for j in range(i+1, len(station_stops)):
                        stop1_id = station_stops[i][0]
                        stop2_id = station_stops[j][0]
                        
                        if stop1_id not in self.transfers:
                            self.transfers[stop1_id] = []
                        if stop2_id not in self.transfers:
                            self.transfers[stop2_id] = []
                        
                        same_stop_time = self.transfer_config['same_stop_transfer']
                        self.transfers[stop1_id].append((stop2_id, same_stop_time))
                        self.transfers[stop2_id].append((stop1_id, same_stop_time))
                        walk_transfers += 2
        
        print(f"      도보 환승: {walk_transfers:,}개")
        
        # 총 환승 개수 계산
        total_transfers = subway_transfers + subway_bus_transfers + same_name_transfers + walk_transfers
        self.stats['transfers'] = total_transfers
        print(f"   ✅ 총 {total_transfers:,}개 환승 생성")
    
    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """거리 계산 (미터)"""
        R = 6371000
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)
        
        a = (np.sin(delta_lat/2)**2 + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * 
             np.sin(delta_lon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    # ========================================================================
    # 4. 모빌리티 데이터 로드
    # ========================================================================
    
    def load_mobility_data(self) -> bool:
        """모빌리티 데이터 로드"""
        print("\n🚲 [4/6] 모빌리티 데이터 로딩...")
        
        mobility_count = 0
        
        # 따릉이
        if self.ttareungee_path and self.ttareungee_path.exists():
            mobility_count += self._load_ttareungee()
        
        # 공유 모빌리티
        if self.shared_mobility_path and self.shared_mobility_path.exists():
            mobility_count += self._load_shared_mobility()
        
        if mobility_count > 0:
            print(f"   ✅ 총 {mobility_count:,}개 모빌리티 포인트")
        else:
            print("   ⚠️ 모빌리티 데이터 없음")
        
        return True
    
    def _load_ttareungee(self) -> int:
        """따릉이 로드"""
        try:
            for encoding in ['cp949', 'utf-8', 'euc-kr']:
                try:
                    df = pd.read_csv(self.ttareungee_path, encoding=encoding)
                    break
                except:
                    continue
            
            # 컬럼 찾기
            lat_col = None
            lon_col = None
            
            for col in df.columns:
                if '위도' in col or 'lat' in col.lower():
                    lat_col = col
                elif '경도' in col or 'lon' in col.lower():
                    lon_col = col
            
            if not lat_col or not lon_col:
                if len(df.columns) > 4:
                    lat_col = df.columns[3]
                    lon_col = df.columns[4]
            
            if lat_col and lon_col:
                df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
                df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
                
                # 강남구 필터링
                mask = (
                    (df[lat_col] >= self.gangnam_bounds['min_lat']) &
                    (df[lat_col] <= self.gangnam_bounds['max_lat']) &
                    (df[lon_col] >= self.gangnam_bounds['min_lon']) &
                    (df[lon_col] <= self.gangnam_bounds['max_lon'])
                )
                
                gangnam_bikes = df[mask]
                
                for idx, row in gangnam_bikes.iterrows():
                    station_id = f"BIKE_{idx}"
                    self.bike_stations[station_id] = {
                        'id': station_id,
                        'lat': float(row[lat_col]),
                        'lon': float(row[lon_col]),
                        'type': 'ttareungee'
                    }
                
                print(f"   ✅ 따릉이: {len(self.bike_stations)}개")
                return len(self.bike_stations)
                
        except Exception as e:
            print(f"   ⚠️ 따릉이 로딩 실패: {e}")
            return 0
    
    def _load_shared_mobility(self) -> int:
        """공유 킥보드/전기자전거 로드"""
        count = 0
        
        try:
            # 킥보드
            kb_path = self.shared_mobility_path / 'kickboards.csv'
            if kb_path.exists():
                kb_df = pd.read_csv(kb_path)
                available = kb_df[kb_df['is_available'] == True] if 'is_available' in kb_df else kb_df
                
                for _, row in available.iterrows():
                    self.shared_vehicles.append({
                        'id': row.get('vehicle_id', f"KB_{_}"),
                        'lat': float(row['lat']),
                        'lon': float(row['lon']),
                        'type': 'kickboard'
                    })
                count += len(available)
                print(f"   ✅ 킥보드: {len(available)}대")
            
            # 전기자전거
            eb_path = self.shared_mobility_path / 'ebikes.csv'
            if eb_path.exists():
                eb_df = pd.read_csv(eb_path)
                available = eb_df[eb_df['is_available'] == True] if 'is_available' in eb_df else eb_df
                
                for _, row in available.iterrows():
                    self.shared_vehicles.append({
                        'id': row.get('vehicle_id', f"EB_{_}"),
                        'lat': float(row['lat']),
                        'lon': float(row['lon']),
                        'type': 'ebike'
                    })
                count += len(available)
                print(f"   ✅ 전기자전거: {len(available)}대")
                
        except Exception as e:
            print(f"   ⚠️ 공유 모빌리티 로딩 실패: {e}")
        
        return count
    
    # ========================================================================
    # 5. 도로망 구축 (수정된 OSMnx 호출)
    # ========================================================================
    
    def build_road_network(self) -> bool:
        """도로망 구축 - 기존 파일 사용"""
        print("\n🗺️ [5/6] 도로망 로딩...")
        
        # 기존 OSM 파일 경로
        pkl_path = Path("gangnam_road_network.pkl")
        graphml_path = Path("gangnam_road_network.graphml")
        
        # 1. pickle 파일 시도
        if pkl_path.exists():
            try:
                with open(pkl_path, 'rb') as f:
                    self.road_graph = pickle.load(f)
                print(f"   ✅ 기존 OSM 로드 (pkl): {self.road_graph.number_of_nodes():,}개 노드")
                print(f"   ✅ 기존 OSM 로드 (pkl): {self.road_graph.number_of_edges():,}개 엣지")
                
                # 속도 정보 추가 (없는 경우만)
                self._add_speed_info_if_missing()
                return True
                
            except Exception as e:
                print(f"   ⚠️ pickle 로드 실패: {e}")
        
        # 2. GraphML 파일 시도
        if graphml_path.exists():
            try:
                self.road_graph = nx.read_graphml(graphml_path)
                print(f"   ✅ 기존 OSM 로드 (graphml): {self.road_graph.number_of_nodes():,}개 노드")
                print(f"   ✅ 기존 OSM 로드 (graphml): {self.road_graph.number_of_edges():,}개 엣지")
                
                # 속도 정보 추가 (없는 경우만)
                self._add_speed_info_if_missing()
                return True
                
            except Exception as e:
                print(f"   ⚠️ GraphML 로드 실패: {e}")
        
        # 3. 기존 파일이 없으면 새로 생성
        print("   ⚠️ 기존 OSM 파일 없음, 새로 생성...")
        if OSMNX_AVAILABLE:
            try:
                import osmnx as ox
                print(f"   OSMnx 버전: {ox.__version__}")
                print(f"   다운로드 범위: {self.gangnam_bounds}")
                
                # OSMnx 2.0+ 문법 - 확장된 범위
                self.road_graph = ox.graph_from_bbox(
                    bbox=(37.460, 37.550, 127.000, 127.140),  # (north, south, west, east)
                    network_type='all',
                    simplify=True
                )
                
                print(f"   ✅ OSM 도로망: {self.road_graph.number_of_nodes():,}개 노드")
                print(f"   ✅ OSM 도로망: {self.road_graph.number_of_edges():,}개 엣지")
                
                # 속도 정보 추가
                self._add_speed_info_if_missing()
                
                # 새로 생성한 네트워크 저장
                try:
                    with open(pkl_path, 'wb') as f:
                        pickle.dump(self.road_graph, f)
                    print(f"   💾 새 OSM 저장: {pkl_path}")
                except Exception as e:
                    print(f"   ⚠️ OSM 저장 실패: {e}")
                
                return True
                    
            except Exception as e:
                print(f"   ❌ OSM 다운로드 실패: {e}")
                
                # 대체: 그리드 네트워크 (확장된 범위)
                print("   🔄 대체 네트워크 생성...")
                self._create_grid_network()
                return True
        else:
            # OSMnx 없으면 그리드 네트워크
            print("   🔄 OSMnx 없음, 그리드 네트워크 생성...")
            self._create_grid_network()
            return True
    
    def _add_speed_info_if_missing(self):
        """속도 정보가 없으면 추가"""
        # 첫 번째 엣지 확인
        if self.road_graph.number_of_edges() > 0:
            first_edge = list(self.road_graph.edges(data=True))[0]
            if 'speed_kmh' not in first_edge[2]:
                print("   🔧 속도 정보 추가 중...")
                for _, _, data in self.road_graph.edges(data=True):
                    highway = data.get('highway', 'residential')
                    if isinstance(highway, list):
                        highway = highway[0]
                    
                    speed_map = {
                        'motorway': 100,
                        'trunk': 80,
                        'primary': 60,
                        'secondary': 50,
                        'tertiary': 40,
                        'residential': 30,
                        'footway': 4,
                        'cycleway': 15,
                        'path': 4
                    }
                    
                    speed = speed_map.get(highway, 30)
                    data['speed_kmh'] = speed
                    
                    if 'length' in data:
                        data['travel_time_min'] = (data['length'] / 1000) / speed * 60
    
    def _create_grid_network(self):
        """그리드 기반 도로망 생성"""
        print("   🏗️ 그리드 네트워크 생성...")
        
        # 20x20 그리드
        G = nx.grid_2d_graph(20, 20)
        
        # 좌표 매핑
        lat_range = self.gangnam_bounds['max_lat'] - self.gangnam_bounds['min_lat']
        lon_range = self.gangnam_bounds['max_lon'] - self.gangnam_bounds['min_lon']
        
        for node in G.nodes():
            i, j = node
            lat = self.gangnam_bounds['min_lat'] + (i/19) * lat_range
            lon = self.gangnam_bounds['min_lon'] + (j/19) * lon_range
            G.nodes[node]['y'] = lat
            G.nodes[node]['x'] = lon
        
        # 거리 추가
        for u, v in G.edges():
            lat1, lon1 = G.nodes[u].get('y', 0), G.nodes[u].get('x', 0)
            lat2, lon2 = G.nodes[v].get('y', 0), G.nodes[v].get('x', 0)
            
            # 간단한 거리 계산
            dist = np.sqrt((lat2-lat1)**2 + (lon2-lon1)**2) * 111000
            G.edges[u, v]['length'] = dist
            G.edges[u, v]['travel_time'] = dist / 1000 / 4 * 60  # 도보 4km/h
        
        self.road_graph = G
        print(f"   ✅ 그리드 네트워크: {G.number_of_nodes()}개 노드")
    
    # ========================================================================
    # 6. 데이터 저장 (NetworkX 최신 버전 대응)
    # ========================================================================
    
    def save_all(self, output_dir: str = "gangnam_raptor_data") -> bool:
        """모든 데이터 저장 - NetworkX 최신 버전 대응"""
        print(f"\n💾 [6/6] 데이터 저장...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # RAPTOR 핵심
            raptor_data = {
                'stops': self.stops,
                'routes': self.routes,
                'trips': self.trips,
                'stop_index_map': self.stop_index_map,
                'index_to_stop': self.index_to_stop,
                'route_stop_sequences': self.route_stop_sequences,
                'route_stop_indices': self.route_stop_indices,
                'timetables': self.timetables,
                'trip_ids_by_route': self.trip_ids_by_route,
                'transfers': self.transfers,
                'stop_routes': self.stop_routes,
                'routes_by_stop': self.routes_by_stop,
                'bike_stations': self.bike_stations,
                'shared_vehicles': self.shared_vehicles,
                'gangnam_bounds': self.gangnam_bounds
            }
            
            with open(output_path / 'raptor_data.pkl', 'wb') as f:
                pickle.dump(raptor_data, f)
            print(f"   ✅ raptor_data.pkl")
            
            # 도로망 저장 - 최신 NetworkX 대응
            if self.road_graph:
                try:
                    # NetworkX 3.0+
                    nx.write_graphml(self.road_graph, output_path / 'road_network.graphml')
                    print(f"   ✅ road_network.graphml")
                except:
                    # 대체: pickle로 저장
                    with open(output_path / 'road_network.pkl', 'wb') as f:
                        pickle.dump(self.road_graph, f)
                    print(f"   ✅ road_network.pkl")
            
            # 메타데이터
            metadata = {
                'created_at': datetime.now().isoformat(),
                'version': '6.1_final',
                'bounds': self.gangnam_bounds,
                'statistics': self.stats,
                'osmnx_available': OSMNX_AVAILABLE,
                'scipy_available': SCIPY_AVAILABLE
            }
            
            with open(output_path / 'metadata.json', 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"   ✅ metadata.json")
            
            return True
            
        except Exception as e:
            print(f"   ❌ 저장 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # ========================================================================
    # 데이터 검증
    # ========================================================================
    
    def validate_data(self) -> bool:
        """데이터 무결성 검증"""
        print("\n🔍 [검증] 데이터 무결성 검사...")
        
        validation_errors = []
        warnings = []
        
        # 1. 기본 데이터 존재 확인
        if not self.stops:
            validation_errors.append("정류장 데이터가 없습니다")
        if not self.routes:
            validation_errors.append("노선 데이터가 없습니다")
        if not self.trips:
            validation_errors.append("운행 데이터가 없습니다")
        
        # 2. 시간표 검증
        invalid_timetables = 0
        for timetable in self.timetables.values():
            for trip_times in timetable:
                # 시간 순서 확인
                for i in range(len(trip_times) - 1):
                    if trip_times[i] >= trip_times[i + 1]:
                        invalid_timetables += 1
                        break
        
        if invalid_timetables > 0:
            warnings.append(f"시간 순서가 맞지 않는 시간표: {invalid_timetables}개")
        
        # 3. 정류장 중복 확인
        stop_names = [stop.stop_name for stop in self.stops.values()]
        duplicate_names = []
        seen_names = set()
        for name in stop_names:
            if name in seen_names:
                duplicate_names.append(name)
            seen_names.add(name)
        
        if duplicate_names:
            warnings.append(f"중복 정류장명: {len(set(duplicate_names))}개")
        
        # 4. 노선-정류장 연결 확인
        orphaned_stops = []
        for stop_id in self.stops:
            if stop_id not in self.stop_routes or not self.stop_routes[stop_id]:
                orphaned_stops.append(stop_id)
        
        if orphaned_stops:
            warnings.append(f"노선이 없는 정류장: {len(orphaned_stops)}개")
        
        # 5. 환승 네트워크 검증
        invalid_transfers = 0
        for from_stop, transfers in self.transfers.items():
            if from_stop not in self.stops:
                invalid_transfers += 1
            for to_stop, _ in transfers:
                if to_stop not in self.stops:
                    invalid_transfers += 1
        
        if invalid_transfers > 0:
            warnings.append(f"잘못된 환승 연결: {invalid_transfers}개")
        
        # 6. 좌표 범위 확인
        out_of_bounds = 0
        for stop in self.stops.values():
            if not (self.gangnam_bounds['min_lat'] <= stop.stop_lat <= self.gangnam_bounds['max_lat'] and
                    self.gangnam_bounds['min_lon'] <= stop.stop_lon <= self.gangnam_bounds['max_lon']):
                if stop.zone_id == 'gangnam':  # 강남으로 분류되었는데 범위 밖
                    out_of_bounds += 1
        
        if out_of_bounds > 0:
            warnings.append(f"좌표 범위 밖 정류장: {out_of_bounds}개")
        
        # 7. 결과 출력
        if validation_errors:
            print("   ❌ 치명적 오류:")
            for error in validation_errors:
                print(f"      - {error}")
            return False
        
        if warnings:
            print("   ⚠️ 경고사항:")
            for warning in warnings:
                print(f"      - {warning}")
        else:
            print("   ✅ 모든 검증 통과")
        
        # 8. 통계 요약
        print(f"\n   📊 검증 통계:")
        print(f"      정류장: {len(self.stops):,}개")
        print(f"      노선: {len(self.routes):,}개")
        print(f"      운행: {len(self.trips):,}개")
        print(f"      환승: {sum(len(t) for t in self.transfers.values()):,}개")
        print(f"      시간표: {sum(len(t) for t in self.timetables.values()):,}개")
        
        return True
    
    # ========================================================================
    # 요약 출력
    # ========================================================================
    
    def print_summary(self):
        """요약"""
        print("\n" + "="*80)
        print("📊 최종 결과 요약")
        print("="*80)
        
        print(f"\n🚇 대중교통:")
        print(f"   총 정류장: {self.stats['total_stops']:,}개")
        print(f"     - 강남구 내부: {self.stats['gangnam_inside_stops']:,}개")
        print(f"     - 강남구 외부: {self.stats['gangnam_outside_stops']:,}개")
        print(f"   노선: {self.stats['total_routes']:,}개")
        print(f"   운행: {self.stats['total_trips']:,}개")
        print(f"   환승: {self.stats['transfers']:,}개")
        
        print(f"\n📅 RAPTOR 구조:")
        print(f"   Route Patterns: {len(self.route_stop_sequences):,}개")
        print(f"   Timetables: {len(self.timetables):,}개")
        
        if self.timetables:
            total_deps = sum(sum(len(t) for t in tt) for tt in self.timetables.values())
            print(f"   총 출발시간: {total_deps:,}개")
        
        print(f"\n🚲 모빌리티:")
        print(f"   따릉이: {len(self.bike_stations)}개")
        print(f"   공유차량: {len(self.shared_vehicles)}개")
        
        if self.road_graph:
            print(f"\n🗺️ 도로망:")
            print(f"   노드: {self.road_graph.number_of_nodes():,}개")
            print(f"   엣지: {self.road_graph.number_of_edges():,}개")
        
        print("\n✅ Part1 완료! Part2에서 RAPTOR 알고리즘 실행 가능")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 실행"""
    print("🚀 강남구 Multi-modal RAPTOR 데이터 구축 시작")
    
    try:
        # 로더 초기화
        loader = GangnamMultiModalRAPTORLoader(
            gtfs_path="cleaned_gtfs_data",
            ttareungee_path="서울시 따릉이대여소 마스터 정보.csv",
            shared_mobility_path="shared_mobility"
        )
        
        # 1. GTFS 로드
        if not loader.load_gtfs_data():
            raise Exception("GTFS 로딩 실패")
        
        # 2. 강남 필터링 (전체 구간 포함)
        if not loader.filter_gangnam_complete():
            raise Exception("강남 필터링 실패")
        
        # 3. RAPTOR 구조
        if not loader.build_raptor_structures():
            raise Exception("RAPTOR 구조 생성 실패")
        
        # 4. 모빌리티
        loader.load_mobility_data()
        
        # 5. 도로망
        loader.build_road_network()
        
        # 6. 데이터 검증
        if not loader.validate_data():
            raise Exception("데이터 검증 실패")
        
        # 7. 저장
        if not loader.save_all("gangnam_raptor_data"):
            raise Exception("저장 실패")
        
        # 8. 요약
        loader.print_summary()
        
        print("\n🎉 성공!")
        
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)