# 멀티모달 RAPTOR 알고리즘 구현 발표

## 시스템 개요

### 강남구 멀티모달 교통 라우팅 시스템
- **4개 교통수단 통합**: 버스, 지하철, 따릉이, PM(킥보드)
- **실제 도로망 기반**: OSM 데이터로 정확한 거리/시간 계산
- **RAPTOR 알고리즘**: 대중교통 최적화 + 퍼스널 모빌리티 확장

---

## 핵심 알고리즘 구현

### 1. PART3_OSM_DIJKSTRA: OSM 기반 다익스트라 접근법

#### 핵심 아이디어
```python
# OSM 도로망에서 실제 거리 기반 액세스/이그레스 계산
def dijkstra_access_egress(self, origin, max_time_sec):
    # 1. 출발지를 OSM 노드에 매핑
    origin_node = self._find_nearest_osm_node(origin)
    
    # 2. 다익스트라로 시간 등고선 생성  
    distances = {origin_node: 0}
    heap = [(0, origin_node)]
    
    while heap:
        current_time, node = heapq.heappop(heap)
        if current_time > max_time_sec:
            continue
            
        for neighbor, edge_time in self.osm_graph[node].items():
            new_time = current_time + edge_time
            if new_time < distances.get(neighbor, float('inf')):
                distances[neighbor] = new_time
                heapq.heappush(heap, (new_time, neighbor))
    
    # 3. 도달 가능한 정류장들 수집
    return self._find_reachable_stops(distances)
```

#### 구현 핵심
- **OSM 노드 매핑**: 좌표 → 실제 도로 교차점
- **엣지 가중치**: 실제 도로 거리 ÷ 이동 속도
- **다중 모드**: 도보(1.2m/s), PM(4.0m/s), 따릉이(3.5m/s)

### 2. PART3_WAVE_EXPANSION_V2: 이중 격자 파동 확산

#### 핵심 아이디어  
```python
# 50m 정밀격자 + 300m 거대격자 이중 시스템
class DualGridWaveExpansion:
    def wave_expansion_search(self, origin, max_waves=14):
        # 1단계: 출발지 50m 격자에서 PM 밀도 계산
        fine_grid = self._find_origin_grid(origin)
        pm_wait_time = 1.5 / fine_grid.density
        
        # 2단계: 300m 격자로 변환하여 파동 확산
        coarse_grid = self.fine_to_coarse_map[fine_grid.id]
        frontier = [(pm_wait_time, coarse_grid, 0, [coarse_grid])]
        
        # 3단계: 14번 파동으로 4.2km 반경 탐색
        for wave in range(max_waves):
            new_frontier = []
            for time, grid_id, distance, path in frontier:
                # 8방향 이웃 격자로 확산
                for neighbor in self._get_8_neighbors(grid_id):
                    ride_time = 300 / PM_SPEED  # 1분 고정
                    new_time = time + ride_time
                    new_path = path + [neighbor]
                    new_frontier.append((new_time, neighbor, distance+300, new_path))
            frontier = new_frontier
        
        # 4단계: 격자 → 정류장 매핑
        return self._map_grids_to_stops(frontier)
```

#### 구현 핵심
- **이중 격자**: 50m(밀도) + 300m(탐색) 역할 분리
- **파동 확산**: BFS 방식으로 동심원 확장
- **공간 프루닝**: 300m 셀당 최대 1개 시드 유지

---

## PM(킥보드) 통합 구현

### PM 밀도 맵 구조
```python
# 50m x 50m 격자로 강남구 전체 분할
class PMDensityMap:
    def __init__(self, pm_data_path):
        # 실제 Swing 수요 데이터 로드
        with open(pm_data_path) as f:
            swing_data = json.load(f)
        
        # 강남구를 50m 격자로 분할 (총 ~15,000개 격자)
        self.grid_bounds = {
            'min_lat': 37.460, 'max_lat': 37.550,
            'min_lon': 127.000, 'max_lon': 127.140
        }
        
        # 격자별 PM 밀도 계산
        for grid_id, demand in swing_data.items():
            row, col = self._parse_grid_id(grid_id)
            lat, lon = self._grid_to_coordinates(row, col)
            
            # 수요 → 밀도 변환 (정규화)
            density = min(0.95, max(0.05, demand / MAX_DEMAND))
            self.fine_grids[grid_id] = FineGrid(
                grid_id=grid_id, lat=lat, lon=lon, density=density
            )
```

### PM 대기시간 계산 알고리즘
```python
def calculate_pm_wait_time(self, grid_density):
    # 밀도가 높을수록 대기시간 짧음 (역비례)
    if grid_density < MIN_PM_DENSITY:
        return float('inf')  # PM 없음
    
    # 경험적 공식: 1.5분 / 밀도
    raw_wait = 1.5 / grid_density
    
    # 현실적 범위로 클램핑
    return max(MIN_PM_WAIT_MIN,  # 최소 0.5분
              min(raw_wait, MAX_PM_WAIT_MIN))  # 최대 8분
```

### PM 비용 모델
```python
def calculate_pm_cost(self, distance_m, ride_time_min):
    base_fare = 1000  # 기본료 1000원
    per_minute_fare = 150  # 분당 150원
    
    # 실제 구현에서 사용하는 공식
    return base_fare + int(ride_time_min * per_minute_fare)
```

---

## 따릉이 시스템 통합 구현

### 따릉이 대여소 데이터 구조
```python
# 693개 실제 대여소 데이터 로드
def load_bike_stations(self, csv_path):
    bike_df = pd.read_csv(csv_path)
    self.bike_stations = []
    
    for _, row in bike_df.iterrows():
        # 강남구 범위 내 대여소만 필터링
        if self._is_in_gangnam_bounds(row['lat'], row['lon']):
            station = BikeStation(
                station_id=row['station_id'],
                station_name=row['station_name'], 
                lat=row['lat'],
                lon=row['lon'],
                n_bikes=row['n_bikes'],  # 실시간 재고
                osm_node_id=self._find_nearest_osm_node(row['lat'], row['lon'])
            )
            self.bike_stations.append(station)
```

### 도킹 제약 처리 알고리즘
```python
def find_bike_access_options(self, origin, max_walk_time):
    options = []
    
    for station in self.bike_stations:
        # 1. 출발지 → 대여소 도보 시간
        walk_dist = haversine_distance(origin, (station.lat, station.lon))
        walk_time = walk_dist / WALK_SPEED_MPS
        
        if walk_time > max_walk_time:
            continue
            
        # 2. 자전거 가용성 확인
        if station.n_bikes <= 0:
            continue  # 자전거 없음
            
        # 3. 픽업 시간 추가 (30초 고정)
        total_access_time = walk_time + BIKE_PICKUP_TIME_SEC
        
        options.append(AccessResult(
            stop_id=station.station_id,
            stop_coords=(station.lat, station.lon),
            access_time_sec=total_access_time,
            mode='bike',
            mode_details={'station_name': station.station_name}
        ))
    
    return options
```

### 따릉이 경로 시간 계산
```python
def calculate_bike_journey_time(self, start_station, end_station):
    # OSM 그래프에서 실제 자전거 경로 계산
    bike_path = self._find_bike_path_osm(
        start_station.osm_node_id, 
        end_station.osm_node_id
    )
    
    # 실제 도로 거리 합산
    total_distance = sum(edge['length'] for edge in bike_path)
    
    # 자전거 속도로 소요시간 계산
    ride_time = total_distance / BIKE_SPEED_MPS
    
    return ride_time + BIKE_PICKUP_TIME_SEC + BIKE_RETURN_TIME_SEC
```

---

## RAPTOR 확장 메커니즘

### 기존 RAPTOR vs 멀티모달 RAPTOR

#### 기존 RAPTOR (대중교통 전용)
```
정류장 → 노선 → 정류장 → 노선 → ...
```

#### 확장된 멀티모달 RAPTOR
```
출발지 → [PM/따릉이] → 정류장 → 노선 → 정류장 → [PM/따릉이] → 도착지
       ↑ Access Layer      RAPTOR Core     Egress Layer ↑
```

---

## 🎯 두 알고리즘 비교

### PART3_OSM_DIJKSTRA (정밀도 우선)

**장점:**
- ✅ 실제 도로망 기반 정확한 거리
- ✅ 복잡한 도로 구조 완벽 반영
- ✅ 높은 정확도 (오차율 5% 이내)

**단점:**
- ⏱️ 긴 쿼리 시간 (29-32초)
- 💾 높은 메모리 사용량
- 🔄 복잡한 데이터 구조

### PART3_WAVE_EXPANSION_V2 (성능 우선)

**장점:**
- ⚡ 빠른 쿼리 시간 (2-5초) 
- 📊 효율적인 격자 시스템
- 🌊 확장 가능한 파동 탐색

**단점:**
- 📍 격자 단위 근사 (정확도 15% 오차)
- 🔧 복잡한 튜닝 필요
- 📐 직선거리 기반 추정

---

## 핵심 구현 아이디어

### 1. 액세스/이그레스 계층 분리
```python
# 출발지 → 대중교통 진입점
access_options = find_pm_bike_access(origin, max_time=15min)

# 대중교통 진입점 → 도착지  
egress_options = find_pm_bike_egress(dest_stations, dest)

# RAPTOR에 시드로 투입
raptor_seeds = {stop_id: (arrival_time, cost) for stop_id in access_options}
```

### 2. 실시간 PM 밀도 반영
```python
def calculate_pm_wait_time(grid_density):
    if grid_density < MIN_DENSITY:
        return float('inf')  # PM 없음
    return max(0.5, min(8.0, 1.5 / grid_density))
```

### 3. 다중 교통수단 비용 모델
```python
# 통합 비용 계산
total_cost = pm_cost + transit_cost + bike_cost
total_time = pm_time + transit_time + bike_time + transfer_time
```

---

## 성능 벤치마크

### 쿼리 성능 비교
| 알고리즘 | 평균 쿼리시간 | 메모리 사용량 | 정확도 |
|---------|-------------|-------------|--------|
| OSM_DIJKSTRA | 29-32초 | ~500MB | 95% |
| WAVE_EXPANSION | 2-5초 | ~200MB | 85% |
| 단순 직선거리 | 0.1초 | ~50MB | 60% |

### 경로 품질
- **경로 다양성**: 4-6개 최적 경로 제공
- **환승 최적화**: 불필요한 환승 제거
- **비용 효율성**: PM vs 대중교통 비용 균형

---

## 실제 사용 사례

### 강남역 → 역삼역 (1.1km)
```
1. PM 직접: 5.5분, 1,500원 🛴
2. 도보: 15.0분, 0원 🚶  
3. 대중교통: 10.2분, 2,760원 🚌
4. 복합경로: 8.1분, 2,100원 🚶→🚌→🛴
```

### 시스템 장점
- **사용자 선택권**: 시간 vs 비용 vs 편의성
- **실시간 반영**: PM 가용성, 교통 상황
- **seamless 연결**: 교통수단 간 자연스러운 환승

---

## 기술적 의의

### 1. RAPTOR 알고리즘 확장
- 기존 대중교통 전용 → 멀티모달 지원
- 액세스/이그레스 계층 아키텍처
- 실시간 PM 가용성 통합

### 2. 실제 도로망 활용
- OSM 기반 정확한 거리 계산
- 격자 vs 그래프 기반 접근법 비교
- 성능과 정확도 트레이드오프 해결

### 3. 도시 교통 패러다임 변화
- 소유 → 공유 모빌리티
- 단일 → 멀티모달 경로
- 정적 → 동적 라우팅

---

## 결론

### 핵심 기여점
1. **RAPTOR 멀티모달 확장**: 이론적 기여
2. **PM 통합 메커니즘**: 실용적 솔루션  
3. **성능-정확도 밸런스**: 두 가지 접근법 제시

### 향후 발전 방향
- 실시간 교통 정보 통합
- 머신러닝 기반 수요 예측
- 탄소 배출량 고려 라우팅

---

## 라이브 데모

### Streamlit 웹 인터페이스
- 실시간 지도 클릭 입력
- 4개 교통수단 통합 결과
- 상세 경로 정보 및 비용 분석

**GitHub**: https://github.com/twtwtiwa05/multimodal-raptor