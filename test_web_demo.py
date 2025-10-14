#!/usr/bin/env python3
"""
웹 데모 테스트 스크립트
GitHub에 올리기 전에 실제로 작동하는지 확인
"""

import sys
import os

# Add both current directory and scripts directory to path
current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'scripts'))
sys.path.append(os.path.join(current_dir, '..'))  # Parent directory for original modules

def test_part3_import():
    """PART3_OSM_DIJKSTRA import 테스트"""
    try:
        # Try importing from scripts directory first
        from PART3_OSM_DIJKSTRA import OSMDijkstraRAPTOR
        print("✅ PART3_OSM_DIJKSTRA import 성공")
        return True
    except ImportError as e:
        print(f"❌ PART3_OSM_DIJKSTRA import 실패: {e}")
        return False

def test_data_files():
    """필수 데이터 파일 존재 확인"""
    required_files = [
        "data/processed/gangnam_raptor_data/raptor_data.pkl",
        "data/processed/gangnam_road_network.pkl", 
        "data/raw/bike_stations_simple/ttareungee_stations.csv",
        "data/processed/grid_pm_data/pm_density_map.json"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - 파일 없음")
            all_exist = False
    
    return all_exist

def test_router_initialization():
    """라우터 초기화 테스트"""
    try:
        from PART3_OSM_DIJKSTRA import OSMDijkstraRAPTOR
        
        router = OSMDijkstraRAPTOR(
            raptor_data_path="data/processed/gangnam_raptor_data/raptor_data.pkl",
            osm_graph_path="data/processed/gangnam_road_network.pkl",
            bike_stations_path="data/raw/bike_stations_simple/ttareungee_stations.csv",
            pm_density_path="data/processed/grid_pm_data/pm_density_map.json"
        )
        print("✅ 라우터 초기화 성공")
        return router
    except Exception as e:
        print(f"❌ 라우터 초기화 실패: {e}")
        return None

def test_simple_routing():
    """간단한 라우팅 테스트"""
    router = test_router_initialization()
    if not router:
        return False
    
    try:
        # 강남역 → 역삼역 테스트
        routes = router.route(
            origin_lat=37.4979, origin_lon=127.0276,  # 강남역
            dest_lat=37.5007, dest_lon=127.0363,      # 역삼역
            dep_time=8.0
        )
        
        if routes and len(routes) > 0:
            best = routes[0]
            print(f"✅ 라우팅 성공: {best.get('total_time_min', 0):.1f}분, {best.get('total_cost_won', 0)}원")
            return True
        else:
            print("❌ 라우팅 실패: 경로를 찾을 수 없음")
            return False
            
    except Exception as e:
        print(f"❌ 라우팅 실패: {e}")
        return False

def main():
    """전체 테스트 실행"""
    print("🧪 웹 데모 테스트 시작")
    print("=" * 50)
    
    tests = [
        ("Import 테스트", test_part3_import),
        ("데이터 파일 테스트", test_data_files),
        ("라우팅 테스트", test_simple_routing),
    ]
    
    passed = 0
    for name, test_func in tests:
        print(f"\n{name}:")
        if test_func():
            passed += 1
        
    print(f"\n📊 테스트 결과: {passed}/{len(tests)} 통과")
    
    if passed == len(tests):
        print(" 모든 테스트 통과! 깃허브에 올려도 됩니다!")
        return True
    else:
        print(" 일부 테스트 실패. 문제를 해결 후 다시 테스트하세요.")
        return False

if __name__ == "__main__":
    main()