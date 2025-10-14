#!/usr/bin/env python3
"""
Multimodal RAPTOR Web Demo
강남구 지도에서 클릭하여 출발지/도착지 설정하고 경로 탐색
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import sys
import os
import time


# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

# Set page config
st.set_page_config(
    page_title="Multimodal RAPTOR Demo",
    page_icon="🚌",
    layout="wide"
)

# CSS for better styling
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    margin: 0.5rem 0;
}
.route-result {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #007bff;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("🚌 Multimodal RAPTOR Demo")
st.markdown("**강남구 멀티모달 교통 라우팅 시스템**")
st.markdown("지도에서 출발지와 도착지를 클릭하여 최적 경로를 찾아보세요!")

# Gangnam bounds
GANGNAM_BOUNDS = {
    'lat_min': 37.460, 'lat_max': 37.550,
    'lon_min': 127.000, 'lon_max': 127.140
}
GANGNAM_CENTER = [37.505, 127.070]

# Initialize session state
if 'origin' not in st.session_state:
    st.session_state.origin = None
if 'destination' not in st.session_state:
    st.session_state.destination = None
if 'routes' not in st.session_state:
    st.session_state.routes = None

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📍 강남구 지도")
    st.markdown("지도를 클릭하여 출발지와 도착지를 설정하세요")
    
    # Create map
    m = folium.Map(
        location=GANGNAM_CENTER,
        zoom_start=13,
        tiles='OpenStreetMap'
    )
    
    # Add Gangnam district bounds
    folium.Rectangle(
        bounds=[[GANGNAM_BOUNDS['lat_min'], GANGNAM_BOUNDS['lon_min']], 
                [GANGNAM_BOUNDS['lat_max'], GANGNAM_BOUNDS['lon_max']]],
        color='blue',
        fill=False,
        popup='강남구 영역'
    ).add_to(m)
    
    # Add markers if coordinates are set
    if st.session_state.origin:
        folium.Marker(
            st.session_state.origin,
            popup="🚩 출발지",
            icon=folium.Icon(color='green', icon='play')
        ).add_to(m)
    
    if st.session_state.destination:
        folium.Marker(
            st.session_state.destination,
            popup="🏁 도착지",
            icon=folium.Icon(color='red', icon='stop')
        ).add_to(m)
    
    # Display map and capture clicks
    map_data = st_folium(m, width=700, height=500)

with col2:
    st.subheader("⚙️ 설정")
    
    # Display current coordinates
    if st.session_state.origin:
        st.success(f"🚩 출발지: {st.session_state.origin[0]:.4f}, {st.session_state.origin[1]:.4f}")
    else:
        st.info("지도를 클릭하여 출발지를 설정하세요")
    
    if st.session_state.destination:
        st.success(f"🏁 도착지: {st.session_state.destination[0]:.4f}, {st.session_state.destination[1]:.4f}")
    else:
        st.info("지도를 클릭하여 도착지를 설정하세요")
    
    # Preset locations
    st.subheader("📌 빠른 설정")
    presets = {
        "강남역 → 역삼역": [(37.4979, 127.0276), (37.5007, 127.0363)],
        "개포동 → 대치동": [(37.4813, 127.0701), (37.4935, 127.0591)],
        "일원동 → 삼성동": [(37.4847, 127.0828), (37.5115, 127.0595)],
    }
    
    for name, (origin, dest) in presets.items():
        if st.button(name, key=f"preset_{name}"):
            st.session_state.origin = origin
            st.session_state.destination = dest
            st.rerun()
    
    # Clear button
    if st.button("🗑️ 초기화"):
        st.session_state.origin = None
        st.session_state.destination = None
        st.session_state.routes = None
        st.rerun()
    
    # Departure time
    departure_time = st.slider("출발 시간", 6.0, 23.0, 8.0, 0.5)
    departure_str = f"{int(departure_time):02d}:{int((departure_time % 1) * 60):02d}"
    st.write(f"출발 시간: {departure_str}")

# Handle map clicks
if map_data['last_object_clicked_popup']:
    # This handles marker clicks, not map clicks
    pass
elif map_data['last_clicked']:
    lat = map_data['last_clicked']['lat']
    lon = map_data['last_clicked']['lng']
    
    # Check if coordinates are within Gangnam bounds
    if (GANGNAM_BOUNDS['lat_min'] <= lat <= GANGNAM_BOUNDS['lat_max'] and
        GANGNAM_BOUNDS['lon_min'] <= lon <= GANGNAM_BOUNDS['lon_max']):
        
        if not st.session_state.origin:
            st.session_state.origin = [lat, lon]
            st.success("출발지가 설정되었습니다!")
            st.rerun()
        elif not st.session_state.destination:
            st.session_state.destination = [lat, lon]
            st.success("도착지가 설정되었습니다!")
            st.rerun()
        else:
            # Both are set, replace destination
            st.session_state.destination = [lat, lon]
            st.info("도착지가 변경되었습니다!")
            st.rerun()
    else:
        st.error("강남구 영역 내에서만 선택할 수 있습니다!")

# Route search
if st.session_state.origin and st.session_state.destination:
    st.subheader("🚀 경로 탐색")
    
    if st.button("🔍 최적 경로 찾기", type="primary"):
        with st.spinner("PART3 OSM 다익스트라 RAPTOR 알고리즘 실행 중..."):
            try:
                # Import PART1_2 classes to register for pickle loading
                sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
                from PART1_2 import Stop, Route, Trip, Transfer
                # Import and run PART3_OSM_DIJKSTRA
                from PART3_OSM_DIJKSTRA import OSMDijkstraRAPTOR
                
                # Initialize router with explicit data paths
                base_dir = os.path.dirname(__file__)
                router = OSMDijkstraRAPTOR(
                    raptor_data_path=os.path.join(base_dir, "data/processed/gangnam_raptor_data/raptor_data.pkl"),
                    osm_graph_path=os.path.join(base_dir, "data/processed/gangnam_road_network.pkl"),
                    bike_stations_path=os.path.join(base_dir, "data/raw/bike_stations_simple/ttareungee_stations.csv"),
                    pm_density_path=os.path.join(base_dir, "data/processed/grid_pm_data/pm_density_map.json")
                )
                
                # Run routing
                start_time = time.time()
                routes = router.route(
                    origin_lat=st.session_state.origin[0],
                    origin_lon=st.session_state.origin[1],
                    dest_lat=st.session_state.destination[0],
                    dest_lon=st.session_state.destination[1],
                    dep_time=departure_time
                )
                end_time = time.time()
                
                st.session_state.routes = routes
                
                # Display results
                if routes:
                    st.success(f"✅ {len(routes)}개 경로 발견 (실행시간: {end_time - start_time:.2f}초)")
                    
                    # Display metrics
                    best_route = routes[0]
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>⏱️ 소요시간</h4>
                            <h2>{best_route.get('total_time_min', 0):.1f}분</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>💰 비용</h4>
                            <h2>{best_route.get('total_cost_won', 0):,}원</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>🔄 환승</h4>
                            <h2>{best_route.get('n_transfers', 0)}회</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Display all routes
                    st.subheader("📋 경로 상세 정보")
                    
                    for i, route in enumerate(routes, 1):
                        with st.expander(f"경로 {i}: {route.get('total_time_min', 0):.1f}분, {route.get('total_cost_won', 0):,}원"):
                            
                            # Route summary
                            st.markdown(f"""
                            **총 소요시간**: {route.get('total_time_min', 0):.1f}분  
                            **총 비용**: {route.get('total_cost_won', 0):,}원  
                            **환승 횟수**: {route.get('n_transfers', 0)}회  
                            **총 도보**: {route.get('total_walk_m', 0)}m
                            """)
                            
                            # Segments
                            if 'segments' in route:
                                st.markdown("**경로 상세:**")
                                for j, segment in enumerate(route['segments'], 1):
                                    description = segment.get('description', '정보 없음')
                                    st.markdown(f"{j}. {description}")
                else:
                    st.error("❌ 경로를 찾을 수 없습니다")
                    
            except ImportError as e:
                st.error(f"❌ PART3_OSM_DIJKSTRA 모듈을 찾을 수 없습니다: {str(e)}")
                st.error("scripts/ 디렉토리에 PART3_OSM_DIJKSTRA.py 파일이 있는지 확인하세요.")
            except Exception as e:
                st.error(f"❌ 오류 발생: {str(e)}")
                st.error("데이터 파일 경로를 확인하세요.")

# Display cached routes
elif st.session_state.routes:
    st.subheader("📋 이전 검색 결과")
    st.info("새로운 경로를 검색하려면 출발지와 도착지를 설정하세요")

# Footer
st.markdown("---")
st.markdown("""
**🎓 연구 정보**  
- **알고리즘**: PART3 OSM 다익스트라 RAPTOR  
- **작성자**: 김태우 (가천대학교 스마트시티학과)  
- **지도교수**: 여지호  
- **GitHub**: [multimodal-raptor](https://github.com/twtwtiwa05/multimodal-raptor)
""")

# Sidebar with instructions
with st.sidebar:
    st.header("📖 사용법")
    st.markdown("""
    1. **출발지 설정**: 지도를 클릭하여 출발지를 설정하세요
    2. **도착지 설정**: 지도를 다시 클릭하여 도착지를 설정하세요  
    3. **경로 탐색**: "최적 경로 찾기" 버튼을 클릭하세요
    4. **결과 확인**: 여러 경로 옵션과 상세 정보를 확인하세요
    
    💡 **팁**: 빠른 설정 버튼을 사용하여 미리 정의된 경로를 테스트해보세요!
    """)
    
    st.header("🔧 기술 정보")
    st.markdown("""
    - **알고리즘**: OSM 다익스트라 RAPTOR
    - **교통수단**: 버스, 지하철, 따릉이, PM
    - **데이터**: 실제 도로망 기반 거리 계산
    - **성능**: ~7-8초 쿼리 시간
    """)
    
    st.header("📊 예상 결과")
    st.markdown("""
    **강남역 → 역삼역 (1.1km)**:
    - 🛴 PM 직접: **5.5분**, 1,500원
    - 🚶 도보: **15.0분**, 0원  
    - 🚌 대중교통: **10.2분**, 2,760원
    """)