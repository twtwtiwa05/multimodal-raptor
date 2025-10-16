# Multimodal RAPTOR

한국어 | [English](README_EN.md)

서울 강남구의 고성능 멀티모달 교통 라우팅 시스템으로 파동-확산 및 OSM 다익스트라 RAPTOR 알고리즘을 구현합니다.

##  핵심 특징

-  **파동-확산 RAPTOR**: 이중 격자 시스템 (50m + 300m)과 동적 PM 가용성
-  **OSM 다익스트라 RAPTOR**: 직선 거리 대신 실제 도로망 거리 계산
-  **완전한 대중교통 통합**: 버스, 지하철, 따릉이, 전동킥보드(PM)
-  **높은 성능**: 현실적 결과로 ~7-8초 쿼리 시간
-  **정확도**: GTFS 오류율 83%에서 0.33%로 개선

##  빠른 시작

###  온라인 데모 

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/twtwtiwa05/multimodal-raptor/main/app.py)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/twtwtiwa05/multimodal-raptor?quickstart=1)

** 원클릭 실행:**
1. 위 "Open in GitHub Codespaces" 버튼 클릭
2. 자동으로 환경 설정됨
3. 터미널에서: `streamlit run streamlit_app.py`
4. 브라우저에서 강남구 지도 클릭하여
5. 
###  로컬 실행
```bash
# 웹 데모 실행
pip install -r requirements_web.txt
streamlit run streamlit_app.py
```

###  로컬 설치
```bash
# 의존성 설치
pip install -r requirements.txt

# 라우팅 시작
python -m mmraptor.pipeline.route --origin "37.4979,127.0276" --dest "37.5007,127.0363"
```

##  결과 예시

**강남역 → 역삼역 (1.1km)**:
-  PM 직접: **5.5분**, 1,500원
-  도보: **15.0분**, 0원  
-  대중교통: **10.2분**, 2,760원

##  아키텍처

```
mmraptor/
├─ data/           # GTFS, OSM, PM 데이터 로더
├─ graph/          # OSM 다익스트라, 노드 스냅핑
├─ transit/        # RAPTOR 핵심 알고리즘
├─ modes/          # 도보/PM/자전거 모델
└─ pipeline/       # End-to-end 워크플로우
```

##  연구 배경

이 프로젝트는 멀티모달 도시 라우팅을 위한 혁신적 알고리즘을 구현합니다:

- **PART3_WAVE_EXPANSION**: PM 연속주행 모델링과 이중 격자 파동 확산
- **PART3_OSM_DIJKSTRA**: 4km 버퍼 탐색과 실제 도로망 통합
- **기반 연구**: 검증된 PART2 RAPTOR 구현을 기반으로 구축

**작성자**: 김태우 (가천대학교, 스마트시티학과)
**기간**: 2024년 8월-12월

##  성능

- **데이터 규모**: 12K 정류장, 944 노선, 693 자전거 대여소, 50 PM 스테이션
- **범위**: 서울 강남구 (37.46°-37.55°N, 127.00°-127.14°E)
- **쿼리 시간**: 완전한 경로 재구성으로 7-8초
- **정확도**: OSM 기반 실제 도보/주행 거리

## 기술 스택

- **핵심**: Python 3.8+, NetworkX, NumPy, SciPy
- **공간**: OSMnx, KDTree (공간 인덱싱)
- **데이터**: GTFS (한국교통안전공단), 서울시 따릉이 API, 스윙 PM 데이터
- **선택적**: Folium (시각화)

##  사용법

### Python API
```python
from mmraptor import MultimodalRouter

# 라우터 초기화
router = MultimodalRouter(algorithm="osm_dijkstra")

# 경로 탐색
routes = router.route(
    origin=(37.4979, 127.0276),    # 강남역
    destination=(37.5007, 127.0363) # 역삼역
)

# 결과 출력
for i, route in enumerate(routes, 1):
    print(f"경로 {i}: {route['total_time_min']:.1f}분, {route['total_cost_won']:,}원")
    for segment in route['segments']:
        print(f"  - {segment['description']}")
```

### 명령행 인터페이스
```bash
# 기본 라우팅
mmraptor --origin "37.4979,127.0276" --dest "37.5007,127.0363"

# 알고리즘 선택
mmraptor --origin "37.4979,127.0276" --dest "37.5007,127.0363" --algorithm wave_expansion

# 출발 시간 지정
mmraptor --origin "37.4979,127.0276" --dest "37.5007,127.0363" --time 8.5
```

### 예시 실행
```bash
# 강남 지역 예시 테스트
python examples/gangnam_quick.py

# 테스트 실행
pytest tests/
```

##  데이터 구조

```
data/
├─ raw/                     # 원본 소스 데이터
│  ├─ gtfs/                 # 한국 GTFS 대중교통 데이터 (2023년 3월)
│  ├─ bike_stations_simple/ # 서울 따릉이 대여소 (693개)
│  └─ PM_DATA/              # 스윙 전동킥보드 이용 데이터 (9,591건)
├─ processed/               # 정리 및 처리된 데이터
│  ├─ cleaned_gtfs_data/    # BOM 정리된 GTFS CSV 파일
│  ├─ gangnam_raptor_data/  # RAPTOR 데이터 구조 (pickle)
│  ├─ grid_pm_data/         # PM 밀도 지도
│  └─ gangnam_road_network.pkl  # OSM 도로망 (NetworkX)
```

##  데이터 파이프라인

### 1. 원본 데이터 처리
```bash
# GTFS 데이터 정리 (한국어 BOM 인코딩 수정)
python -m mmraptor.data.gtfs_loader data/raw/gtfs

# RAPTOR 구조 구축
python -m mmraptor.data.raptor_builder data/processed/cleaned_gtfs_data
```

### 2. 라우팅 실행
```bash
# OSM 다익스트라 RAPTOR (권장)
python scripts/PART3_OSM_DIJKSTRA.py

# 파동-확산 RAPTOR (권장)
python scripts/PART3_WAVE_EXPANSION_V2.py

# 기존 연구용 (참고)
python scripts/PART2_NEW.py
```

##  설치

### 요구사항
- Python 3.8+
- 메모리: 4GB+ RAM 권장
- 저장공간: 완전한 데이터셋에 ~2GB

### 빠른 설치
```bash
# 저장소 클론
git clone https://github.com/username/multimodal-raptor.git
cd multimodal-raptor

# 패키지 설치
pip install -e .

# 설치 테스트
mmraptor --origin "37.4979,127.0276" --dest "37.5007,127.0363"
```

### 개발자 설정
```bash
# 개발 의존성 포함 설치
pip install -e ".[dev,viz]"

# 테스트 실행
pytest tests/

# 코드 포맷팅
black src/ tests/
ruff check src/ tests/
```

##  검증

예상 결과:
```
=== 강남역 - 역삼역 (근거리) ===
 3개 여정 발견:

💡 여정 1: 5.5분, 1500원 (→08:05 도착, 0라운드)
   1. 🛴 PM 직접: 1077m (5.5분)
      💰 비용: 1500원 | ⏱️ 대기: 1.0분, 주행: 4.5분

💡 여정 2: 15.0분, 0원 (→08:14 도착, 0라운드)
   1. 🚶 도보 직접: 1077m (15.0분)

💡 여정 3: 10.2분, 2760원 (→08:10 도착, 3라운드)
   1. 🚶 액세스 (walk): 08:00 → 08:00 (0.5분)
   2. 🚌 버스 서초09: 강남역9번출구 → 강남역.삼성전자
   3. 🚶 도보 환승: 강남역.삼성전자 → 강남역.역삼세무서 (2.0분)
   4. 🚌 버스 360: 강남역.역삼세무서 → 역삼역.포스코타워역삼
   5. 🚶 이그레스 (bike): 2.2분
```

##  데이터 출처 및 라이선스

### 1. GTFS 대중교통 데이터
- **출처**: 한국교통안전공단 (KTDB)
- **범위**: 서울특별시 강남구
- **날짜**: 2023년 3월
- **라이선스**: 한국 정부 공공데이터

### 2. 서울시 따릉이 대여소
- **출처**: 서울 열린데이터 광장
- **대여소**: 강남구 693개
- **라이선스**: 서울특별시 공공데이터 라이선스

### 3. 스윙 PM (전동킥보드) 데이터
- **출처**: 스윙 (민간기업)
- **주행**: 강남 지역 9,591건 익명화 (2023년 5월)
- **라이선스**: 연구 목적 허가

### 4. 오픈스트리트맵 도로망
- **출처**: 오픈스트리트맵 기여자들
- **범위**: 강남구 전체 도로망
- **라이선스**: ODbL (오픈 데이터베이스 라이선스)

##  기술적 혁신

### 핵심 기술 혁신
1. **파동-확산 알고리즘**: 기존 격자 방식 대비 동적 확산으로 더 정확한 범위 계산
2. **OSM 다익스트라 통합**: 직선 거리 × 1.3 근사치를 실제 도로 거리로 대체
3. **이중 격자 시스템**: 정밀도와 성능의 최적 균형
4. **멀티모달 RAPTOR**: 모든 교통수단의 완전한 통합

### 데이터 규모
- **대중교통 정류장**: 12,064개 (강남 지역 9,404개)
- **대중교통 노선**: 944개
- **예정된 운행**: 44,634개
- **환승 연결**: 45,406개
- **가상 PM 정거장**: 50개 (500대 차량)
- **따릉이 대여소**: 693개
- **OSM 도로 노드**: 8,311개
- **OSM 도로 엣지**: 22,541개

### 정확도 개선
- **GTFS 데이터 오류율**: 83%에서 0.33%로 감소
- **도보 거리 정확도**: OSM 기반 실제 경로 사용
- **시간 계산**: 0.5분 최소값으로 현실적 계산
- **중복 경로 제거**: 동일 시그니처 경로 자동 필터링

##  향후 과제

1. **실시간 통합**
   - 실시간 대중교통 지연 정보
   - 동적 PM 가용성 업데이트
   - 교통 상황 반영

2. **AI 기반 개인화**
   - 사용자 행동 패턴 학습
   - 상황 인식 경로 추천
   - 예측 기반 최적화

3. **확장성**
   - 서울 전체 확장
   - 다중 도시 지원
   - 클라우드 네이티브 배포

##  기여

이 프로젝트는 오픈 소스 기여를 환영합니다:
- 알고리즘 최적화 및 혁신
- 데이터 품질 개선
- 새로운 교통수단 통합
- 사용자 인터페이스 개발

##  라이선스

MIT 라이선스 - 자세한 내용은 [LICENSE](LICENSE) 파일 참조

##  감사의 말

- **한국교통안전공단(KTDB)** - 고품질 GTFS 데이터
- **서울특별시** - 공공 자전거 대여소 데이터  
- **스윙** - 익명화된 PM 이용 데이터
- **오픈스트리트맵 기여자들** - 상세한 도로망 데이터
- **가천대학교 스마트시티학과** - 연구 환경 지원

---

##  연락처

**작성자**: 김태우  
**대학**: 가천대학교  
**소속**: 스마트시티학과 학부 연구생  
**이메일**: twdaniel@gachon.ac.kr  



---

**프로젝트 기간**: 2024년 8월 - 2024년 12월  
**연구 분야**: 멀티모달 교통, 도시 이동성, 알고리즘 혁신  


**키워드**: RAPTOR, 파동-확산, OSM 다익스트라, 멀티모달 라우팅, 대중교통, 공유 모빌리티, 서울


