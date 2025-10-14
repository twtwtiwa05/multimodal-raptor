# 설치 가이드

한국어 | [English](INSTALL_EN.md)

##  빠른 설치

```bash
# 저장소 클론
git clone https://github.com/username/multimodal-raptor.git
cd multimodal-raptor

# 패키지 설치
pip install -e .

# 설치 테스트
mmraptor --origin "37.4979,127.0276" --dest "37.5007,127.0363"
```

##  요구사항

- **Python**: 3.8+ 
- **메모리**: 4GB+ RAM 권장
- **저장공간**: 완전한 데이터셋에 ~2GB
- **운영체제**: Linux, macOS, Windows

##  개발자 설정

```bash
# 개발 기능 포함하여 클론
git clone https://github.com/username/multimodal-raptor.git
cd multimodal-raptor

# 가상 환경 생성
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate     # Windows

# 개발 의존성 설치
pip install -e ".[dev,viz]"

# 테스트 실행
pytest tests/

# 코드 포맷팅
black src/ tests/
ruff check src/ tests/
```

##  데이터 설정

저장소에는 필요한 모든 데이터 파일이 포함되어 있습니다:

### 전처리된 데이터 (바로 사용 가능)
- ✅ **RAPTOR 구조**: `data/processed/gangnam_raptor_data/`
- ✅ **OSM 도로망**: `data/processed/gangnam_road_network.pkl`
- ✅ **PM 밀도 지도**: `data/processed/grid_pm_data/`
- ✅ **정리된 GTFS**: `data/processed/cleaned_gtfs_data/`

### 원본 데이터 (참고용)
- 📁 **원본 GTFS**: `data/raw/gtfs/`
- 📁 **자전거 대여소**: `data/raw/bike_stations_simple/`
- 📁 **PM 이용 데이터**: `data/raw/PM_DATA/`

##  설치 확인

```python
# 기본 기능 테스트
from mmraptor import MultimodalRouter

router = MultimodalRouter()
routes = router.route(
    origin=(37.4979, 127.0276),    # 강남역
    destination=(37.5007, 127.0363) # 역삼역
)

print(f"{len(routes)}개 경로 발견")
print(f"최적 경로: {routes[0]['total_time_min']:.1f}분")
```

예상 출력:
```
3개 경로 발견
최적 경로: 5.5분
```

## 🔄 데이터 재구축 (선택사항)

처음부터 데이터를 재구축하려면:

```bash
# 1. GTFS 데이터 정리
python -m mmraptor.data.gtfs_loader data/raw/gtfs

# 2. RAPTOR 구조 구축  
python -m mmraptor.data.raptor_builder data/processed/cleaned_gtfs_data

# 3. 예시 실행
python examples/gangnam_quick.py
```

##  문제 해결

### Import 오류
```bash
# 스크립트가 Python 경로에 있는지 확인
export PYTHONPATH="${PYTHONPATH}:$(pwd)/scripts"

# 또는 개발 모드로 설치
pip install -e .
```

### 메모리 문제
```bash
# 대용량 데이터셋의 경우 메모리 한계 증가
export MEMORY_LIMIT=8GB
python examples/gangnam_quick.py
```

### 파일을 찾을 수 없음
```bash
# 데이터 파일 존재 확인
ls -la data/processed/gangnam_raptor_data/raptor_data.pkl
ls -la data/processed/gangnam_road_network.pkl

# 작업 디렉토리 확인
pwd  # multimodal-raptor/ 디렉토리에 있어야 함
```

### 성능 문제
```bash
# 작은 테스트 데이터셋 사용
python examples/gangnam_quick.py --limit 100

# 디버그 로깅 활성화
export LOG_LEVEL=DEBUG
python examples/gangnam_quick.py
```

##  플랫폼별 참고사항

### Windows
```cmd
# 경로에 백슬래시 사용
set PYTHONPATH=%PYTHONPATH%;%CD%\scripts

# 가상 환경 활성화
venv\Scripts\activate
```

### macOS
```bash
# Xcode 명령행 도구 설치 필요할 수 있음
xcode-select --install

# Apple Silicon Mac의 경우
pip install --upgrade pip setuptools wheel
```

### Linux
```bash
# 공간 라이브러리 설치 (Ubuntu/Debian)
sudo apt-get install libspatialindex-dev libgeos-dev

# 공간 라이브러리 설치 (CentOS/RHEL)
sudo yum install spatialindex-devel geos-devel
```

##  Docker (선택사항)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN pip install -e .

CMD ["mmraptor", "--help"]
```

```bash
# 빌드 및 실행
docker build -t multimodal-raptor .
docker run multimodal-raptor mmraptor --origin "37.4979,127.0276" --dest "37.5007,127.0363"
```

##  다음 단계

-  [사용자 가이드](README.md#사용법) 읽기
-  [테스트 케이스](examples/gangnam_quick.py) 실행
-  [연구 배경](CHANGELOG.md) 탐색
-  [API 문서](src/mmraptor/) 확인

## 🆘 도움 받기

-  **버그 신고**: GitHub에서 이슈 열기
-  **질문**: 기존 이슈 확인하거나 토론 시작
- **연구 협업**: twdaniel@gachon.ac.kr 연락
-  **문서**: 이 가이드 개선에 기여

##  일반적인 사용 패턴

### 1. 연구자용
```bash
# 알고리즘 비교
python scripts/PART3_OSM_DIJKSTRA.py
python scripts/PART3_WAVE_EXPANSION_V2.py

# 성능 분석
python examples/gangnam_quick.py
pytest tests/ -v
```

### 2. 개발자용
```bash
# 패키지 설치
pip install -e ".[dev]"

# 코드 품질 검사
black src/ tests/
ruff check src/ tests/
mypy src/

# 테스트 실행
pytest tests/ --cov=src/mmraptor
```

### 3. 일반 사용자용
```bash
# 간단한 라우팅
mmraptor --origin "37.4979,127.0276" --dest "37.5007,127.0363"

# Python 스크립트에서 사용
from mmraptor import MultimodalRouter
router = MultimodalRouter()
```

##  성능 최적화

### 메모리 사용량 줄이기
```python
# 작은 버퍼 사용
router = MultimodalRouter(
    max_access_time_sec=10*60,  # 10분으로 제한
    max_egress_time_sec=15*60   # 15분으로 제한
)
```

### 쿼리 속도 향상
```python
# 캐시된 데이터 사용
router = MultimodalRouter()
# 첫 번째 쿼리가 느릴 수 있음 (데이터 로딩)
# 이후 쿼리는 빨라짐
```

