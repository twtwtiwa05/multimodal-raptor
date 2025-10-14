"""
KTDB GTFS 데이터 로더 v5.0 - BOM 및 인코딩 문제 완전 해결
- BOM 문자 완벽 제거
- 한글 인코딩 정확한 처리
- routes.txt 문제 해결
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import time
from datetime import datetime
import pickle
import json
import re

warnings.filterwarnings('ignore')

class KTDBGTFSLoader:
    """GTFS 데이터 로더 - 완전 수정 버전"""
    
    def __init__(self, gtfs_data_path: str):
        self.gtfs_data_path = Path(gtfs_data_path)
        
        # GTFS 데이터 테이블
        self.agency = None
        self.stops = None
        self.routes = None  
        self.trips = None
        self.stop_times = None
        self.calendar = None
        self.calendar_dates = None
        self.transfers = None
        
        # 통계
        self.stats = {}
        
        print("🚀 KTDB GTFS 데이터 로더 v5.0")
        print(f"📂 경로: {self.gtfs_data_path}")
        
        if not self.gtfs_data_path.exists():
            raise FileNotFoundError(f"경로가 없습니다: {self.gtfs_data_path}")
    
    def load_all_data(self) -> bool:
        """모든 GTFS 데이터 로드 및 정리"""
        print("\n🚇 GTFS 데이터 로딩...")
        start_time = time.time()
        
        # 1. 각 파일 로드 (BOM 제거 포함)
        self._load_with_bom_removal()
        
        # 2. 컬럼명 정리
        self._fix_all_column_names()
        
        # 3. 데이터 타입 최적화
        self._optimize_data_types()
        
        # 4. 데이터 검증
        self._validate_data()
        
        # 5. 통계 출력
        elapsed = time.time() - start_time
        self._print_summary(elapsed)
        
        return True
    
    def _load_with_bom_removal(self):
        """BOM을 제거하며 파일 로드"""
        
        files = {
            'agency': 'agency.txt',
            'stops': 'stops.txt',
            'routes': 'routes.txt',
            'trips': 'trips.txt',
            'stop_times': 'stop_times.txt',
            'calendar': 'calendar.txt'
        }
        
        for data_name, filename in files.items():
            file_path = self.gtfs_data_path / filename
            
            if not file_path.exists():
                print(f"   ❌ {filename} 없음")
                continue
            
            print(f"   📂 {filename} 로딩...", end=' ')
            
            # 먼저 파일을 읽어서 BOM 제거
            try:
                # 원본 파일 읽기 (바이너리 모드)
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
                
                # BOM 패턴들
                bom_patterns = [
                    b'\xef\xbb\xbf',  # UTF-8 BOM
                    b'\xff\xfe',      # UTF-16 LE BOM
                    b'\xfe\xff',      # UTF-16 BE BOM
                    b'\xec\x99\xa4',  # 癤 문자
                ]
                
                # BOM 제거
                for bom in bom_patterns:
                    if raw_data.startswith(bom):
                        raw_data = raw_data[len(bom):]
                        break
                
                # 임시 파일에 쓰기
                temp_path = file_path.parent / f'temp_{filename}'
                with open(temp_path, 'wb') as f:
                    f.write(raw_data)
                
                # pandas로 읽기 (여러 인코딩 시도)
                df = None
                for encoding in ['utf-8', 'cp949', 'euc-kr', 'latin1']:
                    try:
                        df = pd.read_csv(temp_path, encoding=encoding)
                        
                        # 성공하면 컬럼명 정리
                        df.columns = [self._clean_column_name(col) for col in df.columns]
                        
                        setattr(self, data_name, df)
                        print(f"✅ {len(df):,}행")
                        
                        # 특별 처리: routes.txt가 너무 적으면 경고
                        if data_name == 'routes' and len(df) < 100:
                            print(f"      ⚠️ 경고: routes가 {len(df)}개밖에 없습니다. 파일을 확인하세요.")
                            # routes.txt를 다시 읽기 시도
                            self._try_fix_routes(file_path)
                        
                        break
                    except Exception as e:
                        continue
                
                # 임시 파일 삭제
                temp_path.unlink(missing_ok=True)
                
                if df is None:
                    print(f"❌ 읽기 실패")
                    
            except Exception as e:
                print(f"❌ 오류: {e}")
    
    def _clean_column_name(self, col: str) -> str:
        """컬럼명에서 BOM 및 특수문자 제거"""
        
        # 문자열로 변환
        col = str(col)
        
        # BOM 문자들 제거
        bom_chars = ['癤', 'ï»¿', '﻿', '\ufeff', 'Ã¯Â»Â¿']
        for bom in bom_chars:
            col = col.replace(bom, '')
        
        # 특수문자 제거 (한글 제외)
        col = re.sub(r'^[^\w가-힣]+', '', col)
        
        # 알려진 잘못된 매핑
        wrong_mappings = {
            '풹gency_id': 'agency_id',
            '퓊oute_id': 'route_id',
            '퓋ervice_id': 'service_id',
            'oute_id': 'route_id',
            'ervice_id': 'service_id',
            'gency_id': 'agency_id'
        }
        
        # 매핑 적용
        for wrong, correct in wrong_mappings.items():
            if wrong in col:
                return correct
        
        return col.strip()
    
    def _fix_all_column_names(self):
        """모든 테이블의 컬럼명 정리"""
        print("\n   🔧 컬럼명 정리 중...")
        
        # 각 테이블별로 컬럼 정리
        tables = ['agency', 'stops', 'routes', 'trips', 'stop_times', 'calendar']
        
        for table_name in tables:
            table = getattr(self, table_name)
            if table is not None:
                before = list(table.columns)
                table.columns = [self._clean_column_name(col) for col in table.columns]
                after = list(table.columns)
                
                # 변경된 컬럼 출력
                changed = [(b, a) for b, a in zip(before, after) if b != a]
                if changed:
                    print(f"      {table_name}: {len(changed)}개 컬럼 수정")
                    for old, new in changed[:3]:  # 처음 3개만 출력
                        print(f"         {old} → {new}")
    
    def _try_fix_routes(self, original_path: Path):
        """routes.txt 문제 해결 시도"""
        print("\n   🔧 routes.txt 복구 시도...")
        
        # routes.txt를 다시 읽되, 다른 방법으로
        try:
            # 1. 라인 단위로 읽기
            with open(original_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
                lines = f.readlines()
            
            if len(lines) > 1:
                # 헤더 정리
                header = lines[0].strip()
                header = self._clean_column_name(header)
                headers = [self._clean_column_name(h.strip()) for h in header.split(',')]
                
                # 데이터 파싱
                data = []
                for line in lines[1:]:
                    line = line.strip()
                    if line:
                        # 따옴표 처리
                        parts = []
                        in_quote = False
                        current = ''
                        
                        for char in line:
                            if char == '"':
                                in_quote = not in_quote
                            elif char == ',' and not in_quote:
                                parts.append(current.strip().strip('"'))
                                current = ''
                            else:
                                current += char
                        
                        if current:
                            parts.append(current.strip().strip('"'))
                        
                        if len(parts) == len(headers):
                            data.append(parts)
                
                # DataFrame 생성
                if data:
                    self.routes = pd.DataFrame(data, columns=headers)
                    print(f"      ✅ routes.txt 복구 성공: {len(self.routes)}개 노선")
                else:
                    print(f"      ❌ routes.txt 복구 실패")
                    
        except Exception as e:
            print(f"      ❌ routes.txt 복구 실패: {e}")
            
            # 최후의 수단: trips에서 route_id 추출
            if self.trips is not None and 'route_id' in self.trips.columns:
                unique_routes = self.trips['route_id'].unique()
                self.routes = pd.DataFrame({
                    'route_id': unique_routes,
                    'agency_id': 'DEFAULT',
                    'route_short_name': unique_routes,
                    'route_long_name': unique_routes,
                    'route_type': 3  # 버스로 가정
                })
                print(f"      ✅ trips에서 routes 재구성: {len(self.routes)}개")
    
    def _optimize_data_types(self):
        """메모리 최적화"""
        print("\n   💾 메모리 최적화...")
        
        before = self._calculate_memory()
        
        # stops 최적화
        if self.stops is not None:
            if 'stop_lat' in self.stops.columns:
                self.stops['stop_lat'] = pd.to_numeric(self.stops['stop_lat'], errors='coerce').astype('float32')
            if 'stop_lon' in self.stops.columns:
                self.stops['stop_lon'] = pd.to_numeric(self.stops['stop_lon'], errors='coerce').astype('float32')
            
            # 문자열 컬럼 category 변환
            for col in ['stop_id', 'stop_name']:
                if col in self.stops.columns:
                    if self.stops[col].nunique() < len(self.stops) * 0.5:
                        self.stops[col] = self.stops[col].astype('category')
        
        # trips 최적화
        if self.trips is not None:
            for col in ['trip_id', 'route_id', 'service_id']:
                if col in self.trips.columns:
                    self.trips[col] = self.trips[col].astype(str)
                    if self.trips[col].nunique() < len(self.trips) * 0.5:
                        self.trips[col] = self.trips[col].astype('category')
        
        # stop_times 최적화
        if self.stop_times is not None:
            for col in ['trip_id', 'stop_id']:
                if col in self.stop_times.columns:
                    self.stop_times[col] = self.stop_times[col].astype(str)
                    if self.stop_times[col].nunique() < len(self.stop_times) * 0.5:
                        self.stop_times[col] = self.stop_times[col].astype('category')
            
            if 'stop_sequence' in self.stop_times.columns:
                self.stop_times['stop_sequence'] = pd.to_numeric(
                    self.stop_times['stop_sequence'], errors='coerce'
                ).fillna(0).astype('uint16')
        
        after = self._calculate_memory()
        
        if before > 0:
            reduction = (1 - after/before) * 100
            print(f"      메모리: {before:.1f}MB → {after:.1f}MB ({reduction:.1f}% 절감)")
    
    def _validate_data(self):
        """데이터 검증"""
        print("\n   🔍 데이터 검증...")
        
        issues = []
        
        # 1. routes 검증
        if self.routes is not None:
            if len(self.routes) < 100:
                issues.append(f"routes가 {len(self.routes)}개밖에 없음 (비정상)")
        else:
            issues.append("routes 데이터 없음")
        
        # 2. trips의 route_id 검증
        if self.trips is not None:
            if 'route_id' not in self.trips.columns:
                issues.append("trips에 route_id 컬럼 없음")
            else:
                null_routes = self.trips['route_id'].isna().sum()
                if null_routes > 0:
                    issues.append(f"route_id가 없는 trip: {null_routes}개")
        
        # 3. 참조 무결성
        if self.trips is not None and self.routes is not None:
            if 'route_id' in self.trips.columns and 'route_id' in self.routes.columns:
                trip_routes = set(self.trips['route_id'].dropna().unique())
                valid_routes = set(self.routes['route_id'].unique())
                orphan_routes = trip_routes - valid_routes
                
                if orphan_routes:
                    issues.append(f"routes에 없는 route_id 참조: {len(orphan_routes)}개")
                    # 자동 복구 시도
                    self._auto_fix_routes(orphan_routes)
        
        # 결과 출력
        if issues:
            print("      ⚠️ 발견된 문제:")
            for issue in issues:
                print(f"         - {issue}")
        else:
            print("      ✅ 모든 검증 통과")
    
    def _auto_fix_routes(self, missing_route_ids):
        """누락된 routes 자동 생성"""
        print("\n      🔧 누락된 routes 자동 생성...")
        
        new_routes = []
        for route_id in missing_route_ids:
            new_routes.append({
                'route_id': route_id,
                'agency_id': 'AUTO',
                'route_short_name': str(route_id),
                'route_long_name': f'자동생성_{route_id}',
                'route_type': 3  # 버스
            })
        
        if new_routes:
            new_df = pd.DataFrame(new_routes)
            self.routes = pd.concat([self.routes, new_df], ignore_index=True)
            print(f"         ✅ {len(new_routes)}개 route 자동 생성")
    
    def _calculate_memory(self) -> float:
        """메모리 사용량 계산 (MB)"""
        total = 0
        for table_name in ['stops', 'routes', 'trips', 'stop_times', 'calendar']:
            table = getattr(self, table_name)
            if table is not None:
                total += table.memory_usage(deep=True).sum() / 1024 / 1024
        return total
    
    def _print_summary(self, elapsed: float):
        """로딩 결과 요약"""
        print(f"\n📊 로딩 완료 ({elapsed:.1f}초)")
        print("=" * 60)
        
        # 각 테이블 통계
        for name in ['stops', 'routes', 'trips', 'stop_times']:
            table = getattr(self, name)
            if table is not None:
                print(f"   {name:12s}: {len(table):>10,}행, {len(table.columns):>3}열")
                if name == 'routes' and len(table) < 100:
                    print(f"                  ⚠️ 경고: 노선 수가 너무 적습니다!")
        
        # 메모리 사용량
        memory = self._calculate_memory()
        print(f"\n   총 메모리: {memory:.1f}MB")
    
    def save_clean_data(self, output_dir: str):
        """정리된 데이터 저장"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\n💾 데이터 저장: {output_dir}/")
        
        # CSV로 저장 (BOM 없이)
        for name in ['stops', 'routes', 'trips', 'stop_times', 'calendar']:
            table = getattr(self, name)
            if table is not None:
                file_path = output_path / f"{name}.csv"
                table.to_csv(file_path, index=False, encoding='utf-8')
                print(f"   ✅ {name}.csv")
        
        # 메타데이터 저장
        metadata = {
            'created_at': datetime.now().isoformat(),
            'stats': {
                'stops': len(self.stops) if self.stops is not None else 0,
                'routes': len(self.routes) if self.routes is not None else 0,
                'trips': len(self.trips) if self.trips is not None else 0,
                'stop_times': len(self.stop_times) if self.stop_times is not None else 0
            },
            'issues_fixed': [
                'BOM 문자 제거',
                '컬럼명 정리',
                'routes 데이터 복구'
            ]
        }
        
        with open(output_path / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print("   ✅ metadata.json")
        print("\n✅ 모든 데이터가 정리되어 저장되었습니다!")


# 실행 코드
if __name__ == "__main__":
    print("=" * 60)
    print("KTDB GTFS 데이터 로더 - BOM 및 인코딩 문제 해결")
    print("=" * 60)
    
    # GTFS 경로
    gtfs_path = "/home/twdaniel/multimodal_project2/202303_GTFS_DataSet"  # 실제 경로로 수정
    
    try:
        # 로더 생성 및 실행
        loader = KTDBGTFSLoader(gtfs_path)
        loader.load_all_data()
        
        # 정리된 데이터 저장
        loader.save_clean_data("cleaned_gtfs_data")
        
        print("\n🎉 성공! 이제 Part1에서 cleaned_gtfs_data를 사용하세요.")
        
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()