#!/usr/bin/env python3
"""
Quick examples for Gangnam multimodal routing
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from mmraptor import MultimodalRouter


def main():
    """Run example queries in Gangnam area"""
    
    # Test cases from our research
    test_cases = [
        {
            'name': '강남역 → 역삼역 (근거리)',
            'origin': (37.4979, 127.0276),
            'dest': (37.5007, 127.0363),
            'expected_min': 5.5
        },
        {
            'name': '개포동 → 대치동 (중거리 주택가)',  
            'origin': (37.4813, 127.0701),
            'dest': (37.4935, 127.0591),
            'expected_min': 9.5
        },
        {
            'name': '일원동 → 삼성동 (장거리)',
            'origin': (37.4847, 127.0828), 
            'dest': (37.5115, 127.0595),
            'expected_min': 20.0
        }
    ]
    
    print("🌊 Multimodal RAPTOR - Gangnam Examples")
    print("=" * 50)
    
    # Initialize router with OSM Dijkstra algorithm
    router = MultimodalRouter(algorithm="osm_dijkstra")
    
    for case in test_cases:
        print(f"\n📍 {case['name']}")
        print(f"   출발: {case['origin']}")
        print(f"   도착: {case['dest']}")
        
        try:
            routes = router.route(
                origin=case['origin'],
                destination=case['dest'], 
                departure_time=8.0
            )
            
            if routes:
                best = routes[0]
                print(f" 최적 경로: {best['total_time_min']:.1f}분, {best['total_cost_won']}원")
                print(f"   예상 시간: {case['expected_min']}분 (정확도: {abs(best['total_time_min'] - case['expected_min']):.1f}분 차이)")
                
                # Show first few segments
                for i, segment in enumerate(best.get('segments', [])[:3], 1):
                    print(f"   {i}. {segment.get('description', 'Unknown')}")
                
                if len(best.get('segments', [])) > 3:
                    print(f"   ... ({len(best['segments']) - 3} more segments)")
                    
            else:
                print(" 경로를 찾을 수 없습니다")
                
        except Exception as e:
            print(f" 오류: {e}")
    
    print(f"\n 예제 완료 - {len(test_cases)}개 테스트 케이스")


if __name__ == "__main__":
    main()