"""
Digital Twin ROI Model - Mock Implementation
디지털 트윈 ROI 예측 모델 (Mock 버전)

실제 구현 전까지 사용할 임시 Mock 클래스
"""

from typing import Dict, Any
import random


class DigitalTwinROIPredictor:
    """
    디지털 트윈 시나리오에서 ROI를 예측하는 Mock 클래스
    """
    
    def __init__(self):
        """Mock ROI 예측기 초기화"""
        self.feature_names = [
            # 장비 수량 피처 (5개)
            'cnv_count', 'rtv_count', 'srm_count', 'robotArm_count', 'rack_count',
            
            # 창고 규모 피처 (1개)
            'warehouse_area',
            
            # 컨베이어 성능 피처 (6개)
            'cnv_avg_actual_time', 'cnv_avg_delay_time', 'cnv_avg_wait_time',
            'cnv_avg_idle_time', 'cnv_total_inbound', 'cnv_total_outbound',
            
            # RTV 성능 피처 (6개)  
            'rtv_avg_actual_time', 'rtv_avg_delay_time', 'rtv_avg_wait_time',
            'rtv_avg_idle_time', 'rtv_total_inbound', 'rtv_total_outbound',
            
            # SRM 성능 피처 (6개)
            'srm_avg_actual_time', 'srm_avg_delay_time', 'srm_avg_wait_time',
            'srm_avg_idle_time', 'srm_total_inbound', 'srm_total_outbound',
            
            # 로봇암 성능 피처 (6개)
            'robotArm_avg_actual_time', 'robotArm_avg_delay_time', 'robotArm_avg_wait_time',
            'robotArm_avg_idle_time', 'robotArm_total_inbound', 'robotArm_total_outbound',
            
            # 계산된 피처 (3개)
            'total_throughput', 'completion_rate', 'equipment_density'
        ]
        
        print("📋 Mock DigitalTwinROIPredictor 초기화 완료")
        print(f"   • 지원 피처: {len(self.feature_names)}개")
    
    def extract_features_from_digital_twin(self, scenario: Dict[str, Any]) -> Dict[str, float]:
        """
        디지털 트윈 시나리오에서 ML 피처 추출 (Mock 구현)
        
        Args:
            scenario: 언리얼 디지털 트윈 시나리오 데이터
            
        Returns:
            Dict: ML 훈련용 피처들 (28개)
        """
        query = scenario['query']
        features = {}
        
        # 장비 수량 피처 (5개)
        for equipment in ['cnv', 'rtv', 'srm', 'robotArm']:
            if equipment in query:
                features[f'{equipment}_count'] = float(query[equipment]['Count'])
            else:
                features[f'{equipment}_count'] = 0.0
        
        features['rack_count'] = float(query.get('rack', {}).get('Count', 0))
        features['warehouse_area'] = float(query.get('randingArea', {}).get('Area', 0))
        
        # 성능 지표 피처들 (각 장비당 6개씩, 총 24개)
        for equipment in ['cnv', 'rtv', 'srm', 'robotArm']:
            if equipment in query:
                features[f'{equipment}_avg_actual_time'] = float(query[equipment].get('AvgActualTime', 0))
                features[f'{equipment}_avg_delay_time'] = float(query[equipment].get('AvgDelayTime', 0))
                features[f'{equipment}_avg_wait_time'] = float(query[equipment].get('AvgWaitTime', 0))
                features[f'{equipment}_avg_idle_time'] = float(query[equipment].get('AvgIdleTime', 0))
                features[f'{equipment}_total_inbound'] = float(query[equipment].get('TotalInbound', 0))
                features[f'{equipment}_total_outbound'] = float(query[equipment].get('TotalOutbound', 0))
            else:
                # 장비가 없는 경우 0으로 설정
                features[f'{equipment}_avg_actual_time'] = 0.0
                features[f'{equipment}_avg_delay_time'] = 0.0
                features[f'{equipment}_avg_wait_time'] = 0.0
                features[f'{equipment}_avg_idle_time'] = 0.0
                features[f'{equipment}_total_inbound'] = 0.0
                features[f'{equipment}_total_outbound'] = 0.0
        
        # 계산된 피처들 (3개)
        total_inbound = sum(features[f'{eq}_total_inbound'] for eq in ['cnv', 'rtv', 'srm', 'robotArm'])
        total_outbound = sum(features[f'{eq}_total_outbound'] for eq in ['cnv', 'rtv', 'srm', 'robotArm'])
        
        features['total_throughput'] = total_inbound
        features['completion_rate'] = total_outbound / total_inbound if total_inbound > 0 else 0.0
        features['equipment_density'] = features['rack_count'] / (features['warehouse_area'] / 1000000) if features['warehouse_area'] > 0 else 0.0
        
        # 피처 수 검증
        if len(features) != len(self.feature_names):
            missing_features = set(self.feature_names) - set(features.keys())
            extra_features = set(features.keys()) - set(self.feature_names)
            
            if missing_features:
                print(f"⚠️  누락된 피처: {missing_features}")
            if extra_features:
                print(f"⚠️  추가된 피처: {extra_features}")
        
        return features
    
    def calculate_roi_projections(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        피처를 기반으로 ROI 투영 계산 (Mock 구현)
        
        Args:
            features: extract_features_from_digital_twin에서 추출된 피처들
            
        Returns:
            Dict: ROI 관련 계산 결과들
        """
        # Mock 계산 로직
        warehouse_area = features.get('warehouse_area', 0)
        total_throughput = features.get('total_throughput', 0)
        completion_rate = features.get('completion_rate', 0)
        
        # 투자 금액 추정 (창고 규모 + 장비 수량 기반)
        base_investment = warehouse_area * 0.001  # 면적당 투자비용
        equipment_cost = (
            features.get('cnv_count', 0) * 50000 +
            features.get('rtv_count', 0) * 100000 +
            features.get('srm_count', 0) * 200000 +
            features.get('robotArm_count', 0) * 150000 +
            features.get('rack_count', 0) * 10000
        )
        
        investment_amount = base_investment + equipment_cost
        
        # 연간 개선 효과 추정
        annual_improvement = total_throughput * completion_rate * 1000  # Mock 계산
        
        # ROI 계산
        roi_1year = (annual_improvement - investment_amount * 0.2) / investment_amount * 100 if investment_amount > 0 else 0
        roi_3year = (annual_improvement * 2.5 - investment_amount * 0.4) / investment_amount * 100 if investment_amount > 0 else 0  
        roi_5year = (annual_improvement * 4.0 - investment_amount * 0.6) / investment_amount * 100 if investment_amount > 0 else 0
        
        return {
            'investment_amount': round(investment_amount, 2),
            'annual_improvement': round(annual_improvement, 2),
            'roi_1year': round(roi_1year, 2),
            'roi_3year': round(roi_3year, 2),
            'roi_5year': round(roi_5year, 2)
        }


# 테스트 함수
def test_mock_predictor():
    """Mock 예측기 테스트"""
    predictor = DigitalTwinROIPredictor()
    
    # 테스트 시나리오
    test_scenario = {
        'scenario_id': 1,
        'metadata': {
            'warehouse_type': '중형창고 10,000m²',
            'equipment_config': '표준형',
            'operation_scenario': '평시',
            'equipment_condition': '양호상태'
        },
        'query': {
            'cnv': {
                'Count': '10',
                'AvgActualTime': '5.123456',
                'AvgDelayTime': '2.456789',
                'AvgWaitTime': '1.234567',
                'AvgIdleTime': '0.456789',
                'TotalInbound': '1000',
                'TotalOutbound': '950'
            },
            'rtv': {
                'Count': '5',
                'AvgActualTime': '8.123456',
                'AvgDelayTime': '3.456789',
                'AvgWaitTime': '2.234567',
                'AvgIdleTime': '1.456789',
                'TotalInbound': '500',
                'TotalOutbound': '450'
            },
            'srm': {
                'Count': '8',
                'AvgActualTime': '6.123456',
                'AvgDelayTime': '2.856789',
                'AvgWaitTime': '0.834567',
                'AvgIdleTime': '2.156789',
                'TotalInbound': '800',
                'TotalOutbound': '720'
            },
            'robotArm': {
                'Count': '2',
                'AvgActualTime': '3.123456',
                'AvgDelayTime': '1.456789',
                'AvgWaitTime': '2.234567',
                'AvgIdleTime': '3.456789',
                'TotalInbound': '200',
                'TotalOutbound': '180'
            },
            'rack': {
                'Count': '100'
            },
            'randingArea': {
                'Area': '100000000'
            }
        }
    }
    
    # 피처 추출 테스트
    features = predictor.extract_features_from_digital_twin(test_scenario)
    print(f"\n🧪 피처 추출 테스트 결과:")
    print(f"   • 추출된 피처 수: {len(features)}개")
    print(f"   • 총 처리량: {features['total_throughput']}")
    print(f"   • 완료율: {features['completion_rate']:.2%}")
    
    # ROI 계산 테스트
    roi_results = predictor.calculate_roi_projections(features)
    print(f"\n💰 ROI 계산 테스트 결과:")
    print(f"   • 투자 금액: ${roi_results['investment_amount']:,.2f}")
    print(f"   • 연간 개선: ${roi_results['annual_improvement']:,.2f}")
    print(f"   • 1년 ROI: {roi_results['roi_1year']:.2f}%")
    print(f"   • 3년 ROI: {roi_results['roi_3year']:.2f}%")
    print(f"   • 5년 ROI: {roi_results['roi_5year']:.2f}%")


if __name__ == "__main__":
    test_mock_predictor()
