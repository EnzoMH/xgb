import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple

class DigitalTwinROIPredictor:
    """
    언리얼 디지털 트윈 데이터 기반 창고 자동화 ROI 예측 모델
    1년, 3년, 5년 투자회수 예측
    """
    
    def __init__(self):
        self.models = {
            '1year': None,
            '3year': None, 
            '5year': None
        }
        self.feature_columns = []
        self.roi_calculation_params = {
            'labor_cost_per_hour': 15000,  # 시간당 인건비
            'operation_hours_per_day': 16,  # 일 운영시간 (2교대)
            'operation_days_per_year': 300,  # 연 운영일
            'equipment_depreciation_years': 7,  # 장비 감가상각
            'maintenance_cost_ratio': 0.08,  # 연간 유지보수비 (장비가격의 8%)
            'electricity_cost_per_kwh': 120,  # kWh당 전기료
            'box_processing_revenue': 2000,  # 박스당 처리 수익
        }
    
    def extract_features_from_digital_twin(self, dt_data: Dict[str, Any]) -> Dict[str, float]:
        """
        디지털 트윈 데이터에서 ROI 예측용 Feature 추출
        """
        features = {}
        query_data = dt_data.get('query', {})
        
        # === 1. 기본 인프라 정보 ===
        features['warehouse_area_sqm'] = float(query_data.get('randingArea', {}).get('Area', 0)) / 10000  # cm² → m²
        features['conveyor_count'] = float(query_data.get('cnv', {}).get('Count', 0))
        features['rtv_count'] = float(query_data.get('rtv', {}).get('Count', 0))  # AGV
        features['srm_count'] = float(query_data.get('srm', {}).get('Count', 0))  # 자동창고
        features['robot_arm_count'] = float(query_data.get('robotArm', {}).get('Count', 0))
        features['rack_count'] = float(query_data.get('rack', {}).get('Count', 0))
        
        # 자동화 수준 계산 (1-5 스케일)
        automation_score = 1  # 기본
        if features['conveyor_count'] > 10: automation_score += 1
        if features['rtv_count'] > 0: automation_score += 1  
        if features['srm_count'] > 0: automation_score += 1
        if features['robot_arm_count'] > 0: automation_score += 1
        features['automation_level'] = min(5, automation_score)
        
        # === 2. 현재 성능 지표 (10분 기준) ===
        
        # 처리량 계산
        total_inbound = 0
        total_outbound = 0
        equipment_data = {}
        
        for eq_type in ['cnv', 'rtv', 'srm', 'robotArm']:
            eq_data = query_data.get(eq_type, {})
            inbound = float(eq_data.get('TotalInbound', 0))
            outbound = float(eq_data.get('TotalOutbound', 0))
            
            total_inbound += inbound
            total_outbound += outbound
            
            equipment_data[eq_type] = {
                'count': float(eq_data.get('Count', 0)),
                'actual_time': float(eq_data.get('AvgActualTime', 0)),
                'delay_time': float(eq_data.get('AvgDelayTime', 0)),
                'wait_time': float(eq_data.get('AvgWaitTime', 0)),
                'idle_time': float(eq_data.get('AvgIdleTime', 0)),
                'inbound': inbound,
                'outbound': outbound
            }
        
        # 시간당 처리량으로 환산 (10분 → 60분)
        features['hourly_throughput'] = total_inbound * 6  # 투입량 기준
        features['hourly_completion'] = total_outbound * 6  # 완료량 기준
        features['current_efficiency'] = total_outbound / max(total_inbound, 1)  # 처리 완료율
        
        # === 3. 시간 효율성 지표 ===
        
        # 가중평균 시간 계산
        total_equipment = sum([eq['count'] for eq in equipment_data.values()])
        
        if total_equipment > 0:
            weighted_actual = sum([eq['actual_time'] * eq['count'] for eq in equipment_data.values()]) / total_equipment
            weighted_delay = sum([eq['delay_time'] * eq['count'] for eq in equipment_data.values()]) / total_equipment  
            weighted_wait = sum([eq['wait_time'] * eq['count'] for eq in equipment_data.values()]) / total_equipment
            weighted_idle = sum([eq['idle_time'] * eq['count'] for eq in equipment_data.values()]) / total_equipment
        else:
            weighted_actual = weighted_delay = weighted_wait = weighted_idle = 0
        
        features['avg_actual_time'] = weighted_actual
        features['avg_delay_time'] = weighted_delay
        features['avg_wait_time'] = weighted_wait  
        features['avg_idle_time'] = weighted_idle
        
        total_cycle_time = weighted_actual + weighted_delay + weighted_wait + weighted_idle
        features['total_cycle_time'] = total_cycle_time
        
        # 효율성 비율
        if total_cycle_time > 0:
            features['actual_time_ratio'] = weighted_actual / total_cycle_time
            features['delay_time_ratio'] = weighted_delay / total_cycle_time
            features['waste_time_ratio'] = (weighted_delay + weighted_wait) / total_cycle_time
        else:
            features['actual_time_ratio'] = 0
            features['delay_time_ratio'] = 0  
            features['waste_time_ratio'] = 0
        
        # === 4. 병목 분석 ===
        
        # 각 장비별 병목 지표
        for eq_type, eq_data in equipment_data.items():
            if eq_data['count'] > 0:
                delay_ratio = eq_data['delay_time'] / max(eq_data['actual_time'], 1)
                utilization = eq_data['inbound'] / eq_data['count']  # 대당 처리량
                
                features[f'{eq_type}_delay_ratio'] = delay_ratio
                features[f'{eq_type}_utilization'] = utilization
                features[f'{eq_type}_efficiency'] = eq_data['outbound'] / max(eq_data['inbound'], 1)
        
        # 전체 시스템 병목 점수
        bottleneck_scores = []
        for eq_type in equipment_data.keys():
            if f'{eq_type}_delay_ratio' in features:
                bottleneck_scores.append(features[f'{eq_type}_delay_ratio'])
        
        features['system_bottleneck_score'] = max(bottleneck_scores) if bottleneck_scores else 0
        
        # === 5. 투자 규모 추정 ===
        
        # 현재 장비 가치 추정 (억원)
        equipment_value = 0
        equipment_value += features['conveyor_count'] * 0.5  # 컨베이어 대당 5천만원
        equipment_value += features['rtv_count'] * 8  # AGV 대당 8억원
        equipment_value += features['srm_count'] * 15  # SRM 대당 15억원  
        equipment_value += features['robot_arm_count'] * 5  # 로봇암 대당 5억원
        
        features['current_equipment_value'] = equipment_value
        features['equipment_density'] = equipment_value / max(features['warehouse_area_sqm'], 1000)
        
        return features
    
    def calculate_roi_projections(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Feature를 기반으로 1/3/5년 ROI 계산
        """
        params = self.roi_calculation_params
        
        # === 현재 상태 기준 연간 운영 지표 ===
        
        # 연간 처리 가능량
        annual_throughput = (features['hourly_throughput'] * 
                           params['operation_hours_per_day'] * 
                           params['operation_days_per_year'])
        
        # 연간 완료량 (현재 효율성 적용)
        annual_completion = annual_throughput * features['current_efficiency']
        
        # === 비용 구조 ===
        
        # 연간 인건비 (현재 효율성 기준)
        # 비효율로 인한 추가 인건비 = 지연시간 + 대기시간으로 인한 추가 비용
        inefficiency_multiplier = 1 + (features['waste_time_ratio'] * 0.5)  # 비효율로 인한 50% 추가 비용
        
        estimated_workers = max(10, int(features['warehouse_area_sqm'] / 500))  # 500m²당 1명 추정
        annual_labor_cost = (estimated_workers * params['labor_cost_per_hour'] * 
                           params['operation_hours_per_day'] * params['operation_days_per_year'] * 
                           inefficiency_multiplier)
        
        # 연간 유지보수비
        annual_maintenance = features['current_equipment_value'] * 100000000 * params['maintenance_cost_ratio']
        
        # 연간 전력비 (장비 수에 비례)
        total_equipment = (features['conveyor_count'] + features['rtv_count'] + 
                         features['srm_count'] + features['robot_arm_count'])
        annual_electricity = total_equipment * 50 * 24 * 365 * params['electricity_cost_per_kwh'] / 1000  # 50kW 가정
        
        # === 수익 구조 ===
        
        # 연간 처리 수익
        annual_revenue = annual_completion * params['box_processing_revenue']
        
        # 현재 순이익
        current_annual_profit = annual_revenue - annual_labor_cost - annual_maintenance - annual_electricity
        
        # === 자동화 개선 시나리오 ===
        
        # 개선 가능성 계산
        efficiency_improvement_potential = min(0.4, features['waste_time_ratio'] * 0.6)  # 최대 40% 개선
        bottleneck_resolution_benefit = min(0.3, features['system_bottleneck_score'] * 0.1)  # 최대 30% 개선
        
        total_improvement_potential = efficiency_improvement_potential + bottleneck_resolution_benefit
        
        # 자동화 투자 필요 금액 추정
        current_automation = features['automation_level']
        target_automation = min(5, current_automation + 1)
        
        additional_investment = 0
        if target_automation > current_automation:
            # Level별 추가 투자 (억원)
            investment_per_level = {
                2: 5,   # WMS 고도화
                3: 15,  # AGV 추가  
                4: 25,  # 컨베이어 확장
                5: 40   # 로봇 피킹
            }
            additional_investment = investment_per_level.get(target_automation, 10)
        
        # 개선 후 예상 성과
        improved_efficiency = min(0.95, features['current_efficiency'] + total_improvement_potential)
        improved_annual_completion = annual_throughput * improved_efficiency
        improved_annual_revenue = improved_annual_completion * params['box_processing_revenue']
        
        # 자동화로 인한 인건비 절감 (10-30%)
        labor_reduction = min(0.3, total_improvement_potential * 0.5)
        improved_annual_labor_cost = annual_labor_cost * (1 - labor_reduction)
        
        # 개선 후 순이익
        improved_annual_profit = (improved_annual_revenue - improved_annual_labor_cost - 
                                annual_maintenance - annual_electricity)
        
        # 연간 순개선 효과
        annual_improvement = improved_annual_profit - current_annual_profit
        
        # === ROI 계산 ===
        
        investment_amount = additional_investment * 100000000  # 억원 → 원
        
        roi_results = {}
        
        if investment_amount > 0:
            # 1년 ROI
            roi_1year = ((annual_improvement - investment_amount * 0.1) / investment_amount) * 100  # 감가상각 고려
            
            # 3년 ROI  
            total_3year_benefit = annual_improvement * 3
            roi_3year = ((total_3year_benefit - investment_amount) / investment_amount) * 100
            
            # 5년 ROI
            total_5year_benefit = annual_improvement * 5  
            roi_5year = ((total_5year_benefit - investment_amount) / investment_amount) * 100
        else:
            roi_1year = roi_3year = roi_5year = 0
        
        roi_results = {
            'roi_1year': roi_1year,
            'roi_3year': roi_3year, 
            'roi_5year': roi_5year,
            'investment_amount': investment_amount,
            'annual_improvement': annual_improvement,
            'current_annual_profit': current_annual_profit,
            'improved_annual_profit': improved_annual_profit,
            'efficiency_improvement': total_improvement_potential * 100,
            'current_efficiency': features['current_efficiency'] * 100,
            'target_efficiency': improved_efficiency * 100
        }
        
        return roi_results
    
    def analyze_your_warehouse(self, dt_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        실제 디지털 트윈 데이터 분석 및 ROI 예측
        """
        print("🏭 실제 창고 디지털 트윈 ROI 분석")
        print("=" * 60)
        
        # Feature 추출
        features = self.extract_features_from_digital_twin(dt_data)
        
        print(f"\n📏 창고 기본 정보:")
        print(f"• 창고 면적: {features['warehouse_area_sqm']:,.0f}m² (약 {features['warehouse_area_sqm']/330:.0f}평)")
        print(f"• 자동화 수준: Level {features['automation_level']}/5")
        print(f"• 컨베이어: {features['conveyor_count']:.0f}대")
        print(f"• RTV(AGV): {features['rtv_count']:.0f}대") 
        print(f"• SRM: {features['srm_count']:.0f}대")
        print(f"• 로봇암: {features['robot_arm_count']:.0f}대")
        print(f"• 현재 장비 가치: {features['current_equipment_value']:.1f}억원")
        
        print(f"\n⚡ 현재 성능 지표:")
        print(f"• 시간당 처리량: {features['hourly_throughput']:.0f}박스")
        print(f"• 시간당 완료량: {features['hourly_completion']:.0f}박스")
        print(f"• 처리 효율성: {features['current_efficiency']*100:.1f}%")
        print(f"• 평균 사이클 시간: {features['total_cycle_time']:.1f}초")
        print(f"• 비효율 시간 비율: {features['waste_time_ratio']*100:.1f}%")
        print(f"• 시스템 병목 점수: {features['system_bottleneck_score']:.2f}")
        
        # ROI 계산
        roi_results = self.calculate_roi_projections(features)
        
        print(f"\n💰 ROI 예측 결과:")
        print(f"• 추가 투자 필요: {roi_results['investment_amount']/100000000:.1f}억원")
        print(f"• 연간 개선 효과: {roi_results['annual_improvement']/100000000:.1f}억원")
        print(f"• 효율성 개선: {features['current_efficiency']*100:.1f}% → {roi_results['target_efficiency']:.1f}%")
        
        print(f"\n📊 투자 회수 예측:")
        print(f"• 1년 ROI: {roi_results['roi_1year']:+.1f}%")
        print(f"• 3년 ROI: {roi_results['roi_3year']:+.1f}%")
        print(f"• 5년 ROI: {roi_results['roi_5year']:+.1f}%")
        
        # 투자 권장사항
        print(f"\n💡 투자 권장사항:")
        if roi_results['roi_3year'] > 50:
            print("✅ 3년 내 50% 이상 수익률 예상 - 적극 투자 권장")
        elif roi_results['roi_3year'] > 20:
            print("⚠️  3년 내 20% 이상 수익률 예상 - 신중한 투자 검토")
        elif roi_results['roi_5year'] > 30:
            print("🔄 장기적으로 수익성 있음 - 5년 계획으로 단계적 투자")
        else:
            print("❌ 현재 조건에서는 투자 수익성 낮음 - 운영 효율화 우선")
        
        # 개선 우선순위
        print(f"\n🎯 개선 우선순위:")
        bottlenecks = []
        
        for eq_type in ['cnv', 'rtv', 'srm']:
            if f'{eq_type}_delay_ratio' in features:
                if features[f'{eq_type}_delay_ratio'] > 2:
                    bottlenecks.append((eq_type, features[f'{eq_type}_delay_ratio']))
        
        bottlenecks.sort(key=lambda x: x[1], reverse=True)
        
        equipment_names = {'cnv': '컨베이어', 'rtv': 'RTV(AGV)', 'srm': 'SRM'}
        for i, (eq_type, ratio) in enumerate(bottlenecks[:3]):
            print(f"{i+1}. {equipment_names[eq_type]} 최적화 (지연비율: {ratio:.1f})")
        
        return {
            'features': features,
            'roi_projections': roi_results,
            'recommendations': bottlenecks
        }

# 실제 데이터로 분석 실행
def main():
    # 실제 디지털 트윈 데이터
    actual_data = {
        "query": {
            "cnv": {
                "Count": "23",
                "AvgActualTime": "5.223294",
                "AvgDelayTime": "32.459557",
                "AvgWaitTime": "32.233143", 
                "AvgIdleTime": "0.439609",
                "TotalInbound": "549",
                "TotalOutbound": "0"
            },
            "rtv": {
                "Count": "6",
                "AvgActualTime": "18.526421",
                "AvgDelayTime": "11.570003",
                "AvgWaitTime": "2.529861",
                "AvgIdleTime": "2.186285", 
                "TotalInbound": "184",
                "TotalOutbound": "41"
            },
            "srm": {
                "Count": "10",
                "AvgActualTime": "11.489976",
                "AvgDelayTime": "16.162361",
                "AvgWaitTime": "0.0",
                "AvgIdleTime": "2.807265",
                "TotalInbound": "278", 
                "TotalOutbound": "0"
            },
            "robotArm": {
                "Count": "1",
                "AvgActualTime": "1.040927",
                "AvgDelayTime": "0.0",
                "AvgWaitTime": "1.024083",
                "AvgIdleTime": "0.894164",
                "TotalInbound": "143",
                "TotalOutbound": "0"
            },
            "rack": {
                "Count": "20"
            },
            "randingArea": {
                "Area": "126042896.0"
            }
        }
    }
    
    predictor = DigitalTwinROIPredictor()
    analysis_result = predictor.analyze_your_warehouse(actual_data)
    
    return analysis_result

if __name__ == "__main__":
    result = main()