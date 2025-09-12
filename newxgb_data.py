import pandas as pd
import numpy as np
import json
from typing import Dict, Any, List
import itertools
from datetime import datetime, timedelta

class DigitalTwinBigDataGenerator:
    """
    언리얼 디지털 트윈 시뮬레이션 기반 빅데이터 생성기
    다양한 시나리오를 체계적으로 생성하여 ML 훈련용 데이터셋 구축
    """
    
    def __init__(self):
        # 시뮬레이션 파라미터 범위 정의
        self.scenario_parameters = {
            # 창고 규모별
            "warehouse_sizes": [
                {"area": 50000000, "description": "소형창고 5,000m²"},      # 5천만 cm²
                {"area": 100000000, "description": "중형창고 10,000m²"},    # 1억 cm²  
                {"area": 150000000, "description": "대형창고 15,000m²"},    # 1.5억 cm²
                {"area": 300000000, "description": "초대형창고 30,000m²"},  # 3억 cm²
            ],
            
            # 장비 구성별
            "equipment_configs": [
                {
                    "name": "기본형",
                    "conveyor": (5, 15),     # 5-15대
                    "rtv": (2, 8),          # 2-8대
                    "srm": (3, 12),         # 3-12대
                    "robot_arm": (0, 2),    # 0-2대
                    "automation_level": 2
                },
                {
                    "name": "표준형", 
                    "conveyor": (10, 25),
                    "rtv": (4, 12),
                    "srm": (5, 15),
                    "robot_arm": (1, 3),
                    "automation_level": 3
                },
                {
                    "name": "고급형",
                    "conveyor": (15, 35),
                    "rtv": (6, 18),
                    "srm": (8, 20),
                    "robot_arm": (1, 4),
                    "automation_level": 4
                },
                {
                    "name": "최고급형",
                    "conveyor": (20, 50),
                    "rtv": (10, 25),
                    "srm": (10, 30),
                    "robot_arm": (2, 6),
                    "automation_level": 5
                }
            ],
            
            # 운영 시나리오별
            "operation_scenarios": [
                {
                    "name": "평시",
                    "throughput_multiplier": 1.0,
                    "delay_multiplier": 1.0,
                    "error_rate": 0.03
                },
                {
                    "name": "성수기",
                    "throughput_multiplier": 1.8,
                    "delay_multiplier": 1.4,
                    "error_rate": 0.05
                },
                {
                    "name": "비수기",
                    "throughput_multiplier": 0.6,
                    "delay_multiplier": 0.8,
                    "error_rate": 0.02
                },
                {
                    "name": "피크타임",
                    "throughput_multiplier": 2.2,
                    "delay_multiplier": 2.0,
                    "error_rate": 0.08
                },
                {
                    "name": "야간운영",
                    "throughput_multiplier": 0.7,
                    "delay_multiplier": 0.9,
                    "error_rate": 0.04
                }
            ],
            
            # 장비 상태별
            "equipment_conditions": [
                {
                    "name": "최적상태",
                    "efficiency_factor": 1.0,
                    "breakdown_rate": 0.01
                },
                {
                    "name": "양호상태", 
                    "efficiency_factor": 0.9,
                    "breakdown_rate": 0.03
                },
                {
                    "name": "보통상태",
                    "efficiency_factor": 0.8,
                    "breakdown_rate": 0.05
                },
                {
                    "name": "노후상태",
                    "efficiency_factor": 0.7,
                    "breakdown_rate": 0.1
                }
            ]
        }
    
    def generate_simulation_scenarios(self, num_scenarios_per_combination: int = 50) -> List[Dict]:
        """
        체계적인 시나리오 조합 생성
        """
        scenarios = []
        scenario_id = 1
        
        # 모든 조합 생성
        for warehouse in self.scenario_parameters["warehouse_sizes"]:
            for equipment in self.scenario_parameters["equipment_configs"]:
                for operation in self.scenario_parameters["operation_scenarios"]:
                    for condition in self.scenario_parameters["equipment_conditions"]:
                        
                        # 각 조합당 여러 번 시뮬레이션 (노이즈 추가)
                        for variation in range(num_scenarios_per_combination):
                            scenario = self._create_single_scenario(
                                scenario_id, warehouse, equipment, operation, condition, variation
                            )
                            scenarios.append(scenario)
                            scenario_id += 1
        
        print(f"📊 총 {len(scenarios):,}개 시나리오 생성 완료")
        print(f"   • 창고 규모: {len(self.scenario_parameters['warehouse_sizes'])}가지")
        print(f"   • 장비 구성: {len(self.scenario_parameters['equipment_configs'])}가지") 
        print(f"   • 운영 시나리오: {len(self.scenario_parameters['operation_scenarios'])}가지")
        print(f"   • 장비 상태: {len(self.scenario_parameters['equipment_conditions'])}가지")
        print(f"   • 변동 케이스: {num_scenarios_per_combination}개/조합")
        
        return scenarios
    
    def _create_single_scenario(self, scenario_id: int, warehouse: Dict, equipment: Dict, 
                              operation: Dict, condition: Dict, variation: int) -> Dict:
        """
        단일 시나리오 생성
        """
        # 기본 노이즈 팩터
        noise_factor = 1 + np.random.normal(0, 0.1)  # ±10% 노이즈
        
        # 장비 수량 결정
        conveyor_count = np.random.randint(equipment["conveyor"][0], equipment["conveyor"][1] + 1)
        rtv_count = np.random.randint(equipment["rtv"][0], equipment["rtv"][1] + 1)
        srm_count = np.random.randint(equipment["srm"][0], equipment["srm"][1] + 1)
        robot_arm_count = np.random.randint(equipment["robot_arm"][0], equipment["robot_arm"][1] + 1)
        rack_count = max(10, int(warehouse["area"] / 5000000))  # 면적 대비 랙 수
        
        # 처리량 계산
        base_throughput = warehouse["area"] / 20000  # 면적 기반 기본 처리량
        actual_throughput = (base_throughput * operation["throughput_multiplier"] * 
                           condition["efficiency_factor"] * noise_factor)
        
        # 시간 지표 계산
        base_actual_time = 5 + np.random.normal(0, 2)  # 기본 5초 ± 2초
        base_delay_time = base_actual_time * (2 - condition["efficiency_factor"]) * operation["delay_multiplier"]
        base_wait_time = np.random.exponential(10) * (1 + operation["error_rate"])
        base_idle_time = np.random.exponential(2)
        
        # 장비별 특성 반영
        equipment_performance = {}
        
        # 컨베이어
        cnv_actual = base_actual_time * np.random.uniform(0.8, 1.2)
        cnv_delay = base_delay_time * np.random.uniform(1.5, 3.0)  # 컨베이어는 지연 많음
        cnv_wait = base_wait_time * np.random.uniform(2.0, 4.0)   # 대기도 많음
        cnv_idle = base_idle_time * np.random.uniform(0.1, 0.8)
        cnv_inbound = int(actual_throughput * 0.4 * np.random.uniform(0.8, 1.2))  # 40% 담당
        cnv_outbound = int(cnv_inbound * np.random.uniform(0.0, 0.3))  # 낮은 완료율
        
        equipment_performance["cnv"] = {
            "Count": str(conveyor_count),
            "AvgActualTime": f"{cnv_actual:.6f}",
            "AvgDelayTime": f"{cnv_delay:.6f}",
            "AvgWaitTime": f"{cnv_wait:.6f}",
            "AvgIdleTime": f"{cnv_idle:.6f}",
            "TotalInbound": str(cnv_inbound),
            "TotalOutbound": str(cnv_outbound)
        }
        
        # RTV (AGV)
        rtv_actual = base_actual_time * np.random.uniform(2.0, 4.0)  # AGV는 이동시간 포함
        rtv_delay = base_delay_time * np.random.uniform(0.5, 2.0)
        rtv_wait = base_wait_time * np.random.uniform(0.1, 1.0)
        rtv_idle = base_idle_time * np.random.uniform(0.5, 2.0)
        rtv_inbound = int(actual_throughput * 0.2 * np.random.uniform(0.8, 1.2))  # 20% 담당
        rtv_outbound = int(rtv_inbound * np.random.uniform(0.1, 0.6))  # 중간 완료율
        
        equipment_performance["rtv"] = {
            "Count": str(rtv_count),
            "AvgActualTime": f"{rtv_actual:.6f}",
            "AvgDelayTime": f"{rtv_delay:.6f}",
            "AvgWaitTime": f"{rtv_wait:.6f}",
            "AvgIdleTime": f"{rtv_idle:.6f}",
            "TotalInbound": str(rtv_inbound),
            "TotalOutbound": str(rtv_outbound)
        }
        
        # SRM
        srm_actual = base_actual_time * np.random.uniform(1.5, 3.0)
        srm_delay = base_delay_time * np.random.uniform(0.8, 2.5)
        srm_wait = 0 if np.random.random() > 0.3 else base_wait_time * 0.5  # 70% 확률로 대기없음
        srm_idle = base_idle_time * np.random.uniform(0.5, 3.0)
        srm_inbound = int(actual_throughput * 0.25 * np.random.uniform(0.8, 1.2))  # 25% 담당
        srm_outbound = int(srm_inbound * np.random.uniform(0.0, 0.4))  # 낮은 완료율
        
        equipment_performance["srm"] = {
            "Count": str(srm_count),
            "AvgActualTime": f"{srm_actual:.6f}",
            "AvgDelayTime": f"{srm_delay:.6f}",
            "AvgWaitTime": f"{srm_wait:.6f}",
            "AvgIdleTime": f"{srm_idle:.6f}",
            "TotalInbound": str(srm_inbound),
            "TotalOutbound": str(srm_outbound)
        }
        
        # Robot Arm
        robot_actual = base_actual_time * np.random.uniform(0.2, 2.0)  # 빠른 작업
        robot_delay = 0 if np.random.random() > 0.2 else base_delay_time * 0.5  # 80% 확률로 지연없음
        robot_wait = base_wait_time * np.random.uniform(0.2, 2.0)
        robot_idle = base_idle_time * np.random.uniform(0.2, 3.0)
        robot_inbound = int(actual_throughput * 0.15 * np.random.uniform(0.8, 1.2))  # 15% 담당
        robot_outbound = int(robot_inbound * np.random.uniform(0.0, 0.2))  # 매우 낮은 완료율
        
        equipment_performance["robotArm"] = {
            "Count": str(robot_arm_count),
            "AvgActualTime": f"{robot_actual:.6f}",
            "AvgDelayTime": f"{robot_delay:.6f}",
            "AvgWaitTime": f"{robot_wait:.6f}",
            "AvgIdleTime": f"{robot_idle:.6f}",
            "TotalInbound": str(robot_inbound),
            "TotalOutbound": str(robot_outbound)
        }
        
        # 전체 시나리오 구성
        scenario = {
            "scenario_id": scenario_id,
            "metadata": {
                "warehouse_type": warehouse["description"],
                "equipment_config": equipment["name"],
                "operation_scenario": operation["name"],
                "equipment_condition": condition["name"],
                "variation_id": variation,
                "automation_level": equipment["automation_level"],
                "simulation_date": datetime.now().isoformat()
            },
            "query": {
                **equipment_performance,
                "rack": {
                    "Count": str(rack_count)
                },
                "randingArea": {
                    "Area": str(warehouse["area"])
                }
            }
        }
        
        return scenario
    
    def save_scenarios_for_unreal_simulation(self, scenarios: List[Dict], 
                                           output_file: str = "simulation_scenarios.json") -> None:
        """
        언리얼 시뮬레이션용 시나리오 파일 저장
        """
        # 시나리오를 배치별로 분할 (언리얼에서 처리하기 쉽게)
        batch_size = 100
        batches = [scenarios[i:i + batch_size] for i in range(0, len(scenarios), batch_size)]
        
        simulation_config = {
            "total_scenarios": len(scenarios),
            "batch_count": len(batches),
            "batch_size": batch_size,
            "generation_timestamp": datetime.now().isoformat(),
            "parameters_used": self.scenario_parameters,
            "batches": []
        }
        
        for i, batch in enumerate(batches):
            batch_info = {
                "batch_id": i + 1,
                "scenario_count": len(batch),
                "scenarios": batch
            }
            simulation_config["batches"].append(batch_info)
        
        # JSON 파일로 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(simulation_config, f, ensure_ascii=False, indent=2)
        
        print(f"💾 시뮬레이션 설정 파일 저장 완료: {output_file}")
        print(f"   • 총 {len(scenarios):,}개 시나리오")
        print(f"   • {len(batches)}개 배치로 분할")
        print(f"   • 각 배치당 최대 {batch_size}개 시나리오")
    
    def create_training_dataset_structure(self, scenarios: List[Dict]) -> pd.DataFrame:
        """
        시나리오를 XGB 훈련용 데이터셋 구조로 변환 (시뮬레이션 전 - X-Label만 생성)
        """
        try:
            from digital_twin_roi_model import DigitalTwinROIPredictor
            predictor = DigitalTwinROIPredictor()
        except ImportError:
            print("⚠️  digital_twin_roi_model 모듈이 없어서 Mock 피처 생성기 사용")
            predictor = None
        
        dataset_rows = []
        
        print("🔄 시뮬레이션 전 피처 데이터셋 생성 중...")
        
        for i, scenario in enumerate(scenarios):
            if i % 1000 == 0:
                print(f"   진행률: {i:,}/{len(scenarios):,} ({i/len(scenarios)*100:.1f}%)")
            
            # Feature 추출 (X-Label)
            if predictor:
                features = predictor.extract_features_from_digital_twin(scenario)
            else:
                # Mock 피처 생성 (실제 digital_twin_roi_model이 없을 때)
                features = self._create_mock_features(scenario)
            
            # 메타데이터 추가
            row = {
                'scenario_id': scenario['scenario_id'],
                'warehouse_type': scenario['metadata']['warehouse_type'],
                'equipment_config': scenario['metadata']['equipment_config'],
                'operation_scenario': scenario['metadata']['operation_scenario'],
                'equipment_condition': scenario['metadata']['equipment_condition'],
                **features,  # 28개 기본 X-Label
                # Y-Label은 시뮬레이션 후에 추가됨
                'kpi_score': None,      # 언리얼 시뮬레이션 결과
                'roi_actual': None,     # 언리얼 시뮬레이션 결과  
                'eff_score': None       # 언리얼 시뮬레이션 결과
            }
            
            dataset_rows.append(row)
        
        df = pd.DataFrame(dataset_rows)
        print(f"✅ 시뮬레이션 전 피처셋 생성 완료: {df.shape}")
        
        return df
    
    def _create_mock_features(self, scenario: Dict) -> Dict[str, float]:
        """
        digital_twin_roi_model이 없을 때 Mock 피처 생성
        """
        query = scenario['query']
        
        # 기본 28개 피처 Mock 생성
        features = {}
        
        # 장비 수량 피처
        for equipment in ['cnv', 'rtv', 'srm', 'robotArm']:
            if equipment in query:
                features[f'{equipment}_count'] = float(query[equipment]['Count'])
        
        features['rack_count'] = float(query['rack']['Count'])
        features['warehouse_area'] = float(query['randingArea']['Area'])
        
        # 성능 지표 피처들 (시뮬레이션 전 추정값)
        for equipment in ['cnv', 'rtv', 'srm', 'robotArm']:
            if equipment in query:
                features[f'{equipment}_avg_actual_time'] = float(query[equipment]['AvgActualTime'])
                features[f'{equipment}_avg_delay_time'] = float(query[equipment]['AvgDelayTime'])
                features[f'{equipment}_avg_wait_time'] = float(query[equipment]['AvgWaitTime'])
                features[f'{equipment}_avg_idle_time'] = float(query[equipment]['AvgIdleTime'])
                features[f'{equipment}_total_inbound'] = float(query[equipment]['TotalInbound'])
                features[f'{equipment}_total_outbound'] = float(query[equipment]['TotalOutbound'])
        
        # 추가 계산 피처들
        total_inbound = sum(float(query[eq]['TotalInbound']) for eq in ['cnv', 'rtv', 'srm', 'robotArm'] if eq in query)
        total_outbound = sum(float(query[eq]['TotalOutbound']) for eq in ['cnv', 'rtv', 'srm', 'robotArm'] if eq in query)
        
        features['total_throughput'] = total_inbound
        features['completion_rate'] = total_outbound / total_inbound if total_inbound > 0 else 0
        features['equipment_density'] = features['rack_count'] / (features['warehouse_area'] / 1000000)  # per m²
        
        return features

    def process_simulation_results(self, simulation_results: str, scenario_id: int) -> Dict[str, float]:
        """
        언리얼 시뮬레이션 결과 텍스트를 파싱하여 Y-Label 추출
        
        Args:
            simulation_results: 언리얼에서 받은 결과 텍스트
            scenario_id: 시나리오 ID
            
        Returns:
            Dict: 파싱된 KPI, ROI, EFF 값들
        """
        import re
        
        # 정규식으로 핵심 지표 추출
        patterns = {
            'throughput': r'(\d+\.?\d*)\s*의?\s*처리량',
            'utilization': r'(\d+\.?\d*)%\s*의?\s*활용도',
            'cost': r'비용\s*(\d+\.?\d*)\s*달러',
            'profit': r'(\d+\.?\d*)\s*의?\s*이익',
            'efficiency': r'효율성\s*점수는\s*(\d+\.?\d*)%'
        }
        
        parsed_data = {}
        
        for key, pattern in patterns.items():
            match = re.search(pattern, simulation_results)
            if match:
                parsed_data[key] = float(match.group(1))
            else:
                parsed_data[key] = 0.0
        
        # Y-Label 계산
        result = {
            'kpi_score': self._calculate_kpi_score(
                parsed_data.get('throughput', 0),
                parsed_data.get('utilization', 0)
            ),
            'roi_actual': self._calculate_roi_actual(
                parsed_data.get('profit', 0),
                parsed_data.get('cost', 1)
            ),
            'eff_score': parsed_data.get('efficiency', 0)
        }
        
        print(f"📊 시나리오 {scenario_id} 결과 파싱 완료: {result}")
        return result
    
    def _calculate_kpi_score(self, throughput: float, utilization: float) -> float:
        """
        KPI 종합 점수 계산 (처리량과 활용도를 결합)
        """
        # 가중 평균: 처리량 60%, 활용도 40%
        kpi_score = (throughput * 0.6) + (utilization * 0.4)
        return round(kpi_score, 2)
    
    def _calculate_roi_actual(self, profit: float, cost: float) -> float:
        """
        실제 ROI 계산 (이익/비용 * 100)
        """
        if cost == 0:
            return 0.0
        roi = (profit / cost) * 100
        return round(roi, 2)

    def integrate_analytics_features(self, base_features_df: pd.DataFrame, 
                                   analytics_data: Dict[int, Dict]) -> pd.DataFrame:
        """
        AnalyticsUtil에서 추출한 피처들을 기본 피처셋에 통합
        
        Args:
            base_features_df: 기본 28개 피처 데이터프레임
            analytics_data: AnalyticsUtil에서 계산된 데이터
            
        Returns:
            통합된 피처 데이터프레임
        """
        print("🔗 AnalyticsUtil 피처 통합 중...")
        
        enhanced_df = base_features_df.copy()
        
        # AnalyticsUtil 피처 추가
        for idx, row in enhanced_df.iterrows():
            scenario_id = row['scenario_id']
            
            # analytics_data에서 해당 시나리오 데이터 찾기
            if scenario_id in analytics_data:
                data = analytics_data[scenario_id]
                
                # KPI 피처 추가 (7개)
                if 'kpis' in data:
                    enhanced_df.at[idx, 'analytics_throughput'] = data['kpis'].get('throughput', 0)
                    enhanced_df.at[idx, 'analytics_waiting_time'] = data['kpis'].get('waiting_time', 0)
                    enhanced_df.at[idx, 'analytics_lead_time'] = data['kpis'].get('lead_time', 0)
                    enhanced_df.at[idx, 'analytics_resource_utilization'] = data['kpis'].get('resource_utilization', 0)
                    enhanced_df.at[idx, 'analytics_hourly_output'] = data['kpis'].get('hourly_output', 0)
                    enhanced_df.at[idx, 'analytics_bss'] = data['kpis'].get('bss', 0)
                    enhanced_df.at[idx, 'analytics_bps'] = data['kpis'].get('bps', 0)
                
                # Efficiency 피처 추가 (6개)
                if 'efficiencys' in data:
                    enhanced_df.at[idx, 'analytics_production_line_output_rate'] = data['efficiencys'].get('Production_line_output_rate', 0)
                    enhanced_df.at[idx, 'analytics_source_efficiency'] = data['efficiencys'].get('Source_efficiency', 0)
                    enhanced_df.at[idx, 'analytics_workstation_efficiency'] = data['efficiencys'].get('Workstation_efficiency', 0)
                    enhanced_df.at[idx, 'analytics_production_line_efficiency'] = data['efficiencys'].get('Production_line_efficiency', 0)
                    enhanced_df.at[idx, 'analytics_total_store_rate'] = data['efficiencys'].get('total_store_rate', 0)
                    enhanced_df.at[idx, 'analytics_energy'] = data['efficiencys'].get('energy', 0)
                
                # 기타 피처 추가 (2개)
                enhanced_df.at[idx, 'analytics_area_usage_rate'] = data.get('area_usage_rates', 0)
                enhanced_df.at[idx, 'analytics_logistic_traffic'] = data.get('logistic_traffic', 0)
        
        print(f"✅ 피처 통합 완료: {base_features_df.shape[1]}개 → {enhanced_df.shape[1]}개 피처")
        return enhanced_df

    def create_complete_training_dataset(self, scenarios: List[Dict], 
                                       simulation_results: Dict[int, str],
                                       analytics_data: Dict[int, Dict]) -> pd.DataFrame:
        """
        시뮬레이션 전/후 데이터를 모두 결합한 완전한 훈련 데이터셋 생성
        
        Args:
            scenarios: 시뮬레이션 전 시나리오들
            simulation_results: 언리얼 시뮬레이션 결과들 (scenario_id: result_text)
            analytics_data: AnalyticsUtil 계산 결과들 (scenario_id: analytics_dict)
            
        Returns:
            완전한 훈련 데이터셋
        """
        print("🎯 완전한 XGB 훈련 데이터셋 생성 중...")
        
        # 1단계: 기본 피처 생성
        base_df = self.create_training_dataset_structure(scenarios)
        
        # 2단계: AnalyticsUtil 피처 통합
        enhanced_df = self.integrate_analytics_features(base_df, analytics_data)
        
        # 3단계: 시뮬레이션 결과 Y-Label 추가
        for idx, row in enhanced_df.iterrows():
            scenario_id = row['scenario_id']
            
            if scenario_id in simulation_results:
                # 언리얼 결과 파싱
                parsed_results = self.process_simulation_results(
                    simulation_results[scenario_id], scenario_id
                )
                
                # Y-Label 업데이트
                enhanced_df.at[idx, 'kpi_score'] = parsed_results['kpi_score']
                enhanced_df.at[idx, 'roi_actual'] = parsed_results['roi_actual'] 
                enhanced_df.at[idx, 'eff_score'] = parsed_results['eff_score']
        
        # 결측치 제거 (시뮬레이션 결과가 없는 케이스)
        complete_df = enhanced_df.dropna(subset=['kpi_score', 'roi_actual', 'eff_score'])
        
        print(f"🎯 완전한 XGB 데이터셋 완성!")
        print(f"   • 총 피처 수: {complete_df.shape[1] - 8}개")  # 메타데이터 5개 + Y-Label 3개 제외
        print(f"   • 기본 피처: 28개")
        print(f"   • AnalyticsUtil 피처: 15개") 
        print(f"   • Y-Label: 3개 (KPI, ROI, EFF)")
        print(f"   • 훈련 샘플: {complete_df.shape[0]:,}개")
        
        return complete_df
    
    def generate_raw_data_csv(self, num_samples: int = 2000, output_file: str = "warehouse_raw_data.csv") -> pd.DataFrame:
        """
        시뮬레이션 없이 바로 사용 가능한 로우데이터 CSV 생성
        warehouse_data_2.csv와 유사한 형태이지만 디지털 트윈 관점 반영
        
        Args:
            num_samples: 생성할 데이터 샘플 수
            output_file: 저장할 CSV 파일명
            
        Returns:
            생성된 DataFrame
        """
        import numpy as np
        from datetime import datetime
        
        print(f"🏭 디지털 트윈 로우데이터 생성 중... ({num_samples:,}개 샘플)")
        
        raw_data = []
        
        for i in range(num_samples):
            if i % 500 == 0:
                print(f"   진행률: {i:,}/{num_samples:,} ({i/num_samples*100:.1f}%)")
            
            # 기본 창고 특성
            warehouse_area = np.random.choice([50000000, 100000000, 150000000, 300000000])  # cm²
            warehouse_size_m2 = warehouse_area / 10000  # m²로 변환
            
            # 운영 강도 (시즌/피크 등 반영)
            season = np.random.choice([1, 2, 3, 4])  # 1:비수기, 2:평시, 3:성수기, 4:피크
            operation_intensity = {1: 0.6, 2: 1.0, 3: 1.8, 4: 2.2}[season]
            
            # 장비 구성
            automation_level = np.random.choice([2, 3, 4, 5])  # 기본형, 표준형, 고급형, 최고급형
            
            # 장비별 수량 (자동화 레벨에 따라)
            equipment_ranges = {
                2: {"cnv": (5, 15), "rtv": (2, 8), "srm": (3, 12), "robot": (0, 2)},
                3: {"cnv": (10, 25), "rtv": (4, 12), "srm": (5, 15), "robot": (1, 3)},
                4: {"cnv": (15, 35), "rtv": (6, 18), "srm": (8, 20), "robot": (1, 4)},
                5: {"cnv": (20, 50), "rtv": (10, 25), "srm": (10, 30), "robot": (2, 6)}
            }
            
            ranges = equipment_ranges[automation_level]
            cnv_count = np.random.randint(ranges["cnv"][0], ranges["cnv"][1] + 1)
            rtv_count = np.random.randint(ranges["rtv"][0], ranges["rtv"][1] + 1)
            srm_count = np.random.randint(ranges["srm"][0], ranges["srm"][1] + 1)
            robot_count = np.random.randint(ranges["robot"][0], ranges["robot"][1] + 1)
            
            # 랙 수량
            rack_count = max(10, int(warehouse_area / 5000000))
            
            # 기본 처리량
            base_throughput = warehouse_size_m2 * np.random.uniform(0.8, 1.2)
            daily_throughput = base_throughput * operation_intensity
            
            # 장비 상태 (노후도 반영)
            equipment_age = np.random.randint(1, 20)  # 년
            condition_factor = max(0.6, 1.0 - (equipment_age * 0.02))  # 나이에 따른 성능 저하
            
            # 실제 성능 지표 계산
            noise_factor = np.random.normal(1.0, 0.1)  # ±10% 노이즈
            
            # 처리 시간 (초)
            base_time = 5 + (equipment_age * 0.3)
            processing_time = base_time * (2.0 - condition_factor) * noise_factor / operation_intensity
            processing_time = max(5.0, min(processing_time, 6500.0))  # 현실적 범위
            
            # 정확도 (%)
            base_accuracy = 95.0
            accuracy_penalty = (equipment_age * 0.5) + (1.0 - condition_factor) * 10
            picking_accuracy = base_accuracy - accuracy_penalty + (automation_level - 2) * 2
            picking_accuracy = max(72.0, min(picking_accuracy, 98.0))
            
            # 오류율 (%)
            base_error = 1.0
            error_increase = (equipment_age * 0.1) + (1.0 - condition_factor) * 3
            error_rate = base_error + error_increase - (automation_level - 2) * 0.3
            error_rate = max(0.5, min(error_rate, 8.0))
            
            # 비용 (원화 기준으로 변경)
            labor_base = 50000  # 기본 인건비
            equipment_overhead = (cnv_count * 1000 + rtv_count * 2000 + srm_count * 3000 + robot_count * 5000)
            labor_cost = labor_base + equipment_overhead * (2.0 - condition_factor)
            labor_cost = max(35000, min(labor_cost, 80000))
            
            # 부가 특성들
            workers = max(5, int((cnv_count + rtv_count + srm_count) * 0.8))
            shift_type = np.random.choice([1, 2, 3])  # 1교대, 2교대, 3교대
            day_of_week = np.random.choice([1, 2, 3, 4, 5, 6, 7])
            
            # WMS/AGV 구현 여부 (자동화 레벨과 상관관계)
            wms_prob = min(0.9, automation_level * 0.2)
            agv_prob = min(0.8, (automation_level - 1) * 0.25)
            wms_implemented = 1 if np.random.random() < wms_prob else 0
            agv_implemented = 1 if np.random.random() < agv_prob else 0
            
            # 로우데이터 샘플 생성
            sample = {
                # 창고 기본 특성
                'warehouse_area_m2': int(warehouse_size_m2),
                'daily_throughput': int(daily_throughput),
                'total_equipment_count': cnv_count + rtv_count + srm_count + robot_count,
                'workers': workers,
                'shift_type': shift_type,
                
                # 자동화 설비
                'conveyor_count': cnv_count,
                'rtv_agv_count': rtv_count, 
                'srm_count': srm_count,
                'robot_arm_count': robot_count,
                'rack_count': rack_count,
                
                # 시스템 구현
                'wms_implemented': wms_implemented,
                'agv_implemented': agv_implemented,
                'automation_level': automation_level,
                
                # 운영 조건
                'season': season,
                'day_of_week': day_of_week,
                'equipment_age_years': equipment_age,
                'condition_factor': round(condition_factor, 3),
                
                # 장비 성능 지표 (핵심)
                'conveyor_utilization': round(np.random.uniform(0.3, 0.8), 3),
                'rtv_utilization': round(np.random.uniform(0.4, 0.7), 3),
                'srm_utilization': round(np.random.uniform(0.5, 0.9), 3),
                'robot_utilization': round(np.random.uniform(0.2, 0.6), 3),
                
                # 시간 지표
                'avg_conveyor_time': round(np.random.uniform(3.0, 8.0), 2),
                'avg_rtv_time': round(np.random.uniform(6.0, 12.0), 2),
                'avg_srm_time': round(np.random.uniform(4.0, 10.0), 2),
                'avg_robot_time': round(np.random.uniform(1.0, 5.0), 2),
                
                # 품질 지표
                'equipment_efficiency': round(condition_factor * 100, 1),
                'maintenance_frequency': round(equipment_age * 0.5, 1),
                
                # === Y-Label (타겟 변수) ===
                'processing_time_seconds': round(processing_time, 2),
                'picking_accuracy_percent': round(picking_accuracy, 2),
                'error_rate_percent': round(error_rate, 2),
                'labor_cost_per_order_krw': int(labor_cost)
            }
            
            raw_data.append(sample)
        
        # DataFrame 생성
        df = pd.DataFrame(raw_data)
        
        # CSV 저장
        df.to_csv(output_file, index=False)
        
        print(f"✅ 로우데이터 CSV 생성 완료!")
        print(f"   📄 파일: {output_file}")
        print(f"   📊 크기: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"   🎯 X-Label: {df.shape[1] - 4}개 (마지막 4개는 Y-Label)")
        print(f"   🎯 Y-Label: processing_time, picking_accuracy, error_rate, labor_cost")
        
        # 데이터 품질 체크
        print(f"\n📈 Y-Label 분포:")
        print(f"   • 처리시간: {df['processing_time_seconds'].min():.1f}~{df['processing_time_seconds'].max():.1f}초")
        print(f"   • 정확도: {df['picking_accuracy_percent'].min():.1f}~{df['picking_accuracy_percent'].max():.1f}%")
        print(f"   • 오류율: {df['error_rate_percent'].min():.1f}~{df['error_rate_percent'].max():.1f}%") 
        print(f"   • 비용: {df['labor_cost_per_order_krw'].min():,}~{df['labor_cost_per_order_krw'].max():,}원")
        
        return df

def main_bigdata_strategy():
    """
    빅데이터 구축 전략 실행 (시뮬레이션 전/후 통합 방식)
    """
    generator = DigitalTwinBigDataGenerator()
    
    print("🏗️  디지털 트윈 빅데이터 구축 전략 v2.0")
    print("=" * 60)
    print("📌 시뮬레이션 전/후 데이터 통합형 XGB 훈련셋 구축")
    
    # Step 1: 시나리오 생성
    print("\n📋 1단계: 체계적 시나리오 생성")
    scenarios = generator.generate_simulation_scenarios(num_scenarios_per_combination=5)  # 데모용 5개
    
    # Step 2: 언리얼 시뮬레이션용 파일 저장
    print("\n💾 2단계: 언리얼 시뮬레이션 설정 파일 생성")
    generator.save_scenarios_for_unreal_simulation(scenarios[:20])  # 데모용 20개만
    
    # Step 3: 시뮬레이션 전 피처셋 생성
    print("\n🔧 3단계: 시뮬레이션 전 피처셋 생성")
    pre_simulation_df = generator.create_training_dataset_structure(scenarios[:20])
    
    # Step 4: Mock 시뮬레이션 결과 생성 (실제로는 언리얼에서 받아옴)
    print("\n🎭 4단계: 시뮬레이션 결과 데모 생성")
    mock_simulation_results = {}
    mock_analytics_data = {}
    
    for i in range(1, 21):  # 20개 시나리오
        # 언리얼 결과 시뮬레이션
        throughput = np.random.uniform(5.0, 15.0)
        utilization = np.random.uniform(80.0, 98.0)
        cost = np.random.uniform(3000.0, 8000.0)
        profit = np.random.uniform(10.0, 50.0)
        efficiency = np.random.uniform(85.0, 97.0)
        
        mock_simulation_results[i] = f"""KPI: 성과는 {throughput:.1f}의 처리량과 {utilization:.1f}%의 활용도로 나타납니다.
ROI: 비용 {cost:.2f} 달러에 대해 {profit:.1f}의 이익을 기록했습니다.
EFF: 효율성 점수는 {efficiency:.1f}%입니다.

대기 시간과 지연 시간을 줄이기 위해 로봇의 작업 스케줄을 최적화할 필요가 있습니다."""
        
        # AnalyticsUtil 결과 시뮬레이션
        mock_analytics_data[i] = {
            'kpis': {
                'throughput': np.random.uniform(1, 10),
                'waiting_time': np.random.uniform(1, 10), 
                'lead_time': np.random.uniform(1, 10),
                'resource_utilization': np.random.uniform(1, 10),
                'hourly_output': np.random.uniform(1, 10),
                'bss': np.random.uniform(1, 10),
                'bps': np.random.uniform(1, 10)
            },
            'efficiencys': {
                'Production_line_output_rate': np.random.uniform(1, 10),
                'Source_efficiency': np.random.uniform(1, 10),
                'Workstation_efficiency': np.random.uniform(1, 10), 
                'Production_line_efficiency': np.random.uniform(1, 10),
                'total_store_rate': np.random.uniform(1, 10),
                'energy': np.random.uniform(1, 10)
            },
            'area_usage_rates': np.random.uniform(50.0, 95.0),
            'logistic_traffic': np.random.randint(100, 1000)
        }
    
    # Step 5: 완전한 훈련 데이터셋 생성
    print("\n🎯 5단계: 완전한 XGB 훈련 데이터셋 생성")
    complete_df = generator.create_complete_training_dataset(
        scenarios[:20], 
        mock_simulation_results,
        mock_analytics_data
    )
    
    # Step 6: XGBoost 다중 회귀 가능성 확인
    print(f"\n🤖 6단계: XGBoost MultiOutput Regressor 준비성 체크")
    
    # 피처 컬럼과 타겟 컬럼 분리
    feature_cols = [col for col in complete_df.columns 
                   if col not in ['scenario_id', 'warehouse_type', 'equipment_config', 
                                 'operation_scenario', 'equipment_condition', 
                                 'kpi_score', 'roi_actual', 'eff_score']]
    
    target_cols = ['kpi_score', 'roi_actual', 'eff_score']
    
    print(f"   ✅ X-Label (피처): {len(feature_cols)}개")
    print(f"      • 기본 피처: 28개")
    print(f"      • AnalyticsUtil 피처: 15개")
    print(f"   ✅ Y-Label (타겟): {len(target_cols)}개")
    print(f"      • KPI Score: 처리량 + 활용도 종합")
    print(f"      • ROI Actual: 실제 투자수익률")
    print(f"      • EFF Score: 효율성 점수")
    print(f"   ✅ 훈련 샘플: {complete_df.shape[0]}개")
    
    # 데이터 품질 체크
    print(f"\n📊 데이터 품질 체크:")
    print(f"   • 결측치: {complete_df.isnull().sum().sum()}개")
    print(f"   • Y-Label 범위:")
    for col in target_cols:
        print(f"     - {col}: {complete_df[col].min():.2f} ~ {complete_df[col].max():.2f}")
    
    print(f"\n💡 실제 구현 로드맵:")
    total_scenarios = len(scenarios)
    print(f"   1️⃣ 언리얼 시뮬레이션: {total_scenarios:,}개 시나리오 실행")
    print(f"   2️⃣ AnalyticsUtil 연동: 추가 피처 15개 추출")
    print(f"   3️⃣ XGB MultiOutputRegressor 훈련:")
    print(f"      from sklearn.multioutput import MultiOutputRegressor")
    print(f"      from xgboost import XGBRegressor")
    print(f"      model = MultiOutputRegressor(XGBRegressor())")
    print(f"   4️⃣ 3개 타겟 동시 예측: KPI, ROI, EFF")
    
    return scenarios, complete_df

def demo_xgb_multioutput_training(complete_df: pd.DataFrame) -> None:
    """
    XGBoost MultiOutput Regressor 데모 훈련
    """
    print("\n🚀 XGBoost MultiOutput Regressor 데모 훈련")
    print("=" * 50)
    
    try:
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
        import xgboost as xgb
        
        # 피처와 타겟 분리
        feature_cols = [col for col in complete_df.columns 
                       if col not in ['scenario_id', 'warehouse_type', 'equipment_config', 
                                     'operation_scenario', 'equipment_condition', 
                                     'kpi_score', 'roi_actual', 'eff_score']]
        
        X = complete_df[feature_cols].fillna(0)  # 결측치 처리
        y = complete_df[['kpi_score', 'roi_actual', 'eff_score']]
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # MultiOutput XGBoost 모델
        model = MultiOutputRegressor(
            xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        )
        
        # 훈련
        print("🔄 모델 훈련 중...")
        model.fit(X_train, y_train)
        
        # 예측
        y_pred = model.predict(X_test)
        
        # 평가
        print("\n📊 모델 성능 평가:")
        target_names = ['KPI Score', 'ROI Actual', 'EFF Score']
        
        for i, target_name in enumerate(target_names):
            mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
            r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
            print(f"   • {target_name}:")
            print(f"     - MSE: {mse:.4f}")
            print(f"     - R²: {r2:.4f}")
        
        print("\n✅ XGBoost MultiOutput 회귀 성공적으로 완료!")
        
    except ImportError as e:
        print(f"❌ 필요한 라이브러리가 설치되지 않음: {e}")
        print("   설치 명령: pip install scikit-learn xgboost")
    except Exception as e:
        print(f"❌ 훈련 중 오류 발생: {e}")

def generate_raw_data_only(num_samples: int = 2000):
    """
    로우데이터 CSV만 생성하는 간단한 함수
    """
    generator = DigitalTwinBigDataGenerator()
    
    print("🏭 디지털 트윈 기반 로우데이터 생성 모드")
    print("=" * 50)
    
    # 로우데이터 CSV 생성
    raw_df = generator.generate_raw_data_csv(num_samples, "warehouse_raw_data.csv")
    
    print(f"\n🎯 생성 완료! XGBoost 훈련 즉시 가능")
    print(f"   📄 파일: warehouse_raw_data.csv") 
    print(f"   🔍 사용법: pd.read_csv('warehouse_raw_data.csv')")
    
    return raw_df

if __name__ == "__main__":
    import sys
    
    # 실행 모드 선택
    if len(sys.argv) > 1 and sys.argv[1] == "--raw-only":
        # 로우데이터만 생성
        num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
        raw_df = generate_raw_data_only(num_samples)
    else:
        # 기본 모드: 시뮬레이션 통합 데이터셋
        scenarios, complete_df = main_bigdata_strategy()
        
        # XGBoost 데모 실행
        if complete_df.shape[0] > 0:
            demo_xgb_multioutput_training(complete_df)