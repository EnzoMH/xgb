"""
warehouse_raw_data.csv 형태의 가상 창고 데이터 생성기
디지털 트윈 장비 중심 데이터셋 생성
"""

import pandas as pd
import numpy as np

# 시드 설정 (재현성을 위해)
np.random.seed(42)

def generate_warehouse_raw_data(n_samples=1000):
    """
    warehouse_raw_data.csv와 동일한 구조의 가상 데이터 생성
    - 디지털 트윈 장비 중심 데이터
    - 31개 컬럼 (27개 X-Label + 4개 Y-Label)
    
    Args:
        n_samples (int): 생성할 샘플 수
        
    Returns:
        pd.DataFrame: 생성된 창고 데이터
    """
    
    data = []
    
    for i in range(n_samples):
        # === 기본 창고 특성 ===
        # warehouse_area_m2 (창고 면적)
        warehouse_area_m2 = np.random.choice([5000, 10000, 15000, 30000])  # m²
        
        # daily_throughput (일일 처리량)
        base_throughput = warehouse_area_m2 * np.random.uniform(0.8, 1.2)
        operation_intensity = np.random.choice([0.6, 1.0, 1.8, 2.2])  # 운영 강도
        daily_throughput = int(base_throughput * operation_intensity)
        
        # automation_level (자동화 수준)
        automation_level = np.random.choice([2, 3, 4, 5])  # 기본형~최고급형
        
        # 장비별 수량 (자동화 레벨에 따라)
        equipment_ranges = {
            2: {"cnv": (5, 15), "rtv": (2, 8), "srm": (3, 12), "robot": (0, 2)},
            3: {"cnv": (10, 25), "rtv": (4, 12), "srm": (5, 15), "robot": (1, 3)},
            4: {"cnv": (15, 35), "rtv": (6, 18), "srm": (8, 20), "robot": (1, 4)},
            5: {"cnv": (20, 50), "rtv": (10, 25), "srm": (10, 30), "robot": (2, 6)}
        }
        
        ranges = equipment_ranges[automation_level]
        conveyor_count = np.random.randint(ranges["cnv"][0], ranges["cnv"][1] + 1)
        rtv_agv_count = np.random.randint(ranges["rtv"][0], ranges["rtv"][1] + 1)
        srm_count = np.random.randint(ranges["srm"][0], ranges["srm"][1] + 1)
        robot_arm_count = np.random.randint(ranges["robot"][0], ranges["robot"][1] + 1)
        
        total_equipment_count = conveyor_count + rtv_agv_count + srm_count + robot_arm_count
        rack_count = max(10, int(warehouse_area_m2 / 500))  # 면적 대비 랙 수
        
        # workers (작업자 수)
        workers = max(5, int(total_equipment_count * np.random.uniform(0.6, 1.2)))
        
        # 기본 설정
        shift_type = np.random.choice([1, 2, 3])  # 1교대, 2교대, 3교대
        season = np.random.choice([1, 2, 3, 4])  # 비수기, 평시, 성수기, 피크
        day_of_week = np.random.choice([1, 2, 3, 4, 5, 6, 7])
        
        # WMS/AGV 구현 여부 (자동화 레벨과 상관관계)
        wms_prob = min(0.9, automation_level * 0.2)
        agv_prob = min(0.8, (automation_level - 1) * 0.25) 
        wms_implemented = 1 if np.random.random() < wms_prob else 0
        agv_implemented = 1 if np.random.random() < agv_prob else 0
        
        # === 장비 상태 및 성능 지표 ===
        equipment_age_years = np.random.randint(1, 20)  # 장비 연식
        condition_factor = max(0.6, 1.0 - (equipment_age_years * 0.02))  # 상태 계수
        
        # 장비별 활용도 (Utilization)
        conveyor_utilization = round(np.random.uniform(0.3, 0.8), 3)
        rtv_utilization = round(np.random.uniform(0.4, 0.7), 3) 
        srm_utilization = round(np.random.uniform(0.5, 0.9), 3)
        robot_utilization = round(np.random.uniform(0.2, 0.6), 3)
        
        # 장비별 평균 작업 시간 (초)
        avg_conveyor_time = round(np.random.uniform(3.0, 8.0), 2)
        avg_rtv_time = round(np.random.uniform(6.0, 12.0), 2)
        avg_srm_time = round(np.random.uniform(4.0, 10.0), 2)
        avg_robot_time = round(np.random.uniform(1.0, 5.0), 2)
        
        # 종합 효율성 지표
        equipment_efficiency = round(condition_factor * 100, 1)
        maintenance_frequency = round(equipment_age_years * 0.5, 1)
        
        # === Y-Label (타겟 변수) 계산 ===
        noise_factor = np.random.normal(1.0, 0.1)  # ±10% 노이즈
        
        # 1. processing_time_seconds (처리시간)
        base_time = 5 + (equipment_age_years * 0.3)
        operation_multiplier = {1: 1.2, 2: 1.0, 3: 0.8, 4: 0.6}[season]  # 계절별
        processing_time_seconds = base_time * (2.0 - condition_factor) * noise_factor * operation_multiplier
        processing_time_seconds = round(max(5.0, min(processing_time_seconds, 30.0)), 2)
        
        # 2. picking_accuracy_percent (피킹 정확도)
        base_accuracy = 95.0
        accuracy_penalty = (equipment_age_years * 0.5) + (1.0 - condition_factor) * 10
        picking_accuracy_percent = base_accuracy - accuracy_penalty + (automation_level - 2) * 2
        if wms_implemented:
            picking_accuracy_percent += np.random.uniform(2, 5)
        if agv_implemented:
            picking_accuracy_percent += np.random.uniform(1, 3)
        picking_accuracy_percent = round(max(75.0, min(picking_accuracy_percent, 98.0)), 1)
        
        # 3. error_rate_percent (오류율)
        base_error = 1.0
        error_increase = (equipment_age_years * 0.1) + (1.0 - condition_factor) * 3
        error_rate_percent = base_error + error_increase - (automation_level - 2) * 0.3
        if wms_implemented:
            error_rate_percent *= 0.7
        if agv_implemented:
            error_rate_percent *= 0.8
        error_rate_percent = round(max(0.5, min(error_rate_percent, 5.0)), 2)
        
        # 4. labor_cost_per_order_krw (주문당 인건비)
        labor_base = 50000  # 기본 인건비
        equipment_overhead = (conveyor_count * 1000 + rtv_agv_count * 2000 + 
                            srm_count * 3000 + robot_arm_count * 5000)
        labor_cost_per_order_krw = int(labor_base + equipment_overhead * (2.0 - condition_factor))
        labor_cost_per_order_krw = max(35000, min(labor_cost_per_order_krw, 85000))
        
        # warehouse_raw_data.csv와 동일한 구조로 데이터 추가
        data.append({
            # === X-Label (27개 피처) ===
            'warehouse_area_m2': warehouse_area_m2,
            'daily_throughput': daily_throughput,
            'total_equipment_count': total_equipment_count,
            'workers': workers,
            'shift_type': shift_type,
            'conveyor_count': conveyor_count,
            'rtv_agv_count': rtv_agv_count,
            'srm_count': srm_count,
            'robot_arm_count': robot_arm_count,
            'rack_count': rack_count,
            'wms_implemented': wms_implemented,
            'agv_implemented': agv_implemented,
            'automation_level': automation_level,
            'season': season,
            'day_of_week': day_of_week,
            'equipment_age_years': equipment_age_years,
            'condition_factor': condition_factor,
            'conveyor_utilization': conveyor_utilization,
            'rtv_utilization': rtv_utilization,
            'srm_utilization': srm_utilization,
            'robot_utilization': robot_utilization,
            'avg_conveyor_time': avg_conveyor_time,
            'avg_rtv_time': avg_rtv_time,
            'avg_srm_time': avg_srm_time,
            'avg_robot_time': avg_robot_time,
            'equipment_efficiency': equipment_efficiency,
            'maintenance_frequency': maintenance_frequency,
            
            # === Y-Label (4개 타겟) ===
            'processing_time_seconds': processing_time_seconds,
            'picking_accuracy_percent': picking_accuracy_percent,
            'error_rate_percent': error_rate_percent,
            'labor_cost_per_order_krw': labor_cost_per_order_krw
        })
    
    return pd.DataFrame(data)


def save_warehouse_data(df, filename='warehouse_synthetic_data.csv'):
    """
    생성된 창고 데이터를 CSV 파일로 저장
    
    Args:
        df (pd.DataFrame): 저장할 데이터프레임
        filename (str): 저장할 파일명
        
    Returns:
        str: 저장된 파일명
    """
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    return filename


def generate_and_save_data(n_samples=2000, filename='warehouse_synthetic_data.csv', verbose=True):
    """
    창고 데이터 생성 및 저장을 한 번에 수행
    
    Args:
        n_samples (int): 생성할 샘플 수
        filename (str): 저장할 파일명
        verbose (bool): 상세 정보 출력 여부
        
    Returns:
        pd.DataFrame: 생성된 데이터프레임
    """
    
    if verbose:
        print(f"🏭 창고 가상 데이터 생성 중... ({n_samples:,}개 샘플)")
    
    # 데이터 생성
    df = generate_warehouse_raw_data(n_samples)
    
    if verbose:
        print(f"✅ 데이터 생성 완료!")
        print(f"   📊 크기: {df.shape}")
        print(f"   🎯 X-Label: 27개")
        print(f"   🎯 Y-Label: 4개")
        
        # 기본 통계
        print(f"\n📈 Y-Label 분포:")
        print(f"   • 처리시간: {df['processing_time_seconds'].min():.1f}~{df['processing_time_seconds'].max():.1f}초")
        print(f"   • 정확도: {df['picking_accuracy_percent'].min():.1f}~{df['picking_accuracy_percent'].max():.1f}%")
        print(f"   • 오류율: {df['error_rate_percent'].min():.1f}~{df['error_rate_percent'].max():.1f}%")
        print(f"   • 비용: {df['labor_cost_per_order_krw'].min():,}~{df['labor_cost_per_order_krw'].max():,}원")
        
        print(f"\n💡 기술 도입 현황:")
        print(f"   • WMS 도입률: {df['wms_implemented'].mean():.1%}")
        print(f"   • AGV 도입률: {df['agv_implemented'].mean():.1%}")
        print(f"   • 동시 도입률: {((df['wms_implemented']==1) & (df['agv_implemented']==1)).mean():.1%}")
    
    # 파일 저장
    save_warehouse_data(df, filename)
    
    if verbose:
        print(f"\n💾 파일 저장 완료: {filename}")
        print(f"🎯 사용법:")
        print(f"   import pandas as pd")
        print(f"   df = pd.read_csv('{filename}')")
        print(f"   # 바로 ML 모델 훈련 가능!")
    
    return df


def get_feature_target_columns():
    """
    X-Label과 Y-Label 컬럼명 반환
    
    Returns:
        tuple: (feature_columns, target_columns)
    """
    feature_columns = [
        'warehouse_area_m2', 'daily_throughput', 'total_equipment_count', 'workers', 'shift_type',
        'conveyor_count', 'rtv_agv_count', 'srm_count', 'robot_arm_count', 'rack_count',
        'wms_implemented', 'agv_implemented', 'automation_level', 'season', 'day_of_week',
        'equipment_age_years', 'condition_factor', 'conveyor_utilization', 'rtv_utilization',
        'srm_utilization', 'robot_utilization', 'avg_conveyor_time', 'avg_rtv_time',
        'avg_srm_time', 'avg_robot_time', 'equipment_efficiency', 'maintenance_frequency'
    ]
    
    target_columns = [
        'processing_time_seconds', 'picking_accuracy_percent', 
        'error_rate_percent', 'labor_cost_per_order_krw'
    ]
    
    return feature_columns, target_columns


# 직접 실행시 데모 데이터 생성
if __name__ == "__main__":
    print("🚀 data.py 직접 실행 모드")
    print("=" * 40)
    
    # 데모 데이터 생성 (1000개 샘플)
    demo_df = generate_and_save_data(
        n_samples=1000, 
        filename='warehouse_demo_data.csv',
        verbose=True
    )
    
    print(f"\n✅ 데모 완료! 생성된 파일을 확인하세요.")
