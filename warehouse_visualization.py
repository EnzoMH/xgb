# 📊 창고 운영 데이터 시각화 분석
# 확장된 현실적 창고 데이터 기반 종합 시각화

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 시드 설정
np.random.seed(42)

# 데이터 생성 함수 (xgb.py에서 가져온 확장된 버전)
def generate_realistic_warehouse_data(n_samples=1000):
    """
    현실적인 창고 운영 데이터 생성 (확장 버전)
    - 실제 물류센터 운영 지표에 기반
    - 25개 특성 변수로 확장
    """
    
    data = []
    
    for i in range(n_samples):
        # === 기본 창고 특성 ===
        warehouse_size = np.random.lognormal(np.log(20000), 0.5)
        warehouse_size = np.clip(warehouse_size, 3000, 80000)
        
        # 창고 크기에 따른 주문량 (상관관계 반영)
        base_orders = warehouse_size * np.random.uniform(0.01, 0.05)
        daily_orders = int(np.random.normal(base_orders, base_orders * 0.3))
        daily_orders = np.clip(daily_orders, 20, 1200)
        
        # 아이템 종류 수 (주문량과 어느 정도 상관)
        item_types = int(np.random.uniform(50, min(3000, daily_orders * 8)))
        
        # 작업자 수 (주문량 기반, 현실적 비율)
        base_workers = daily_orders / np.random.uniform(8, 25)
        workers = int(np.random.normal(base_workers, base_workers * 0.2))
        workers = np.clip(workers, 5, 150)
        
        shift_type = np.random.choice([1, 2, 3], p=[0.2, 0.5, 0.3])
        
        # 시스템 도입 상태
        wms_implemented = np.random.choice([0, 1], p=[0.4, 0.6])
        agv_implemented = np.random.choice([0, 1], p=[0.75, 0.25])
        
        # 계절성 및 요일 효과
        season = np.random.choice([1, 2, 3, 4])
        day_of_week = np.random.choice([1, 2, 3, 4, 5, 6, 7])
        
        # === 추가 영향 변수들 ===
        # 물리적 환경
        temperature_control = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])  # 상온, 냉장, 냉동
        ceiling_height = np.random.uniform(6, 15)  # 천장 높이 (m)
        loading_dock_count = int(warehouse_size / 5000) + np.random.randint(1, 4)  # 상하차장 수
        warehouse_age = np.random.randint(1, 25)  # 창고 연식
        
        # 재고/상품 특성
        average_item_weight = np.random.lognormal(0.5, 0.8)  # 평균 상품 무게 (kg)
        fragile_items_ratio = np.random.uniform(0, 0.3)  # 파손위험 상품 비율
        inventory_turnover = np.random.uniform(2, 20)  # 연간 재고 회전율
        
        # 기술 수준
        conveyor_belt_ratio = np.random.uniform(0, 0.8) if wms_implemented else np.random.uniform(0, 0.3)
        voice_picking = np.random.choice([0, 1], p=[0.7, 0.3]) if wms_implemented else 0
        barcode_scanner_quality = np.random.randint(3, 6) if wms_implemented else np.random.randint(2, 5)
        
        # 인력 특성
        average_employee_experience = np.random.uniform(3, 60)  # 평균 경험 (개월)
        employee_turnover_rate = np.random.uniform(10, 80)  # 이직률 (%)
        temp_worker_ratio = np.random.uniform(0, 0.4)  # 임시직 비율
        
        # 운영 정책
        batch_picking_enabled = np.random.choice([0, 1], p=[0.4, 0.6])
        quality_check_level = np.random.randint(1, 6)  # 품질검사 강도
        order_complexity_score = daily_orders / item_types if item_types > 0 else 1  # 주문 복잡도
        
        # === 기본 처리 시간 계산 ===
        complexity_factor = min(2.0, 1 + (item_types / 2000))
        
        workload_per_worker = daily_orders / workers
        if workload_per_worker > 20:
            efficiency_penalty = 1 + (workload_per_worker - 20) * 0.1
        else:
            efficiency_penalty = 1.0
        
        base_processing_time = (
            daily_orders * 0.8 * complexity_factor * efficiency_penalty +
            np.random.normal(0, 15)
        )
        
        # === 추가 변수들이 처리 시간에 미치는 영향 ===
        # 온도 관리 영향
        if temperature_control == 2:  # 냉동
            base_processing_time *= np.random.uniform(1.2, 1.3)
        elif temperature_control == 1:  # 냉장
            base_processing_time *= np.random.uniform(1.05, 1.15)
        
        # 상품 무게 영향
        weight_factor = min(1.5, 1 + (average_item_weight - 1) * 0.1)
        base_processing_time *= weight_factor
        
        # 파손위험 상품 영향
        base_processing_time *= (1 + fragile_items_ratio * 0.2)
        
        # 직원 경험 영향
        experience_factor = max(0.7, 1 - (average_employee_experience / 100))
        base_processing_time *= experience_factor
        
        # 이직률 영향
        turnover_factor = 1 + (employee_turnover_rate / 200)
        base_processing_time *= turnover_factor
        
        # 배치 피킹 효과
        if batch_picking_enabled:
            base_processing_time *= np.random.uniform(0.85, 0.95)
        
        # 음성 피킹 효과
        if voice_picking:
            base_processing_time *= np.random.uniform(0.9, 0.95)
        
        # 창고 노후도 영향
        age_factor = 1 + (warehouse_age / 100)
        base_processing_time *= age_factor
        
        # 재고 회전율 영향
        turnover_factor = max(0.8, 1 - (inventory_turnover / 50))
        base_processing_time *= turnover_factor
        
        # WMS/AGV 효과
        if wms_implemented:
            wms_efficiency = np.random.uniform(0.12, 0.28)
            base_processing_time *= (1 - wms_efficiency)
            
        if agv_implemented:
            agv_efficiency = np.random.uniform(0.08, 0.20)
            base_processing_time *= (1 - agv_efficiency)
            
        if wms_implemented and agv_implemented:
            synergy_effect = np.random.uniform(0.03, 0.10)
            base_processing_time *= (1 - synergy_effect)
        
        # 계절별 효과
        seasonal_multiplier = {1: 1.05, 2: 0.95, 3: 1.15, 4: 1.25}[season]
        base_processing_time *= seasonal_multiplier
        
        # 요일별 효과
        if day_of_week in [6, 7]:
            weekend_multiplier = 0.8
        elif day_of_week == 1:
            weekend_multiplier = 1.1
        else:
            weekend_multiplier = 1.0
        base_processing_time *= weekend_multiplier
        
        # 최종 처리 시간
        processing_time = max(5, base_processing_time)
        
        # === 성과 지표 계산 ===
        # 피킹 정확도
        base_accuracy = 82 + np.random.normal(0, 4)
        if wms_implemented:
            base_accuracy += np.random.uniform(5, 12)
        if agv_implemented:
            base_accuracy += np.random.uniform(2, 6)
        if voice_picking:
            base_accuracy += np.random.uniform(1, 3)
        
        # 직원 경험 효과
        base_accuracy += (average_employee_experience / 60) * 2
        picking_accuracy = np.clip(base_accuracy, 65, 98)
        
        # 오류율
        error_rate = np.random.uniform(1.5, 8.0)
        if wms_implemented:
            error_rate *= np.random.uniform(0.6, 0.8)
        if agv_implemented:
            error_rate *= np.random.uniform(0.7, 0.9)
        if voice_picking:
            error_rate *= np.random.uniform(0.8, 0.9)
        
        # 품질 검사 강도에 따른 오류율 조정
        error_rate *= (1 - (quality_check_level - 1) * 0.1)
        error_rate = np.clip(error_rate, 0.5, 12)
        
        # 주문당 인건비
        daily_labor_cost = workers * np.random.uniform(100000, 140000)
        time_efficiency = processing_time / (daily_orders * 0.5)
        
        labor_cost_per_order = (
            daily_labor_cost / daily_orders * 
            time_efficiency * 
            np.random.uniform(0.9, 1.1)
        )
        
        # 임시직 비율에 따른 비용 조정 (임시직 시급이 더 높음)
        labor_cost_per_order *= (1 + temp_worker_ratio * 0.2)
        labor_cost_per_order = np.clip(labor_cost_per_order, 300, 3500)
        
        data.append({
            # 기본 변수
            'warehouse_size': warehouse_size,
            'daily_orders': daily_orders,
            'item_types': item_types,
            'workers': workers,
            'shift_type': shift_type,
            'wms_implemented': wms_implemented,
            'agv_implemented': agv_implemented,
            'season': season,
            'day_of_week': day_of_week,
            # 추가 변수들
            'temperature_control': temperature_control,
            'ceiling_height': ceiling_height,
            'loading_dock_count': loading_dock_count,
            'warehouse_age': warehouse_age,
            'average_item_weight': average_item_weight,
            'fragile_items_ratio': fragile_items_ratio,
            'inventory_turnover': inventory_turnover,
            'conveyor_belt_ratio': conveyor_belt_ratio,
            'voice_picking': voice_picking,
            'barcode_scanner_quality': barcode_scanner_quality,
            'average_employee_experience': average_employee_experience,
            'employee_turnover_rate': employee_turnover_rate,
            'temp_worker_ratio': temp_worker_ratio,
            'batch_picking_enabled': batch_picking_enabled,
            'quality_check_level': quality_check_level,
            'order_complexity_score': order_complexity_score,
            # 타겟 변수들
            'processing_time': processing_time,
            'picking_accuracy': picking_accuracy,
            'error_rate': error_rate,
            'labor_cost_per_order': labor_cost_per_order
        })
    
    return pd.DataFrame(data)

def create_comprehensive_visualizations():
    """종합적인 창고 운영 데이터 시각화"""
    
    print("🎨 창고 운영 데이터 종합 시각화 생성 중...")
    
    # 데이터 생성
    df = generate_realistic_warehouse_data(2000)
    
    # 시스템 타입 분류
    df['system_type'] = df.apply(lambda x: 
        'WMS+AGV' if x['wms_implemented']==1 and x['agv_implemented']==1 else
        'WMS만' if x['wms_implemented']==1 else
        'AGV만' if x['agv_implemented']==1 else '미도입', axis=1)
    
    # 전체 시각화 대시보드 생성
    fig = plt.figure(figsize=(24, 18))
    fig.suptitle('🏭 Warehouse Operational Optimization Comprehensive Analysis Dashboard', fontsize=20, fontweight='bold', y=0.98)
    
    # === 1. 기본 데이터 분포 ===
    # 1-1. 창고 규모 vs 일일 주문량
    plt.subplot(4, 6, 1)
    colors = {'미도입': '#ff9999', 'AGV만': '#66b3ff', 'WMS만': '#99ff99', 'WMS+AGV': '#ffcc99'}
    for system_type in df['system_type'].unique():
        mask = df['system_type'] == system_type
        plt.scatter(df.loc[mask, 'warehouse_size'], df.loc[mask, 'daily_orders'], 
                   c=colors[system_type], label=system_type, alpha=0.6, s=20)
    plt.xlabel('Warehouse Size (m²)')
    plt.ylabel('Daily Orders')
    plt.title('Warehouse Size vs Daily Orders', fontweight='bold')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # 1-2. 시스템 도입 현황 (파이 차트)
    plt.subplot(4, 6, 2)
    system_counts = df['system_type'].value_counts()
    colors_pie = [colors[label] for label in system_counts.index]
    plt.pie(system_counts.values, labels=system_counts.index, colors=colors_pie, 
            autopct='%1.1f%%', startangle=90)
    plt.title('System Implementation Status', fontweight='bold')
    
    # 1-3. Processing Time Distribution
    plt.subplot(4, 6, 3)
    plt.hist(df['processing_time'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(df['processing_time'].mean(), color='red', linestyle='--', 
                label=f'Average: {df["processing_time"].mean():.0f} minutes')
    plt.xlabel('Processing Time (minutes)')
    plt.ylabel('Frequency')
    plt.title('Processing Time Distribution', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 1-4. 피킹 정확도 분포
    plt.subplot(4, 6, 4)
    plt.hist(df['picking_accuracy'], bins=25, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.axvline(df['picking_accuracy'].mean(), color='red', linestyle='--',
                label=f'Average: {df["picking_accuracy"].mean():.1f}%')
    plt.xlabel('Picking Accuracy (%)')
    plt.ylabel('Frequency')
    plt.title('Picking Accuracy Distribution', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # === 2. 시스템 효과 분석 ===
    # 2-1. 시스템별 처리시간 박스플롯
    plt.subplot(4, 6, 5)
    box_data = [df[df['system_type']==st]['processing_time'] for st in ['미도입', 'AGV만', 'WMS만', 'WMS+AGV']]
    bp = plt.boxplot(box_data, labels=['미도입', 'AGV만', 'WMS만', 'WMS+AGV'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    plt.ylabel('Processing Time (minutes)')
    plt.title('System-wise Processing Time Comparison', fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 2-2. 시스템별 성과 지표
    plt.subplot(4, 6, 6)
    metrics_data = df.groupby('system_type').agg({
        'processing_time': 'mean',
        'picking_accuracy': 'mean',
        'error_rate': 'mean',
        'labor_cost_per_order': 'mean'
    })
    
    # 정규화 (0-1 스케일, 처리시간과 오류율은 역순)
    normalized_data = metrics_data.copy()
    normalized_data['processing_time'] = 1 - (normalized_data['processing_time'] - normalized_data['processing_time'].min()) / (normalized_data['processing_time'].max() - normalized_data['processing_time'].min())
    normalized_data['picking_accuracy'] = (normalized_data['picking_accuracy'] - normalized_data['picking_accuracy'].min()) / (normalized_data['picking_accuracy'].max() - normalized_data['picking_accuracy'].min())
    normalized_data['error_rate'] = 1 - (normalized_data['error_rate'] - normalized_data['error_rate'].min()) / (normalized_data['error_rate'].max() - normalized_data['error_rate'].min())
    normalized_data['labor_cost_per_order'] = 1 - (normalized_data['labor_cost_per_order'] - normalized_data['labor_cost_per_order'].min()) / (normalized_data['labor_cost_per_order'].max() - normalized_data['labor_cost_per_order'].min())
    
    x = range(len(normalized_data.columns))
    width = 0.2
    for i, (system, color) in enumerate(zip(normalized_data.index, ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])):
        plt.bar([p + width * i for p in x], normalized_data.loc[system], width, 
                label=system, color=color, alpha=0.8)
    
    plt.xlabel('KPI')
    plt.ylabel('Normalized Performance (Higher is Better)')
    plt.title('System-wise Comprehensive Performance Comparison', fontweight='bold')
    plt.xticks([p + width * 1.5 for p in x], ['Processing Time', 'Accuracy', 'Error Rate', 'Labor Cost per Order'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # === 3. 상관관계 및 패턴 분석 ===
    # 3-1. 주요 변수 상관관계 히트맵
    plt.subplot(4, 6, 7)
    corr_vars = ['daily_orders', 'workers', 'processing_time', 'picking_accuracy', 
                 'error_rate', 'labor_cost_per_order', 'wms_implemented', 'agv_implemented']
    corr_matrix = df[corr_vars].corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title('Correlation Analysis of Key Variables', fontweight='bold')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # 3-2. 작업자 경험도 vs 성과
    plt.subplot(4, 6, 8)
    plt.scatter(df['average_employee_experience'], df['picking_accuracy'], 
                c=df['processing_time'], cmap='viridis_r', alpha=0.6, s=20)
    plt.colorbar(label='Processing Time (minutes)')
    plt.xlabel('Average Employee Experience (months)')
    plt.ylabel('Picking Accuracy (%)')
    plt.title('Employee Experience vs Picking Accuracy', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 3-3. 창고 연식 vs 효율성
    plt.subplot(4, 6, 9)
    df['efficiency'] = df['daily_orders'] / df['processing_time']
    plt.scatter(df['warehouse_age'], df['efficiency'], 
                c=df['temperature_control'], cmap='coolwarm', alpha=0.6, s=20)
    plt.colorbar(label='Temperature Control (0:Room, 1:Refrigerated, 2:Frozen)')
    plt.xlabel('Warehouse Age (years)')
    plt.ylabel('Processing Efficiency (orders/minute)')
    plt.title('Warehouse Age vs Processing Efficiency', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # === 4. XGBoost 모델 분석 ===
    # 모델 훈련
    feature_columns = ['warehouse_size', 'daily_orders', 'item_types', 'workers', 
                      'shift_type', 'wms_implemented', 'agv_implemented', 
                      'season', 'day_of_week', 'temperature_control', 'ceiling_height', 
                      'loading_dock_count', 'warehouse_age', 'average_item_weight', 
                      'fragile_items_ratio', 'inventory_turnover', 'conveyor_belt_ratio', 
                      'voice_picking', 'barcode_scanner_quality', 'average_employee_experience', 
                      'employee_turnover_rate', 'temp_worker_ratio', 'batch_picking_enabled', 
                      'quality_check_level', 'order_complexity_score']
    
    X = df[feature_columns]
    y = df['processing_time']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, 
                        random_state=42, verbosity=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # 4-1. 특성 중요도
    plt.subplot(4, 6, 10)
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    top_features = feature_importance.tail(10)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title('Top 10 Feature Importance', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 4-2. 예측 vs 실제 성능
    plt.subplot(4, 6, 11)
    plt.scatter(y_test, y_pred, alpha=0.6, color='green', s=30)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Processing Time')
    plt.ylabel('Predicted Processing Time')
    plt.title(f'Prediction Performance (R² = {r2_score(y_test, y_pred):.3f})', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 4-3. 잔차 분포
    plt.subplot(4, 6, 12)
    residuals = y_test - y_pred
    plt.hist(residuals, bins=20, alpha=0.7, color='orange', edgecolor='black')
    plt.axvline(0, color='red', linestyle='--')
    plt.xlabel('Residual (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.title('Prediction Error Distribution', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # === 5. 계절성 및 요일 효과 ===
    # 5-1. 계절별 처리시간
    plt.subplot(4, 6, 13)
    season_names = {1: 'Spring', 2: 'Summer', 3: 'Autumn', 4: 'Winter'}
    df['season_name'] = df['season'].map(season_names)
    season_data = df.groupby(['season_name', 'system_type'])['processing_time'].mean().unstack()
    season_data.plot(kind='bar', ax=plt.gca(), color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
    plt.xlabel('Season')
    plt.ylabel('Average Processing Time (minutes)')
    plt.title('Season-wise System Effect', fontweight='bold')
    plt.legend(title='시스템 타입', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3)
    
    # 5-2. 요일별 주문량 패턴
    plt.subplot(4, 6, 14)
    day_names = {1: '월', 2: '화', 3: '수', 4: '목', 5: '금', 6: '토', 7: '일'}
    df['day_name'] = df['day_of_week'].map(day_names)
    day_orders = df.groupby('day_name')['daily_orders'].mean().reindex(['월', '화', '수', '목', '금', '토', '일'])
    
    colors_day = ['lightcoral' if day in ['토', '일'] else 'lightblue' for day in day_orders.index]
    bars = plt.bar(day_orders.index, day_orders.values, color=colors_day)
    plt.xlabel('Day')
    plt.ylabel('Average Orders')
    plt.title('Day-wise Order Pattern', fontweight='bold')
    
    # 막대 위에 값 표시
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                 f'{height:.0f}', ha='center', va='bottom')
    plt.grid(True, alpha=0.3)
    
    # === 6. 비즈니스 인사이트 ===
    # 6-1. ROI 분석 (투자회수기간)
    plt.subplot(4, 6, 15)
    
    # 시스템별 평균 처리시간으로 ROI 계산 (간단한 시뮬레이션)
    baseline_time = df[df['system_type'] == '미도입']['processing_time'].mean()
    monthly_orders = 15000  # 가정
    
    roi_data = []
    investments = {'WMS만': 150, 'AGV만': 80, 'WMS+AGV': 230}  # 백만원 단위
    
    for system in ['WMS만', 'AGV만', 'WMS+AGV']:
        if system == 'WMS만':
            system_time = df[df['wms_implemented']==1]['processing_time'].mean()
        elif system == 'AGV만':
            system_time = df[df['agv_implemented']==1]['processing_time'].mean()
        else:
            system_time = df[df['system_type']=='WMS+AGV']['processing_time'].mean()
        
        time_saved = baseline_time - system_time
        monthly_savings = time_saved * monthly_orders * 0.001  # 백만원 단위
        roi_months = investments[system] / monthly_savings if monthly_savings > 0 else float('inf')
        roi_data.append(roi_months)
    
    bars = plt.bar(['WMS만', 'AGV만', 'WMS+AGV'], roi_data, 
                   color=['#99ff99', '#66b3ff', '#ffcc99'])
    plt.xlabel('Implemented System')
    plt.ylabel('ROI Period (months)')
    plt.title('ROI Period by System', fontweight='bold')
    
    for bar, months in zip(bars, roi_data):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                 f'{months:.1f} months', ha='center', va='bottom', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 6-2. 기술 조합별 효과
    plt.subplot(4, 6, 16)
    
    # WMS + Voice Picking + Batch Picking 조합 효과
    tech_combinations = []
    for wms in [0, 1]:
        for voice in [0, 1]:
            for batch in [0, 1]:
                if voice == 1 and wms == 0:  # 음성피킹은 WMS가 있어야 가능
                    continue
                
                subset = df[
                    (df['wms_implemented'] == wms) & 
                    (df['voice_picking'] == voice) & 
                    (df['batch_picking_enabled'] == batch)
                ]
                
                if len(subset) > 10:
                    tech_combinations.append({
                        'combination': f"WMS:{wms}|Voice:{voice}|Batch:{batch}",
                        'avg_time': subset['processing_time'].mean(),
                        'count': len(subset)
                    })
    
    tech_df = pd.DataFrame(tech_combinations).sort_values('avg_time')
    
    bars = plt.barh(range(len(tech_df)), tech_df['avg_time'])
    plt.yticks(range(len(tech_df)), tech_df['combination'])
    plt.xlabel('Average Processing Time (minutes)')
    plt.title('Technical Combination Performance', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # === 6. 종합 성과 텍스트 요약 ===
    # 하단에 성과 요약 텍스트 박스 추가
    fig.text(0.02, 0.02, 
             f'📊 Key Insights:\n'
             f'• WMS+AGV 조합이 평균 {baseline_time - df[df["system_type"]=="WMS+AGV"]["processing_time"].mean():.1f} minutes 단축 효과\n'
             f'• 직원 경험이 1개월 증가할 때마다 정확도 {(df["picking_accuracy"].corr(df["average_employee_experience"]) * 100):.1f}% 상관관계\n'
             f'• 직원 경험이 1개월 증가할 때마다 정확도 {(df["picking_accuracy"].corr(df["average_employee_experience"]) * 100):.1f}% 상관관계\n'
             f'• 음성 피킹 도입시 평균 {df[df["voice_picking"]==1]["processing_time"].mean() - df[df["voice_picking"]==0]["processing_time"].mean():.1f} minutes 추가 단축\n'
             f'• 모델 예측 정확도: R² = {r2_score(y_test, y_pred):.3f}, RMSE = {np.sqrt(mean_squared_error(y_test, y_pred)):.1f} minutes',
             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    plt.savefig('warehouse_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # === 추가 특화 시각화 ===
    create_detailed_system_comparison(df)
    create_business_impact_analysis(df, model, feature_columns)
    
    print("✅ 모든 시각화 완료!")
    return df

def create_detailed_system_comparison(df):
    """시스템 도입 효과 상세 비교 차트"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('🔍 시스템 도입 효과 상세 분석', fontsize=16, fontweight='bold')
    
    # 1. 시스템별 성과 지표 레이더 차트 스타일
    ax = axes[0, 0]
    systems = ['미도입', 'AGV만', 'WMS만', 'WMS+AGV']
    metrics = ['processing_time', 'picking_accuracy', 'error_rate', 'labor_cost_per_order']
    
    # 각 시스템별 성과를 0-1로 정규화
    system_scores = []
    for system in systems:
        subset = df[df['system_type'] == system]
        scores = []
        # 처리시간: 낮을수록 좋음 (역순)
        scores.append(1 - (subset['processing_time'].mean() - df['processing_time'].min()) / 
                     (df['processing_time'].max() - df['processing_time'].min()))
        # 정확도: 높을수록 좋음
        scores.append((subset['picking_accuracy'].mean() - df['picking_accuracy'].min()) / 
                     (df['picking_accuracy'].max() - df['picking_accuracy'].min()))
        # 오류율: 낮을수록 좋음 (역순)
        scores.append(1 - (subset['error_rate'].mean() - df['error_rate'].min()) / 
                     (df['error_rate'].max() - df['error_rate'].min()))
        # 인건비: 낮을수록 좋음 (역순)
        scores.append(1 - (subset['labor_cost_per_order'].mean() - df['labor_cost_per_order'].min()) / 
                     (df['labor_cost_per_order'].max() - df['labor_cost_per_order'].min()))
        system_scores.append(scores)
    
    x = np.arange(len(metrics))
    width = 0.2
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    
    for i, (system, scores, color) in enumerate(zip(systems, system_scores, colors)):
        ax.bar(x + i * width, scores, width, label=system, color=color, alpha=0.8)
    
    ax.set_xlabel('KPI')
    ax.set_ylabel('Normalized Performance (1이 최고)')
    ax.set_title('System-wise Comprehensive Performance Score')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(['Processing Time', 'Accuracy', 'Error Rate', 'Labor Cost per Order'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 창고 규모별 시스템 효과
    ax = axes[0, 1]
    df['size_category'] = pd.cut(df['warehouse_size'], bins=3, labels=['소형', '중형', '대형'])
    size_effect = df.groupby(['size_category', 'system_type'])['processing_time'].mean().unstack()
    size_effect.plot(kind='bar', ax=ax, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
    ax.set_xlabel('Warehouse Size')
    ax.set_ylabel('Average Processing Time (minutes)')
    ax.set_title('Warehouse Size vs System Effect')
    ax.legend(title='System Type')
    ax.tick_params(axis='x', rotation=0)
    ax.grid(True, alpha=0.3)
    
    # 3. 온도 관리별 시스템 성능
    ax = axes[0, 2]
    temp_labels = {0: 'Room', 1: 'Refrigerated', 2: 'Frozen'}
    df['temp_name'] = df['temperature_control'].map(temp_labels)
    temp_performance = df.groupby(['temp_name', 'system_type'])['processing_time'].mean().unstack()
    temp_performance.plot(kind='bar', ax=ax, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
    ax.set_xlabel('Temperature Control Type')
    ax.set_ylabel('Average Processing Time (minutes)')
    ax.set_title('Temperature Control vs System Performance')
    ax.legend(title='System Type')
    ax.tick_params(axis='x', rotation=0)
    ax.grid(True, alpha=0.3)
    
    # 4. 직원 수 vs 시스템 효율성
    ax = axes[1, 0]
    df['worker_category'] = pd.cut(df['workers'], bins=3, labels=['Small', 'Medium', 'Large'])
    for system_type in df['system_type'].unique():
        subset = df[df['system_type'] == system_type]
        efficiency = subset['daily_orders'] / subset['processing_time']
        ax.scatter(subset['workers'], efficiency, label=system_type, 
                  alpha=0.6, s=30, c=colors[list(df['system_type'].unique()).index(system_type)])
    
    ax.set_xlabel('Worker Count')
    ax.set_ylabel('Processing Efficiency (orders/minute)')
    ax.set_title('Worker Size vs System Efficiency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. 시간대별/계절별 효과
    ax = axes[1, 1]
    seasonal_data = []
    seasons = [1, 2, 3, 4]
    season_names = ['Spring', 'Summer', 'Autumn', 'Winter']
    
    for system in systems:
        seasonal_performance = []
        for season in seasons:
            if system == '미도입':
                subset = df[(df['season'] == season) & (df['system_type'] == system)]
            else:
                subset = df[(df['season'] == season) & (df['system_type'] == system)]
            
            if len(subset) > 0:
                seasonal_performance.append(subset['processing_time'].mean())
            else:
                seasonal_performance.append(0)
        
        ax.plot(season_names, seasonal_performance, marker='o', 
               label=system, linewidth=2, markersize=6)
    
    ax.set_xlabel('Season')
    ax.set_ylabel('Average Processing Time (minutes)')
    ax.set_title('Season-wise System Performance Change')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. 투자 대비 효과 (Bubble Chart)
    ax = axes[1, 2]
    
    # 각 시스템의 투자비용, 절감효과, 도입률을 시각화
    baseline_cost = df[df['system_type'] == '미도입']['labor_cost_per_order'].mean()
    
    investment_data = {
        'AGV만': {'investment': 80, 'savings': baseline_cost - df[df['system_type'] == 'AGV만']['labor_cost_per_order'].mean(), 
                 'adoption': (df['system_type'] == 'AGV만').mean()},
        'WMS만': {'investment': 150, 'savings': baseline_cost - df[df['system_type'] == 'WMS만']['labor_cost_per_order'].mean(),
                 'adoption': (df['system_type'] == 'WMS만').mean()},
        'WMS+AGV': {'investment': 230, 'savings': baseline_cost - df[df['system_type'] == 'WMS+AGV']['labor_cost_per_order'].mean(),
                   'adoption': (df['system_type'] == 'WMS+AGV').mean()}
    }
    
    for i, (system, data) in enumerate(investment_data.items()):
        x = data['investment']
        y = data['savings']
        size = data['adoption'] * 1000  # 도입률을 원 크기로
        color = ['#66b3ff', '#99ff99', '#ffcc99'][i]
        
        ax.scatter(x, y, s=size, c=color, alpha=0.7, 
                  edgecolors='black', linewidth=1, label=system)
        
        # 라벨 추가
        ax.annotate(system, (x, y), xytext=(5, 5), 
                   textcoords='offset points', fontweight='bold')
    
    ax.set_xlabel('Initial Investment Cost (million won)')
    ax.set_ylabel('Monthly Cost Savings (won)')
    ax.set_title('Investment vs Effect Analysis\n(Circle Size = Adoption Rate)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('detailed_system_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_business_impact_analysis(df, model, feature_columns):
    """비즈니스 임팩트 분석 차트"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('💼 Business Impact Analysis', fontsize=16, fontweight='bold')
    
    # 1. 월별 비용 절감 시뮬레이션
    ax = axes[0, 0]
    
    monthly_orders = 15000
    baseline_cost_per_order = df[df['system_type'] == '미도입']['labor_cost_per_order'].mean()
    
    systems = ['WMS만', 'AGV만', 'WMS+AGV']
    monthly_savings = []
    
    for system in systems:
        system_cost_per_order = df[df['system_type'] == system]['labor_cost_per_order'].mean()
        monthly_saving = (baseline_cost_per_order - system_cost_per_order) * monthly_orders
        monthly_savings.append(monthly_saving / 1000000)  # 백만원 단위
    
    bars = ax.bar(systems, monthly_savings, color=['#99ff99', '#66b3ff', '#ffcc99'])
    ax.set_xlabel('Implemented System')
    ax.set_ylabel('Monthly Cost Savings (million won)')
    ax.set_title('System-wise Monthly Cost Savings Effect')
    
    # 막대 위에 값 표시
    for bar, saving in zip(bars, monthly_savings):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{saving:.1f}M', ha='center', va='bottom', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 2. 누적 ROI 시뮬레이션
    ax = axes[0, 1]
    
    # 3년간 누적 ROI 계산
    months = range(1, 37)  # 36개월
    investments = {'WMS만': 150, 'AGV만': 80, 'WMS+AGV': 230}  # 백만원
    
    for i, system in enumerate(systems):
        investment = investments[system]
        monthly_saving = monthly_savings[i]
        cumulative_roi = [(monthly_saving * month - investment) for month in months]
        
        ax.plot(months, cumulative_roi, marker='o', label=system, linewidth=2,
               color=['#99ff99', '#66b3ff', '#ffcc99'][i])
    
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('개월')
    ax.set_ylabel('Cumulative Profit (million won)')
    ax.set_title('3-Year Cumulative ROI Simulation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 시나리오 민감도 분석
    ax = axes[1, 0]
    
    # 주문량 변화에 따른 효과
    order_scenarios = [10000, 15000, 20000, 25000, 30000]
    wms_agv_benefits = []
    
    for orders in order_scenarios:
        baseline_monthly = baseline_cost_per_order * orders
        wms_agv_monthly = df[df['system_type'] == 'WMS+AGV']['labor_cost_per_order'].mean() * orders
        benefit = (baseline_monthly - wms_agv_monthly) / 1000000  # 백만원
        wms_agv_benefits.append(benefit)
    
    ax.plot(order_scenarios, wms_agv_benefits, marker='o', linewidth=3, 
           color='#ffcc99', markersize=8)
    ax.set_xlabel('Monthly Orders')
    ax.set_ylabel('WMS+AGV Monthly Cost Savings (million won)')
    ax.set_title('Order Volume Effect on Cost Savings')
    ax.grid(True, alpha=0.3)
    
    # 4. 기술별 도입 우선순위 매트릭스
    ax = axes[1, 1]
    
    # 각 기술의 도입 용이성 vs 효과 매트릭스
    technologies = {
        'WMS': {'effect': 0.8, 'ease': 0.6, 'cost': 150},
        'AGV': {'effect': 0.5, 'ease': 0.4, 'cost': 80},
        'Voice Picking': {'effect': 0.3, 'ease': 0.8, 'cost': 20},
        'Batch Picking': {'effect': 0.4, 'ease': 0.9, 'cost': 10},
        'WMS+AGV': {'effect': 1.0, 'ease': 0.3, 'cost': 230}
    }
    
    for tech, data in technologies.items():
        x = data['ease']
        y = data['effect'] 
        size = (300 - data['cost']) / 2  # 비용이 낮을수록 원이 커짐
        
        ax.scatter(x, y, s=size, alpha=0.7, edgecolors='black', linewidth=1)
        ax.annotate(tech, (x, y), xytext=(5, 5), 
                   textcoords='offset points', fontweight='bold')
    
    ax.set_xlabel('Ease of Implementation')
    ax.set_ylabel('Effect Level') 
    ax.set_title('Technology Implementation Priority Matrix\n(Circle Size = Cost-Effectiveness)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    
    # 사분면 가이드라인 추가
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.text(0.75, 0.75, '우선순위\n높음', ha='center', va='center', 
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    ax.text(0.25, 0.25, '우선순위\n낮음', ha='center', va='center',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('business_impact_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("🚀 창고 운영 최적화 시각화 분석 시작!")
    df = create_comprehensive_visualizations()
    
    print(f"\n📈 생성된 시각화 파일:")
    print("1. warehouse_comprehensive_analysis.png - 종합 대시보드")
    print("2. detailed_system_comparison.png - 시스템 비교 상세 분석") 
    print("3. business_impact_analysis.png - 비즈니스 임팩트 분석")
    
    print(f"\n🎯 핵심 인사이트:")
    baseline = df[df['system_type'] == '미도입']['processing_time'].mean()
    wms_agv = df[df['system_type'] == 'WMS+AGV']['processing_time'].mean()
    improvement = (baseline - wms_agv) / baseline * 100
    
    print(f"• WMS+AGV 조합으로 {improvement:.1f}% 처리시간 단축 가능")
    print(f"• 음성 피킹 도입률: {df['voice_picking'].mean():.1%}")
    print(f"• 배치 피킹 도입률: {df['batch_picking_enabled'].mean():.1%}")
    print(f"• 평균 직원 경험: {df['average_employee_experience'].mean():.1f}개월")
    
    print(f"\n✅ 시각화 분석 완료!")
