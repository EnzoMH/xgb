import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# data.py에서 데이터 생성 함수들 import
from data import generate_warehouse_raw_data, get_feature_target_columns, save_warehouse_data

# 시드 설정 (재현성을 위해)  
np.random.seed(42)

# =============================================================================
# 메인 실행부: XGBoost MultiOutput 모델 훈련 및 분석
# =============================================================================

# 데이터 생성 (data.py에서 함수 호출)
print("🏭 data.py에서 창고 데이터 생성 중...")
df = generate_warehouse_raw_data(2000)

# 데이터 확인
print("🏭 warehouse_raw_data.csv 형태의 가상 데이터 생성 완료!")
print("=" * 60)
print(f"📊 데이터 크기: {df.shape}")

# 주요 통계
print(f"\n=== 기본 통계 ===")
print(f"창고 면적 평균: {df['warehouse_area_m2'].mean():.0f}m² (범위: {df['warehouse_area_m2'].min():.0f}-{df['warehouse_area_m2'].max():.0f})")
print(f"일일 처리량 평균: {df['daily_throughput'].mean():.0f}건 (범위: {df['daily_throughput'].min()}-{df['daily_throughput'].max()})")
print(f"총 장비 수 평균: {df['total_equipment_count'].mean():.0f}대 (범위: {df['total_equipment_count'].min()}-{df['total_equipment_count'].max()})")
print(f"작업자 수 평균: {df['workers'].mean():.0f}명 (범위: {df['workers'].min()}-{df['workers'].max()})")
print(f"자동화 수준 평균: {df['automation_level'].mean():.1f} (범위: {df['automation_level'].min()}-{df['automation_level'].max()})")

print(f"\n=== Y-Label 분포 ===")
print(f"처리시간: {df['processing_time_seconds'].min():.1f}~{df['processing_time_seconds'].max():.1f}초")
print(f"피킹 정확도: {df['picking_accuracy_percent'].min():.1f}~{df['picking_accuracy_percent'].max():.1f}%")
print(f"오류율: {df['error_rate_percent'].min():.1f}~{df['error_rate_percent'].max():.1f}%")
print(f"주문당 비용: {df['labor_cost_per_order_krw'].min():,}~{df['labor_cost_per_order_krw'].max():,}원")

print(f"\n=== 기술 도입 현황 ===")
print(f"WMS 도입률: {df['wms_implemented'].mean():.1%}")
print(f"AGV 도입률: {df['agv_implemented'].mean():.1%}")
print(f"WMS+AGV 동시 도입률: {((df['wms_implemented']==1) & (df['agv_implemented']==1)).mean():.1%}")

print(f"\n=== 장비 구성 ===")
print(f"컨베이어 평균: {df['conveyor_count'].mean():.1f}대")
print(f"RTV/AGV 평균: {df['rtv_agv_count'].mean():.1f}대") 
print(f"SRM 평균: {df['srm_count'].mean():.1f}대")
print(f"로봇암 평균: {df['robot_arm_count'].mean():.1f}대")
print(f"랙 수 평균: {df['rack_count'].mean():.1f}개")

# XGBoost MultiOutput 모델 훈련
print("\n🤖 XGBoost MultiOutput 회귀 모델 훈련")
print("=" * 50)

# X-Label과 Y-Label 컬럼 정보 가져오기 (data.py에서)
feature_columns, target_columns = get_feature_target_columns()

X = df[feature_columns]
y = df[target_columns]

print(f"📊 피처 수: {len(feature_columns)}개")
print(f"🎯 타겟 수: {len(target_columns)}개")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MultiOutput XGBoost 모델
try:
    from sklearn.multioutput import MultiOutputRegressor
    
    model = MultiOutputRegressor(
        XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbosity=0
        )
    )
    
    print("🔄 모델 훈련 중...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("\n📊 모델 성능 평가:")
    target_names = ['처리시간(초)', '정확도(%)', '오류율(%)', '비용(원)']
    
    for i, target_name in enumerate(target_names):
        mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
        r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
        print(f"   • {target_name:12s}: MAE={mae:6.2f}, RMSE={rmse:6.2f}, R²={r2:.3f}")

    print("\n✅ MultiOutput 회귀 훈련 완료!")
    
except ImportError:
    print("❌ scikit-learn의 MultiOutputRegressor를 사용할 수 없습니다.")
    print("   대신 단일 타겟 모델로 훈련합니다.")
    
    # 단일 타겟 모델 (processing_time_seconds)
    model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbosity=0
    )
    
    y_single = df['processing_time_seconds']
    X_train, X_test, y_train, y_test = train_test_split(X, y_single, test_size=0.2, random_state=42)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    print(f"R²: {r2_score(y_test, y_pred):.3f}")

# 특성 중요도 분석 (MultiOutput의 경우 첫 번째 타겟 기준)
try:
    if hasattr(model, 'feature_importances_'):
        # 단일 모델인 경우
        importances = model.feature_importances_
    else:
        # MultiOutput 모델인 경우 첫 번째 estimator 사용
        importances = model.estimators_[0].feature_importances_
    
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print(f"\n📈 특성 중요도 Top 10:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']:25s}: {row['importance']:.3f}")
        
except Exception as e:
    print(f"특성 중요도 분석 중 오류: {e}")

# 기술 도입 효과 분석
print("\n💡 WMS/AGV 도입 효과 분석:")
print("=" * 40)

# WMS 도입 효과
wms_effect = df.groupby('wms_implemented')[['processing_time_seconds', 'picking_accuracy_percent', 'error_rate_percent']].mean()
print("WMS 도입 효과:")
print(wms_effect.round(2))

print()

# AGV 도입 효과  
agv_effect = df.groupby('agv_implemented')[['processing_time_seconds', 'picking_accuracy_percent', 'error_rate_percent']].mean()
print("AGV 도입 효과:")
print(agv_effect.round(2))

print()

# WMS+AGV 동시 도입 효과
df['tech_combo'] = df['wms_implemented'].astype(str) + '_' + df['agv_implemented'].astype(str)
combo_labels = {'0_0': 'None', '1_0': 'WMS만', '0_1': 'AGV만', '1_1': 'WMS+AGV'}
df['tech_combo'] = df['tech_combo'].map(combo_labels)

combo_effect = df.groupby('tech_combo')[['processing_time_seconds', 'picking_accuracy_percent', 'error_rate_percent']].mean()
print("기술 조합별 효과:")
print(combo_effect.round(2))

# 데이터 저장 (data.py 함수 사용)
output_file = 'warehouse_synthetic_data.csv'
df_save = df.drop('tech_combo', axis=1)  # 임시 컬럼 제거
save_warehouse_data(df_save, output_file)
print(f"\n💾 가상 창고 데이터 저장 완료: {output_file}")
print(f"   📊 크기: {df_save.shape}")
print(f"   🎯 X-Label: {len(feature_columns)}개")
print(f"   🎯 Y-Label: {len(target_columns)}개")

print(f"\n✅ 데이터 품질 체크:")
print(f"   • 결측값: {df_save.isnull().sum().sum()}개")
print(f"   • 데이터 타입:")
for dtype, count in df_save.dtypes.value_counts().items():
    print(f"     - {dtype}: {count}개 컬럼")

print(f"\n🎯 사용법:")
print(f"   import pandas as pd")
print(f"   df = pd.read_csv('{output_file}')")
print(f"   # 바로 XGBoost MultiOutput 훈련 가능!")