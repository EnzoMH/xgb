"""
WMS & AGV 도입 효과 분석 프롬프트 템플릿
"""

# 1. 데이터 생성 프롬프트
data_generation_prompt = """창고관리시스템(WMS)과 무인운반차(AGV) 도입 전후의 성과를 비교분석할 수 있는 가상 데이터셋을 생성해주세요.

**데이터 요구사항:**
- 샘플 수: 1,000~2,000개
- 독립변수: 창고규모, 일일주문량, 아이템종류, 작업자수, WMS도입여부, AGV도입여부, 계절성, 요일효과
- 종속변수: 처리시간, 피킹정확도, 오류율, 인건비
- WMS 효과: 처리시간 15-35% 단축, 정확도 10% 향상
- AGV 효과: 처리시간 10-25% 단축, 인건비 10% 절감
- 시너지 효과: WMS+AGV 동시 도입시 추가 5-15% 개선

**출력 형식:** XGBoost Regressor 훈련 가능한 CSV 파일"""

# 2. 모델 성능 분석 프롬프트
model_analysis_prompt = """생성된 창고 운영 데이터를 사용해서 XGBoost 회귀모델을 훈련하고 다음 분석을 수행해주세요:

**분석 요청사항:**
1. 처리시간 예측 모델 구축 (MAE, MSE, R² 평가)
2. 특성 중요도 분석 (WMS, AGV 영향력 확인)
3. WMS/AGV 도입 조합별 성과 비교
4. 비용효과 분석 (ROI 계산)
5. 시각화 (상관관계, 특성중요도, 그룹별 성과)

**비즈니스 질문:**
- WMS와 AGV 중 어느 것이 더 효과적인가?
- 동시 도입시 시너지 효과는 얼마나 되는가?
- 투자 우선순위는 어떻게 정해야 하는가?"""

# 3. 시나리오 시뮬레이션 프롬프트
scenario_simulation_prompt = """훈련된 XGBoost 모델을 사용해서 다음 시나리오들의 예상 성과를 예측해주세요:

**시나리오 설정:**
- 기준: 창고면적 10,000m², 일일주문 300건, 아이템 1,000종, 작업자 30명
- 시나리오 1: 현재 상태 (WMS X, AGV X)
- 시나리오 2: WMS만 도입
- 시나리오 3: AGV만 도입  
- 시나리오 4: WMS + AGV 동시 도입

**분석 요청:**
- 각 시나리오별 처리시간, 정확도, 비용 예측
- 투자비용 대비 절감효과 (3년 ROI)
- 도입 순서별 효과 비교
- 민감도 분석 (주문량 변화에 따른 효과)"""

# 4. 실무 적용 프롬프트
business_application_prompt = """WMS/AGV 도입을 검토중인 물류회사를 위한 의사결정 지원 보고서를 작성해주세요.

**회사 정보:**
- 창고 규모: [입력]
- 현재 일일 처리량: [입력]  
- 작업자 수: [입력]
- 예산 제약: [입력]

**보고서 구성:**
1. 현황 분석 (현재 성과 수준 진단)
2. 도입 효과 예측 (데이터 기반 정량 분석)
3. 비용편익 분석 (투자비용 vs 절감액)
4. 리스크 요인 (도입시 고려사항)
5. 단계적 도입 로드맵 (우선순위 및 일정)
6. 핵심 KPI 및 성과측정 방법

**의사결정 기준:**
- ROI 3년 이내 회수
- 처리시간 20% 이상 단축
- 정확도 95% 이상 달성"""

# 5. A/B 테스트 설계 프롬프트
ab_test_design_prompt = """WMS 도입 효과를 검증하기 위한 A/B 테스트를 설계해주세요.

**테스트 설계 요구사항:**
- 대조군: WMS 미도입 창고구역
- 실험군: WMS 도입 창고구역
- 측정 지표: 처리시간, 정확도, 오류율, 만족도
- 테스트 기간: 3개월
- 표본 크기: 통계적 유의성 확보

**분석 방법:**
- 사전/사후 비교 분석
- 실험군/대조군 차이 검정
- 혼재변수 통제 방법
- 결과 해석 및 일반화 가능성

**산출물:** 테스트 계획서, 데이터 수집 템플릿, 분석 코드"""

# 6. 고급 분석 프롬프트
advanced_analysis_prompt = """머신러닝을 활용한 고도화된 WMS/AGV 효과 분석을 수행해주세요.

**고급 분석 기법:**
1. **앙상블 모델링**: XGBoost, Random Forest, LightGBM 비교
2. **하이퍼파라미터 튜닝**: GridSearch/RandomSearch
3. **특성 엔지니어링**: 상호작용 변수, 다항식 특성
4. **시계열 분석**: 도입 후 성과 변화 추이
5. **클러스터링**: 유사한 창고 그룹 분석
6. **인과추론**: Causal Impact Analysis

**비즈니스 인사이트:**
- 최적 도입 조건 (창고 규모, 주문 패턴별)
- 효과 지속성 분석 (시간 경과에 따른 변화)
- 업계 벤치마크 대비 위치
- 추가 개선 기회 식별"""

# 프롬프트 사용 방법 및 설명
usage_guide = """
이 프롬프트 템플릿들을 사용하시면 WMS/AGV 도입 분석의 다양한 측면을 체계적으로 다룰 수 있습니다. 

**사용 방법:**
1. 목적에 맞는 프롬프트 변수를 선택
2. 필요시 [입력] 부분을 실제 값으로 수정
3. AI에게 해당 프롬프트 전달

**변수 목록:**
- data_generation_prompt: 데이터 생성
- model_analysis_prompt: 모델 분석
- scenario_simulation_prompt: 시나리오 시뮬레이션
- business_application_prompt: 실무 적용
- ab_test_design_prompt: A/B 테스트
- advanced_analysis_prompt: 고급 분석

실제 데이터가 있으시면 가상 데이터 대신 활용하실 수 있고, 분석 목적에 따라 적절한 프롬프트를 선택해서 사용하시면 됩니다.
"""

def print_prompt(prompt_name):
    """선택한 프롬프트를 출력하는 함수"""
    prompts = {
        'data_generation': data_generation_prompt,
        'model_analysis': model_analysis_prompt,
        'scenario_simulation': scenario_simulation_prompt,
        'business_application': business_application_prompt,
        'ab_test_design': ab_test_design_prompt,
        'advanced_analysis': advanced_analysis_prompt,
        'usage_guide': usage_guide
    }
    
    if prompt_name in prompts:
        print(f"=== {prompt_name.upper()} ===")
        print(prompts[prompt_name])
    else:
        print(f"사용 가능한 프롬프트: {list(prompts.keys())}")

def get_all_prompts():
    """모든 프롬프트를 딕셔너리로 반환"""
    return {
        'data_generation': data_generation_prompt,
        'model_analysis': model_analysis_prompt,
        'scenario_simulation': scenario_simulation_prompt,
        'business_application': business_application_prompt,
        'ab_test_design': ab_test_design_prompt,
        'advanced_analysis': advanced_analysis_prompt,
        'usage_guide': usage_guide
    }

if __name__ == "__main__":
    print("=== WMS/AGV 분석 프롬프트 템플릿 ===")
    print("\n사용 예시:")
    print("print_prompt('data_generation')  # 데이터 생성 프롬프트 출력")
    print("print_prompt('model_analysis')   # 모델 분석 프롬프트 출력")
    print("\n모든 프롬프트 목록:")
    for name in get_all_prompts().keys():
        print(f"- {name}")