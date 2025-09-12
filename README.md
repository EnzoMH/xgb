# 🏭 Digital Twin Warehouse Optimization with XGBoost

A comprehensive machine learning system for warehouse operation optimization using XGBoost MultiOutput regression and digital twin simulation data.

## 🎯 Overview

This project provides an end-to-end ML pipeline for warehouse performance optimization, combining:
- **Digital twin simulation data generation**
- **XGBoost MultiOutput regression modeling**
- **Warehouse automation impact analysis**
- **Real-time performance prediction**

## 📊 Key Features

### 🔧 Data Generation Engine (`data.py`)
- Generates synthetic warehouse operation data (31 columns)
- **27 X-Features**: Equipment counts, utilization rates, automation levels
- **4 Y-Targets**: Processing time, picking accuracy, error rate, labor cost
- Supports 1,000 to 100,000+ samples

### 🤖 ML Model Training (`xgb.py`)
- **MultiOutput XGBoost Regression** for simultaneous prediction
- Feature importance analysis
- WMS/AGV adoption impact assessment
- Comprehensive performance evaluation

### 📈 Performance Metrics
- **Processing Time**: 5.0~30.0 seconds
- **Picking Accuracy**: 75.0~98.0%
- **Error Rate**: 0.5~5.0%
- **Labor Cost**: 35,000~85,000 KRW per order

## 🚀 Quick Start

### 1. Generate Training Data
```python
# Generate 2000 warehouse operation samples
from data import generate_and_save_data

df = generate_and_save_data(
    n_samples=2000,
    filename='warehouse_data.csv'
)
```

### 2. Train XGBoost Model
```bash
# Run complete ML pipeline
python xgb.py
```

### 3. Use Data Module Independently
```bash
# Generate demo data only
python data.py
```

## 📁 Project Structure

```
📦 xgb/
├── 📄 data.py                    # Data generation engine
├── 🤖 xgb.py                     # XGBoost training & analysis
├── 🔧 digital_twin_roi_model.py  # ROI calculation utilities
├── 📊 warehouse_visualization.py # Data visualization tools
├── 🎯 newxgb_data.py             # Advanced scenario generator
├── 📋 forecaster.py              # Prediction utilities
├── 📝 template.py                # Project templates
├── 📈 Various Analysis PNGs      # Generated visualizations
└── 📊 Sample CSV Files           # Generated datasets
```

## 🛠️ Technology Stack

- **Python 3.12+**
- **XGBoost**: MultiOutput regression
- **scikit-learn**: ML utilities and evaluation
- **pandas/numpy**: Data manipulation
- **matplotlib/seaborn**: Visualization

## 📊 Model Performance

### XGBoost MultiOutput Results:
- **Processing Time**: R² = 0.911 (Excellent)
- **Picking Accuracy**: R² = 0.964 (Near Perfect)
- **Error Rate**: R² = 0.999 (Perfect)
- **Labor Cost**: R² = 0.249 (Room for improvement)

### Feature Importance:
1. **Season** (44.8%) - Seasonal demand variations
2. **Equipment Age** (41.5%) - Equipment condition impact
3. **Automation Level** (1.6%) - Technology adoption level
4. **Robot Arm Count** (0.9%) - Automation equipment

## 🏭 Use Cases

### 1. Warehouse Design Optimization
- Predict performance before construction
- Optimize equipment allocation
- Compare automation scenarios

### 2. Technology ROI Analysis
- WMS/AGV adoption impact assessment
- Cost-benefit analysis
- Investment prioritization

### 3. Operational Performance Prediction
- Real-time efficiency forecasting
- Bottleneck identification
- Resource optimization

## 📈 Business Impact Analysis

### WMS/AGV Adoption Effects:
- **WMS Only**: 1.63% error rate reduction
- **AGV Only**: 1.78% error rate reduction  
- **WMS + AGV**: 1.16% error rate (Best performance)
- **No Automation**: 2.41% error rate

## 🔧 Installation & Setup

```bash
# Clone repository
git clone https://github.com/EnzoMH/xgb.git
cd xgb

# Install dependencies
pip install pandas numpy scikit-learn xgboost matplotlib seaborn

# Run quick demo
python data.py
```

## 📝 Usage Examples

### Generate Custom Dataset
```python
from data import generate_warehouse_raw_data, save_warehouse_data

# Create 5000 samples
df = generate_warehouse_raw_data(5000)
save_warehouse_data(df, 'my_warehouse_data.csv')
```

### Load and Analyze
```python
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

# Load data
df = pd.read_csv('warehouse_synthetic_data.csv')

# Separate features and targets
feature_cols = [col for col in df.columns if col not in [
    'processing_time_seconds', 'picking_accuracy_percent', 
    'error_rate_percent', 'labor_cost_per_order_krw'
]]
target_cols = [
    'processing_time_seconds', 'picking_accuracy_percent',
    'error_rate_percent', 'labor_cost_per_order_krw'
]

X = df[feature_cols]
y = df[target_cols]

# Train model
model = MultiOutputRegressor(XGBRegressor())
model.fit(X, y)
```

## 🎯 Future Enhancements

- [ ] Real-time data integration
- [ ] Deep learning model comparison
- [ ] Cloud deployment pipeline
- [ ] Interactive dashboard
- [ ] A/B testing framework

## 📊 Sample Data

The repository includes several sample datasets:
- `warehouse_demo_data.csv` (1,000 samples)
- `warehouse_synthetic_data.csv` (2,000 samples)
- `warehouse_raw_data.csv` (1,000 samples)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 👨‍💻 Author

**EnzoMH** - [GitHub Profile](https://github.com/EnzoMH)

---

🏭 **Optimize your warehouse operations with AI-powered insights!** 🚀
