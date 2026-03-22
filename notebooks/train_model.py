import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# 1. Load & Clean Data [cite: 19]
df = pd.read_csv('weatherHistory.csv')

# จัดการ Outliers: ค่า Pressure เป็น 0 ให้แทนด้วย Median [cite: 21]
median_pressure = df.loc[df['Pressure (millibars)'] > 0, 'Pressure (millibars)'].median()
df['Pressure (millibars)'] = df['Pressure (millibars)'].replace(0, median_pressure)

# เลือก Features ที่เหมาะสม [cite: 15]
features = ['Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Visibility (km)', 'Pressure (millibars)']
X = df[features]
y = df['Apparent Temperature (C)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Build Pipeline [cite: 21, 23]
# รวมการ Scaling และ Model เข้าด้วยกันเพื่อป้องกัน Data Leakage
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(random_state=42))
])

# 3. Hyperparameter Tuning (GridSearchCV) [cite: 27, 29]
param_grid = {
    'regressor__n_estimators': [50, 100],
    'regressor__max_depth': [None, 10]
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# 4. Evaluation [cite: 25, 28]
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"MAE: {mean_absolute_error(y_test, predictions):.4f}")
print(f"R2 Score: {r2_score(y_test, predictions):.4f}")

# 5. Save Model สำหรับ Deployment [cite: 7, 35]
joblib.dump(best_model, 'weather_model.pkl')

print("Done")