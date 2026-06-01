# ============================================
#   STUDENT MARKS PREDICTOR
#   ML Algorithm: Linear Regression
#   Input: Study Hours | Output: Predicted Marks
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ─────────────────────────────────────────
# STEP 1: DATA (Study Hours vs Marks)
# ─────────────────────────────────────────

study_hours = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5,
                         6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 4]).reshape(-1, 1)

marks       = np.array([22, 28, 33, 38, 42, 48, 52, 55, 60, 63,
                         68, 72, 74, 79, 83, 85, 88, 91, 95, 49])

# ─────────────────────────────────────────
# STEP 2: DATA CLEANING
# ─────────────────────────────────────────

# Clip marks to valid range 0–100
marks = np.clip(marks, 0, 100)

# Check for missing/NaN values
if np.isnan(study_hours).any() or np.isnan(marks).any():
    print("Warning: Missing values found! Removing them.")
    mask = ~np.isnan(marks)
    study_hours = study_hours[mask]
    marks = marks[mask]

print("✅ Data Cleaning Done")
print(f"   Total samples: {len(marks)}")
print(f"   Hours range  : {study_hours.min()} – {study_hours.max()}")
print(f"   Marks range  : {marks.min()} – {marks.max()}")
print()

# ─────────────────────────────────────────
# STEP 3: TRAIN / TEST SPLIT (80% / 20%)
# ─────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    study_hours, marks, test_size=0.2, random_state=42
)

print(f"📦 Train size: {len(X_train)} samples")
print(f"🧪 Test size : {len(X_test)} samples")
print()

# ─────────────────────────────────────────
# STEP 4: TRAINING THE MODEL
# ─────────────────────────────────────────

model = LinearRegression()
model.fit(X_train, y_train)

slope     = model.coef_[0]
intercept = model.intercept_

print("🤖 Model Trained!")
print(f"   Slope (m)     : {slope:.4f}")
print(f"   Intercept (b) : {intercept:.4f}")
print(f"   Equation      : marks = {slope:.2f} × hours + {intercept:.2f}")
print()

# ─────────────────────────────────────────
# STEP 5: EVALUATION
# ─────────────────────────────────────────

y_pred = model.predict(X_test)
r2  = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("📊 Model Evaluation:")
print(f"   R² Score (Accuracy) : {r2 * 100:.2f}%")
print(f"   Mean Absolute Error : {mae:.2f} marks")
print()

# ─────────────────────────────────────────
# STEP 6: PREDICTION
# ─────────────────────────────────────────

def predict_marks(hours):
    hours = np.array([[hours]])
    predicted = model.predict(hours)[0]
    predicted = np.clip(predicted, 0, 100)
    return round(predicted, 2)

def get_grade(score):
    if score >= 90: return "O  (Outstanding)"
    if score >= 75: return "A+ (Excellent)"
    if score >= 60: return "A  (Very Good)"
    if score >= 50: return "B  (Good)"
    if score >= 40: return "C  (Average)"
    return "F  (Fail)"

print("🎯 Predictions:")
for h in [2, 4, 6, 8, 10]:
    score = predict_marks(h)
    grade = get_grade(score)
    print(f"   {h} hrs/day → {score} marks → Grade: {grade}")
print()

# Custom input
user_hours = float(input("Enter your daily study hours (1–10): "))
predicted_score = predict_marks(user_hours)
print(f"\n📌 Predicted Marks : {predicted_score} / 100")
print(f"📌 Grade           : {get_grade(predicted_score)}")

# ─────────────────────────────────────────
# STEP 7: VISUALIZATION
# ─────────────────────────────────────────

all_hours_line = np.linspace(1, 10, 100).reshape(-1, 1)
all_preds_line = model.predict(all_hours_line)

plt.figure(figsize=(9, 5))

# Scatter: training data
plt.scatter(X_train, y_train, color='steelblue', label='Training data', zorder=3, s=60)

# Scatter: test data
plt.scatter(X_test, y_test, color='orange', label='Test data', zorder=3, s=60, marker='D')

# Regression line
plt.plot(all_hours_line, all_preds_line, color='green', linewidth=2, label='Regression line')

# User prediction point
plt.scatter([user_hours], [predicted_score], color='red', s=120,
            zorder=5, label=f'Your prediction ({predicted_score})', marker='*')

plt.title('Student Marks Predictor — Linear Regression', fontsize=14)
plt.xlabel('Study Hours per Day')
plt.ylabel('Marks (out of 100)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('marks_predictor_plot.png', dpi=150)
plt.show()
print("\n✅ Graph saved as marks_predictor_plot.png")
