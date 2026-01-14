# ===============================
# TASK 5: SALES PREDICTION USING PYTHON
# ===============================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv(r"C:\Users\anisc\Downloads\TASK 5\Advertising.csv")

print("\n--- Dataset Preview ---")
print(df.head())

print("\n--- Dataset Information ---")
print(df.info())

# -------------------------------
# Exploratory Data Analysis (EDA)
# -------------------------------
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

sns.pairplot(df)
plt.show()

# -------------------------------
# Feature Selection
# -------------------------------
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Model Training
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# Predictions
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# Model Evaluation
# -------------------------------
print("\n--- Model Evaluation ---")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

# -------------------------------
# Cross Validation
# -------------------------------
cv_scores = cross_val_score(model, X, y, cv=5)
print("\nCross Validation Mean R2 Score:", cv_scores.mean())

# -------------------------------
# Feature Importance
# -------------------------------
importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\n--- Feature Importance ---")
print(importance)

plt.figure(figsize=(6,4))
sns.barplot(x="Coefficient", y="Feature", data=importance)
plt.title("Impact of Advertising Channels on Sales")
plt.show()

# -------------------------------
# Residual Analysis
# -------------------------------
residuals = y_test - y_pred

plt.figure(figsize=(6,4))
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")
plt.xlabel("Prediction Error")
plt.show()

# -------------------------------
# Actual vs Predicted Plot
# -------------------------------
plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

# -------------------------------
# User Input Prediction
# -------------------------------
print("\n--- Predict Sales for New Advertising Budget ---")

tv = float(input("Enter TV Advertising Budget: "))
radio = float(input("Enter Radio Advertising Budget: "))
newspaper = float(input("Enter Newspaper Advertising Budget: "))

user_data = np.array([[tv, radio, newspaper]])
predicted_sales = model.predict(user_data)

print(f"\nPredicted Sales: {predicted_sales[0]:.2f} units")

# -------------------------------
# Business Insights
# -------------------------------
print("\n--- Business Insights ---")

coefficients = pd.DataFrame({"Advertising Channel": X.columns, "Impact on Sales": model.coef_}).sort_values(by="Impact on Sales", ascending=False)

print(coefficients)

# Highest impact channel
top_channel = coefficients.iloc[0]
low_channel = coefficients.iloc[-1]

print(f"\nKey Insights:")
print(f"1. {top_channel['Advertising Channel']} advertising has the strongest positive impact on sales " f"with a coefficient of {top_channel['Impact on Sales']:.2f}.")

print(f"2. {low_channel['Advertising Channel']} advertising has the least impact on sales " f"with a coefficient of {low_channel['Impact on Sales']:.2f}.")

print("3. Increasing budget for high-impact channels is likely to yield better sales growth.")
print("4. Budget allocation should be optimized based on channel effectiveness shown by the model.")