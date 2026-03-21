import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# =========================
# STEP 1: LOAD DATA
# =========================
data = pd.read_csv("Churn_Modelling.csv")

print("First 5 rows:")
print(data.head())

# =========================
# STEP 2: CLEAN DATA
# =========================
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Convert categorical variables
data = pd.get_dummies(data, drop_first=True)

print("\nCleaned Data:")
print(data.head())

# =========================
# STEP 3: VISUALIZATIONS
# =========================

# 1. Churn Distribution
plt.figure()
sns.countplot(x='Exited', data=data)
plt.title("Churn Distribution")
plt.show()

# 2. Correlation Matrix
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# 3. Age vs Churn
plt.figure()
sns.boxplot(x='Exited', y='Age', data=data)
plt.title("Age vs Churn")
plt.show()

# 4. Balance vs Churn
plt.figure()
sns.boxplot(x='Exited', y='Balance', data=data)
plt.title("Balance vs Churn")
plt.show()

# 5. Active Member vs Churn
plt.figure()
sns.countplot(x='IsActiveMember', hue='Exited', data=data)
plt.title("Active Member vs Churn")
plt.show()

# =========================
# STEP 4: MACHINE LEARNING
# =========================

X = data.drop('Exited', axis=1)
y = data['Exited']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nModel Performance:")
print(classification_report(y_test, predictions))

# =========================
# STEP 5: FEATURE IMPORTANCE
# =========================

importance = pd.Series(model.feature_importances_, index=X.columns)
importance = importance.sort_values()

print("\nFeature Importance:")
print(importance)

plt.figure()
importance.plot(kind='barh')
plt.title("Feature Importance")
plt.show()

input("Press Enter to exit...")