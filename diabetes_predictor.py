
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
data = pd.read_csv("diabetes_data.csv")

# Features and labels
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
with open("diabetes_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Diabetes model trained and saved as 'diabetes_model.pkl'")