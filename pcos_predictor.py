import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
data = pd.read_csv("pcos_data.csv")

# Split features and label
X = data.drop("PCOS", axis=1)
y = data["PCOS"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model
with open("pcos_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as 'pcos_model.pkl'")