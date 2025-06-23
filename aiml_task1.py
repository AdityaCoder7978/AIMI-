import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load CSV
df = pd.read_csv("data.csv")

# Drop Name column
df.drop(columns=["Name"], inplace=True)

# Fill missing numeric values
df.fillna(df.mean(numeric_only=True), inplace=True)

# One-hot encode categorical
df = pd.get_dummies(df, drop_first=True)

# Normalize numeric columns
scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Save cleaned data
df.to_csv("cleaned_data.csv", index=False)
print("✅ Data cleaned and saved to 'cleaned_data.csv'")
