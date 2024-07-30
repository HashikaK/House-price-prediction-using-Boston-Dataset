import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the Boston housing dataset
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

# Process the raw data
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# Define column names
columns = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM",
    "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"
]
data_df = pd.DataFrame(data, columns=columns)
data_df['PRICE'] = target

# Define features and target
X = data_df.drop(columns=['PRICE'])
y = data_df['PRICE']

# Preprocessing
# ColumnTransformer is not necessary for the Boston dataset since it has no categorical features
transformer = ColumnTransformer(
    transformers=[
        # Add transformers here if needed
    ],
    remainder='passthrough'
)

# Define the model pipeline
pipeline = Pipeline([
    ('preprocessor', transformer),
    ('model', LinearRegression())
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
pipeline.fit(X_train, y_train)

# Predict house prices for the test set
predicted_prices = pipeline.predict(X_test)

# Add predicted prices to the DataFrame
test_df = X_test.copy()
test_df['Actual Price'] = y_test
test_df['Predicted Price'] = predicted_prices

# Print predicted prices
print("Predicted Prices:")
print(test_df)

# Calculate and print Mean Squared Error
mse = mean_squared_error(y_test, predicted_prices)
print(f"Mean Squared Error: {mse:.2f}")
