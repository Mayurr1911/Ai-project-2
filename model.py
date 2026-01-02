import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np

# Generate synthetic real estate data
np.random.seed(42)
n_samples = 1000
size = np.random.randint(500, 5000, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)
location_score = np.random.uniform(0, 10, n_samples)
price = size * 100 + bedrooms * 5000 + location_score * 10000 + np.random.normal(0, 10000, n_samples)

data = pd.DataFrame({'size': size, 'bedrooms': bedrooms, 'location_score': location_score, 'price': price})

X = data[['size', 'bedrooms', 'location_score']]
y = data['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse:.2f}")

# Save model
joblib.dump(model, 'real_estate_model.pkl')
