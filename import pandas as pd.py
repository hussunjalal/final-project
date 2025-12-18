import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

data = {
    'Area': [100, 150, 120, 200, 180, 90, 130, 210, 160, 110],
    'Bedrooms': [2, 3, 2, 4, 3, 2, 3, 4, 3, 2],
    'Bathrooms': [1, 2, 1, 2, 2, 1, 1, 3, 2, 1],
    'Price': [150000, 250000, 180000, 350000, 300000,
              130000, 200000, 400000, 280000, 160000]
}

df = pd.DataFrame(data)

X = df[['Area', 'Bedrooms', 'Bathrooms']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"R-squared: {r2:.2f}")

area = float(input("Enter area: "))
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = int(input("Enter number of bathrooms: "))

user_data = pd.DataFrame([[area, bedrooms, bathrooms]],
                         columns=['Area', 'Bedrooms', 'Bathrooms'])

predicted_price = model.predict(user_data)

print(f"Predicted house price: {predicted_price[0]:.2f}")