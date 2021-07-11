import pandas as pd
from sklearn import linear_model
df = pd.read_csv('homeprices.csv')
# as data is missing for 1 house we're handling it
df.bedrooms.median()
df.bedrooms = df.bedrooms.fillna(df.bedrooms.median())
reg = linear_model.LinearRegression()
reg.fit(df.drop('price', axis='columns'), df.price)
print(reg.coef_)
print(reg.intercept_)
# Find price of home with 3000 sqr ft area, 3 bedrooms, 40 year old
reg.predict([[3000, 3, 40]])

reg.predict([[2500, 4, 5]])
