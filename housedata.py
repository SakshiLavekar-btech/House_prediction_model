import pandas as pd
import numpy as np

house_data=pd.read_csv("price.csv")
test_info=pd.read_csv("test.csv")
train_info=pd.read_csv("train.csv")

print(house_data.shape)
print(test_info.shape)
print(train_info.shape)

house=train_info.merge(house_data,on="Id")
print(house.sample(5))

print(house.shape)
print(house.info())
house["ConstructedArea"]=house["GrLivArea"]+house["TotalBsmtSF"]
house["TotalWashrooms"]=house["FullBath"]+house["BsmtFullBath"]+0.5*(house["HalfBath"]+house["BsmtHalfBath"])
house["HouseAge"]=house["YrSold"]-house["YearBuilt"]
house["RemodAge"]=house["YrSold"]-house["YearRemodAdd"]
house["IsRemodeled"]=(house["YearBuilt"]!=house["YearRemodAdd"]).astype(int)
house = house[[
    # "Id",
    "MSSubClass","BldgType","HouseStyle","Foundation",
    "LotArea","MSZoning","Street","Neighborhood","Condition1",
    "OverallQual","OverallCond","KitchenQual","ExterQual",
    "ConstructedArea","LotFrontage",
    "TotalWashrooms","TotRmsAbvGrd",
    "GarageCars","GarageArea",
    "Fireplaces",
    "HouseAge","RemodAge","IsRemodeled",
    "SalePrice"
]]
house=house[house["MSZoning"]!="C"]

print(house.duplicated().sum())
print(house.shape)
print(house.isnull().sum())

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
x=house.drop(columns=['SalePrice'])
y=house['SalePrice']
print(x,y)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=24
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.decomposition import PCA

numeric_features = ['MSSubClass','LotArea','ConstructedArea','LotFrontage','GarageArea','HouseAge','RemodAge']
categorical_features = [
    'BldgType','HouseStyle','Foundation','MSZoning','Street','Neighborhood',
    'KitchenQual','ExterQual','Condition1'
]
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=5))  
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder())  
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])
from xgboost import XGBRegressor
xgb_model = XGBRegressor(
    objective="reg:squarederror",    # regression objective
    n_estimators=500,                 # number of boosted trees
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=24,
    n_jobs=-1
)
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', xgb_model)
   
])
from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    'model__n_estimators': [100, 300, 500],
    'model__max_depth': [3, 4, 5, 6],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__subsample': [0.7, 0.8, 1.0],
    'model__colsample_bytree': [0.7, 0.8, 1.0]
}

search = RandomizedSearchCV(pipe, param_distributions=param_grid,
                            n_iter=10, cv=5, scoring='r2', n_jobs=-1, random_state=24)
search.fit(x_train, y_train)

pipe = search.best_estimator_

y_pred = pipe.predict(x_test)

y_pred_df = pd.DataFrame(y_pred, columns=['SalePrice'], index=x_test.index)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

import pickle
with open('house_price_pipeline.pkl', 'wb') as f:
    pickle.dump(pipe, f)
