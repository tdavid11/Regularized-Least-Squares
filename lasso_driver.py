import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import models
import numpy as np
data = pd.read_csv('train.csv')
X_cols=['LotArea','OverallQual','OverallCond','TotalBsmtSF','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd','GarageArea']
X = np.array(data[X_cols])
y = np.log1p(np.array(data['SalePrice']))
X = X/np.linalg.norm(X,axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

lasso = models.Lasso(5)
lasso.fit(X_train,y_train,10)
print(mean_absolute_error(y_train,lasso.predict(X_train)))
print(mean_absolute_error(y_test,lasso.predict(X_test)))