import numpy as np
import models
from sklearn.preprocessing import MinMaxScaler
X = np.random.normal(size=(1000,20))
scalars = np.random.normal(size=(20))
mm = MinMaxScaler()
#X = mm.fit_transform(X)
X = X/np.linalg.norm(X,axis=0)
print(scalars)
y= np.dot(X,scalars)
Lasso = models.Lasso(.01)
Lasso.fit(X,y,10)