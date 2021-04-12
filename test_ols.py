import numpy as np
import models
X = np.random.normal(size=(1000,20))
scalars = np.random.normal(size=(20))
print(scalars)
y= np.dot(X,scalars)
ols = models.OLS()
ols.fit(X,y)