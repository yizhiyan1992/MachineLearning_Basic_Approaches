'''Three different linear regression model
    1. LinearRegression
    2. Ridge Regression
    3. Lasso Regression
    Meanwhile, it introduces the Polynomial regression model'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

rng=np.random.RandomState(1)
x=10*rng.rand(50)
y=np.sin(x)+rng.rand(50)
plt.scatter(x,y)

poly=PolynomialFeatures(degree=7)
#replace them correspondingly
linear=LinearRegression()
linear=Lasso(alpha=0.1)
linear=Ridge(alpha=10)
pipeline=make_pipeline(poly,linear)
pipeline.fit(x[:,np.newaxis],y)
X_new=np.linspace(0,10,30)
y_new=pipeline.predict(X_new[:,np.newaxis])
plt.plot(X_new,y_new)
plt.show()