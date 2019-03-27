import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

np.random.seed(0)
X1=30*(np.random.rand(100,1)-0.5)
X2=100*(np.random.rand(100,1)-0.5)
outliers_x1=[[20],[24],[17],[-20],[-19]];outliers_x2=[[89],[96],[-99],[94],[-95]]
X1_total=np.vstack((X1,outliers_x1))
X2_total=np.vstack((X2,outliers_x2))
X=np.hstack((X1_total,X2_total))
plt.subplot(3,2,1)
plt.scatter(X1_total,X2_total)
plt.xlim(-100,100)
plt.ylim(-100,100)

plt.subplot(3,2,2)
scaler1=MinMaxScaler()
X_new=scaler1.fit_transform(X)
plt.scatter(X_new[:,0],X_new[:,1])

plt.subplot(3,2,3)
scaler2=MaxAbsScaler()
X_new2=scaler2.fit_transform(X)
plt.scatter(X_new2[:,0],X_new2[:,1])

plt.subplot(3,2,4)
scaler3=StandardScaler()
X_new3=scaler3.fit_transform(X)
plt.scatter(X_new3[:,0],X_new3[:,1])
plt.xlim(-2,2)
plt.ylim(-2,2)

plt.subplot(3,2,5)
scaler4=RobustScaler()
X_new4=scaler4.fit_transform(X)
plt.scatter(X_new4[:,0],X_new4[:,1])
plt.xlim(-1,1)
plt.ylim(-1,1)

plt.subplot(3,2,6)
scaler5=Normalizer()
X_new5=scaler5.fit_transform(X)
plt.scatter(X_new5[:,0],X_new5[:,1])
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()