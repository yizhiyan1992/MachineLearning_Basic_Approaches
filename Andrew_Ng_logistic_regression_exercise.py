import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

File=pd.read_table(r'C:/Users/Zhiyan/Desktop/ex2data1.txt',sep=',')
X=File.values[:,0:2];Y=File.values[:,-1]
n=X.shape[0];m=X.shape[1]
print(File.head())
print(X.shape,Y.shape)

def sigmoid(z):
    sig=1/(1+np.exp(-z))
    return sig

def gradient_descent(x,y,theta,b,n):
    # x is the feature vec; y is the target value vec; theta is the parameter vec and b is the theta0
    loss_function=-(np.sum(y*np.log(sigmoid(np.dot(x,theta.T)+b))+(1-y)*np.log(1-sigmoid(np.dot(x,theta.T)+b))))/n
    diff=sigmoid(np.dot(x,theta.T)+b)-y
    dtheta=np.dot(diff,x)/n
    db=np.sum(diff)/n
    return loss_function,dtheta,db

def optimization(x,y,theta,b,n):
    lost_values=[]
    alpha=0.001
    for i in range(150000): #number of iterations
        loss,dtheta,db=gradient_descent(x,y,theta,b,n)
        lost_values.append(loss)
        theta=theta-alpha*dtheta
        b=b-alpha*db
    return theta,b,lost_values

theta=np.array([0,0])
b=0
theta,b,lost_values=optimization(X,Y,theta,b,n)
print(theta,b)
print(lost_values[-1])

result=np.zeros((100,100))
for i in range(100):
    for j in range(100):
        result[i,j]=sigmoid(-6.25+0.055*i+0.051*j)
plt.scatter(X[:,0], X[:,1], c=Y,cmap='cool')
plt.imshow(result,cmap='binary')
plt.axis([30,100,30,100])
plt.show()