import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
File=pd.read_table(r'C:/Users/Zhiyan/Desktop/ex2data2.txt',sep=',')
print(File.head())
print(File.values.shape)
X=File.values[:,0:2]
Y=File.values[:,-1]
n=X.shape[0];m=X.shape[1]
plt.scatter(X[:,0],X[:,1],c=Y,cmap='cool')
#plt.show()

#feature expansion to 3-dimensional
F2=X[:,0];F3=X[:,1];F4=F2*F3;F5=F2*F3*F3;F6=F2*F2*F3;F7=F2*F2*F2;F8=F3*F3*F3;F9=F2*F2*F2*F2;
F10=F3*F3*F3*F3;F11=F2*F3*F3*F3;F12=F3*F2*F2*F2;F13=F2*F2*F3*F3;F14=F2*F2*F2*F2*F2;F15=F3*F3*F3*F3*F3
F16=F2*F2*F2*F2*F2*F2;F17=F3*F3*F3*F3*F3*F3;F18=F2*F2*F2*F3*F3;F19=F3*F3*F2*F2*F2;F20=F2*F3*F3*F3*F3;F21=F3*F2*F2*F2*F2
X_new=np.vstack((F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13,F14,F15,F16,F17,F18,F19,F20,F21))
X_new=X_new.T
print(X_new.shape)

def sigmoid(t):
    return 1/(1+np.exp(-t))

#add regularization term
def loss_funtion(x,y,theta,b,n):
    A=sigmoid(np.dot(x,theta.T)+b)
    loss=-np.sum(y*np.log(A)+(1-y)*np.log(1-A))/n+0.1/(2*n)*np.sum(theta*theta)
    dtheta=np.dot((A-y),x)/n+0.1/n*theta
    db=np.sum(A-y)/n
    return loss,dtheta,db

def gradient_descent(x,y,theta,b,n):
    loss_values=[]
    alpha=0.1
    for i in range(200000):
        loss,dtheta,db=loss_funtion(x,y,theta,b,n)
        loss_values.append(loss)
        theta=theta-alpha*dtheta
        b=b-alpha*db
    return loss_values,theta,b

theta=np.array([0]*20)
b=0
loss,theta,b=gradient_descent(X_new,Y,theta,b,n)
print(loss[-1],theta,b)

result=np.zeros((200,200))
for i in range(200):
    for j in range(200):
        o1=(i-100)/100;o2=(j-100)/100
        result[i,j]=sigmoid(np.dot(np.array([o1,o2,o1*o2,o1*o2*o2,o1*o1*o2,o1*o1*o1,o2*o2*o2,o1*o1*o1*o1,\
        o2*o2*o2*o2,o1*o2*o2*o2,o2*o1*o1*o1,o1*o1*o2*o2,o1*o1*o1*o1*o1,o2*o2*o2*o2*o2,o1*o1*o1*o1*o1*o1,o2*o2*o2*o2*o2*o2,\
        o1*o1*o1*o2*o2,o2*o2*o1*o1*o2,o1*o2*o2*o2*o2,o2*o1*o1*o1*o1]),theta.T)+b)
print(result)
plt.scatter(X[:,0], X[:,1], c=Y,cmap='cool_r')
plt.imshow(result,cmap='cool',extent=[-1,1,-1,1])
#plt.axis([-1,1,-1,1])
plt.show()