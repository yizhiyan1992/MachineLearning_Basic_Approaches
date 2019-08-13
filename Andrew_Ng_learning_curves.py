import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer

def linear_regression(x,theta,b):
    linear=theta*x+b
    return linear

def polynomial_regression(x,theta,b):
    poly=x*theta+b
    return poly
def loss_function(actual,x,theta,b,lamda):
    m=len(actual)
    '''change the linear/poly function'''
    #predicted = linear_regression(x, theta, b)
    predicted = polynomial_regression(x, theta, b)
    loss=np.sum(np.square(predicted-actual))/(2*m)+lamda*np.sum(np.square(theta))/(2*m)
    return loss

def gradient_descent(actual,x,theta,b,lamda):
    alpha=0.001
    iter=2000
    m=len(actual)
    loss_value=[]
    for i in range(iter):
        '''change the linear/poly function'''
        #predicted=linear_regression(x,theta,b)
        predicted=polynomial_regression(x,theta,b)
        d_theta = np.sum((predicted - actual) * x,axis=0) / m + lamda * theta / m
        d_b = np.sum(predicted - actual) / m
        theta=theta-alpha*d_theta
        b=b-alpha*d_b
        loss_value.append(loss_function(actual,x,theta,b,lamda))
    return loss_value,theta,b

if __name__=='__main__':
    data=sio.loadmat(r'C:/Users/Zhiyan/Desktop/ex5data1.mat')
    X_train,y_train=data['X'],data['y']
    X_validation,y_validation=data['Xval'],data['yval']
    X_test,y_test=data['Xtest'],data['ytest']
    plt.scatter(X_train,y_train,c='red',marker='*')
    plt.show()

    #initialize the parameters
    lamda=1
    theta=np.array([1]);b=-2
    theta=np.array([1,1,1,1,1,1])
    #print(np.mean(X_train,axis=1).shape)
    X_train=np.hstack((X_train,np.square(X_train),X_train*np.square(X_train),np.square(X_train)*np.square(X_train),\
                      X_train*np.square(X_train)*np.square(X_train),np.square(X_train)*np.square(X_train)*np.square(X_train)))
    print('kkk',np.mean(X_train,axis=0).shape)
    X_train=(X_train-np.mean(X_train,axis=0))/np.std(X_train,axis=0)
    loss,theta,b=(gradient_descent(y_train,X_train,theta,b,lamda))
    print(loss[0],loss[-1],theta,b)
    plt.plot(range(len(loss)),loss)
    plt.show()
    #linear
    #plt.scatter(X_train, y_train, c='red', marker='*')
    #plt.plot(X_train,X_train*0.363+12.32)
    #poly
    plt.scatter(X_train[:,0],y_train, c='red', marker='*')
    plt.scatter(X_train[:,0],np.dot(X_train,theta)+11)
    plt.show()

    #plot the learning_curve
    #theta = np.array([1]);b = -1
    #loss_train=[];loss_cross=[]
    #for i in range(len(X_train)):
    #    X_new=X_train[:i+1]
    #    y_new=y_train[:i+1]
    #    loss, theta, b = (gradient_descent(y_new, X_new, theta, b,lamda))
    #    loss_train.append(loss[-1])
    #    loss_cross.append(loss_function(y_validation,X_validation,theta,b,lamda))
    #plt.plot(range(len(X_train)),loss_train,label='train_loss')
    #plt.plot(range(len(X_train)), loss_cross,label='cross_validation_loss')
    #plt.legend()
    #plt.axis([0,12,0,150])
    #plt.show()

    #plot the error with the change of regularization term
    regu=[0,0.01,0.1,1,10,25,50,100]
    X_validation = (X_validation - np.mean(X_validation, axis=0)) / np.std(X_validation, axis=0)
    regu_train=[];regu_cross=[]
    for i in range(len(regu)):
        loss, theta, b = (gradient_descent(y_train, X_train, theta, b, regu[i]))
        regu_train.append(loss[-1])
        regu_cross.append(loss_function(y_validation,X_validation,theta,b,regu[i]))
    print(regu_train)
    print(regu_cross)
    plt.plot(range(len(regu)),regu_train,label='train_loss')
    plt.plot(range(len(regu)), regu_cross,label='cross_validation_loss')
    plt.legend()
    plt.show()
    #the trend is not very obvious because the degree of polynomial is not that high