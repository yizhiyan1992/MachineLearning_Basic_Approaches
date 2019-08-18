from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
import numpy as np
# download the dataset -face recognition
lfw_people=fetch_lfw_people()
m=100 #use the first 100 images
X=lfw_people.images[:m,:,:]
print(X.shape)

# Principal component analysis
'''Step1 : mean normalization (this step is necessary for PCA
   step2 : derive the eigenvectors and eigenvalues for covariance matrix (1/m)*X.T*X
   step3 : build the contribution curve to find out how many information we want to save
   step4 : use the first k eigenvectors to reduce the dimension X*P
   hint: when we want to display samples after dimension reduce at the original coordinates, we need to *P.T'''
X_new=X.reshape(m,2914)
X_new=X_new-np.mean(X_new,axis=0)/np.std(X_new,axis=0)
covariance=np.dot(X_new.T,X_new)/m
eigen_values,eigen_vectors=np.linalg.eig(covariance)
print(eigen_values,eigen_vectors.shape)

#plot the contribution curve for the first k eigenvectors
eigen_sum=np.sum(eigen_values)
contribution=[np.sum(eigen_values[:i])/eigen_sum for i in range(len(eigen_values))]
plt.plot(range(len(contribution)),contribution)
plt.axis([0,50,0,1])
plt.show()
#use only 10 eigen vectors
eigen_vectors_10=eigen_vectors[:,:10]
Y_10=np.dot(X_new,np.dot(eigen_vectors_10,eigen_vectors_10.T))
#use the first 50 eigen vectors
eigen_vectors_50=eigen_vectors[:,:50]
Y_50=np.dot(X_new,np.dot(eigen_vectors_50,eigen_vectors_50.T))

#plot the first four pictures
for i in range(4):
    plt.subplot(3,4,i+1)
    plt.imshow(X_new[i,:].reshape((62,47)), cmap='viridis')
    plt.subplot(3,4,4+i+1)
    plt.imshow(Y_10[i,:].reshape(62,47),cmap='viridis')
    plt.subplot(3,4,8+i+1)
    plt.imshow(Y_50[i,:].reshape(62, 47),cmap='viridis')
plt.show()

'''note: although PCA and SVD both can achieve data compression, there are different method.
   PCA can be implemented by SVD'''
