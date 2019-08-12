import numpy as np
import matplotlib.pyplot as plt

class Neural_Network:
    def __init__(self):
        self.lamda =0.001
        self.learning_rate=0.2
    def initial(self,neurons_input,neurons_hidden,neurons_output):
        np.random.seed(43)
        self.theta_layer1=(0.5-np.random.random((neurons_hidden,neurons_input)))*0.3
        self.input_b=5
        self.input_b_theta=(0.5-np.random.random((neurons_hidden,1)))*0.3
        self.theta_layer2=(0.5-np.random.random((neurons_output,neurons_hidden)))*0.3
        self.hidden_b=5
        self.hidden_b_theta =(0.5-np.random.random((neurons_output, 1)))*0.3
    def activation(self,t):
        return 1/(1+np.exp(-t))

    def forward_propagation(self,input_values):
        number=len(input_values)
        self.inputlayer=[[] for _ in range(number)]
        self.hiddenlayer_z=[[] for _ in range(number)]
        self.hiddenlayer_a=[[] for _ in range(number)]
        self.outputlayer_z=[[] for _ in range(number)]
        self.outputlayer_a=[[] for _ in range(number)]
        for i in range(number):
            self.inputlayer[i]=np.hstack((np.array([self.input_b]),input_values[i,:]))
            self.hiddenlayer_z[i]=np.dot(np.hstack((self.input_b_theta,self.theta_layer1)),self.inputlayer[i])
            self.hiddenlayer_a[i]=self.activation(self.hiddenlayer_z[i])
            self.outputlayer_z[i]=np.dot(np.hstack((self.hidden_b_theta,self.theta_layer2)),np.hstack((np.array([self.hidden_b]),self.hiddenlayer_a[i])))
            self.outputlayer_a[i]=self.activation(self.outputlayer_z[i])[0]
        return self.outputlayer_a

    def cost_function(self,output,actual_values):
        output=np.array(output)
        m=len(output)
        loss=np.sum(np.square(output-actual_values))/m
        #loss=-np.sum(actual_values*np.log(self.activation(output))+(1-actual_values)*np.log(1-self.activation(output)))/m
        regularization_term=self.lamda/(2*m)*(np.sum(np.square(self.theta_layer1))+np.sum(np.square(self.theta_layer2))+np.sum(np.square(self.input_b_theta))+np.sum(np.square(self.hidden_b_theta)))
        total_loss=loss+regularization_term
        return total_loss

    def backward_propagation(self,output,actual_values):
        m=len(output)
        d_theta_hiddenlayer_accum=np.zeros(self.theta_layer2.shape[1])
        d_theta_b_hiddenlayer_accum=np.zeros(self.hidden_b_theta.shape[1])
        d_theta_inputlayer_accum=np.zeros((self.theta_layer1.shape[1],self.theta_layer1.shape[0]))
        d_theta_b_inputlayer_accum=np.zeros(self.input_b_theta.shape[1])
        for i in range(m):
            d_cost_to_a=2*(output[i]-actual_values[i])
            #d_cost_to_a=-(actual_values[i]/output[i]+(output[i]-1)/(1-actual_values[i]))
            d_a_to_z=self.activation(self.outputlayer_z[i])*(1-self.activation(self.outputlayer_z[i]))
            d_z_to_theta=self.hiddenlayer_a[i]
            d_z_to_b_theta=1
            #gradient for hidden layer
            d_theta_hiddenlayer=d_cost_to_a*d_a_to_z*d_z_to_theta
            d_theta_b_hiddenlayer=d_cost_to_a*d_a_to_z*d_z_to_b_theta
            d_theta_hiddenlayer_accum=d_theta_hiddenlayer_accum+d_theta_hiddenlayer
            d_theta_b_hiddenlayer_accum=d_theta_b_hiddenlayer_accum+d_theta_b_hiddenlayer
            # gradient for input layer
            d_z3_to_a2=self.theta_layer2
            d_a2_to_z2=self.activation(self.hiddenlayer_z[i])*(1-self.activation(self.hiddenlayer_z[i]))
            d_z2_to_theta=self.inputlayer[i][1:]
            d_theta_inputlayer = d_cost_to_a * d_a_to_z * d_z3_to_a2*d_a2_to_z2
            d_theta_inputlayer=d_z2_to_theta[:,np.newaxis]*d_theta_inputlayer
            d_theta_b_inputlayer = d_cost_to_a * d_a_to_z * d_z3_to_a2*d_a2_to_z2*d_z_to_b_theta
            d_theta_inputlayer_accum=d_theta_inputlayer_accum+d_theta_inputlayer
            d_theta_b_inputlayer_accum=d_theta_b_inputlayer_accum+d_theta_b_inputlayer
        self.theta_layer2=self.theta_layer2-(self.learning_rate*d_theta_hiddenlayer_accum.T/m+self.lamda*self.theta_layer2)
        self.hidden_b_theta=self.hidden_b_theta-(self.learning_rate*d_theta_b_hiddenlayer_accum.T/m+self.lamda*self.hidden_b_theta)
        self.theta_layer1=self.theta_layer1-(self.learning_rate*d_theta_inputlayer.T/m+self.lamda*self.theta_layer1)
        self.input_b_theta=self.input_b_theta-(self.learning_rate*d_theta_b_inputlayer.T/m+self.lamda*self.input_b_theta)
        return

def prediction_accuracy(output,actual):
    score=0
    for i in range(len(output)):
        if output[i]>=0.4:
            output[i]=1
        else:
            output[i]=0
        if  output[i]==actual[i]:
            score+=1
    return score/len(output)
if __name__=='__main__':
    from sklearn import datasets
    iris=datasets.load_iris()
    X=iris.data[:80,:]
    Y=iris.target[:80]
    # visualize two features
    plt.scatter(X[:,0],X[:,1],c=Y)
    plt.show()
    # initialize the ANN
    neurons_input=4;neurons_hidden=4;neurons_output=1
    ANN=Neural_Network()
    ANN.initial(neurons_input,neurons_hidden,neurons_output)
    output=ANN.forward_propagation(X)
    # gradient descent plot
    loss=[];x_axis=list(range(2000))
    for i in range(2000):
        ANN.backward_propagation(output,Y)
        output=ANN.forward_propagation(X)
        loss.append(ANN.cost_function(output,Y))
    # compute accuracy
    print(prediction_accuracy(output,Y))
    plt.plot(x_axis,loss)
    plt.show()

