#Import dependancies
import numpy as np #For matrix math
import random #For random
from tqdm import tqdm #Progress bar

def sig(x,deriv=0): #Sigmoid function and derivative
    if deriv:
        return sig(x)*(1-sig(x))
    else:
        return 1/(1+np.exp(-x))

class neural_network: #Feed Forward Network Class
    def __init__(self, x_size, y_size, h_size): #Subroutine to intiailise network
        #Normal distribution parameters
        sigma = 0.2
        mu = 0

        #Network Dimensions
        self.x_size = x_size
        self.y_size = y_size
        self.h_size = h_size

        #Weight matrices
        self.input_weights = np.random.normal(mu, sigma, (h_size,x_size))
        self.hidden_weights_one = np.random.normal(mu, sigma, (h_size, h_size))
        self.output_weights = np.random.normal(mu, sigma, (y_size, h_size))

        #Bias vectors
        self.hidden_bias_one = np.random.normal(mu, sigma,(self.h_size, 1))
        self.hidden_bias_two = np.random.normal(mu, sigma,(self.h_size, 1))
        self.output_bias = np.random.normal(mu, sigma,(self.y_size, 1))

    def forward(self, x): #Function to perform forward pass through network
        hidden_state_one = sig(np.dot(self.input_weights, x) + self.hidden_bias_one)
        hidden_state_two = sig(np.dot(self.hidden_weights_one, hidden_state_one) + self.hidden_bias_two)
        return sig(np.dot(self.output_weights, hidden_state_two) + self.output_bias)

    def back(self, t_in, t_out, alpha):
        #Reshape training data to explicitily define dimensions
        t_in = np.reshape(t_in, (self.x_size, 1))
        t_out = np.reshape(t_out, (self.y_size, 1))

        #Explicitly define dimensions of partial derivatives and layer states
        hidden_state_one = np.zeros((self.h_size,1))
        hidden_state_one_sig = np.zeros((self.h_size,1))
        hidden_state_two = np.zeros((self.h_size,1))
        hidden_state_two_sig = np.zeros((self.h_size,1))
        a_out = np.zeros((self.y_size,1))
        a_out_sig = np.zeros((self.y_size,1))
        d_output = np.zeros((self.y_size,1))
        d_hidden_one = np.zeros((self.h_size,1))
        d_hidden_two = np.zeros((self.h_size, 1))

        #forward pass
        hidden_state_one = np.dot(self.input_weights, t_in) + self.hidden_bias_one
        hidden_state_one_sig = sig(hidden_state_one)
        hidden_state_two = np.dot(self.hidden_weights_one, hidden_state_one_sig) + self.hidden_bias_two
        hidden_state_two_sig = sig(hidden_state_two)
        a_out = np.dot(self.output_weights, hidden_state_two_sig) + self.output_bias
        a_out_sig = sig(a_out)

        #backpropogation of errors
        d_output = (a_out_sig - t_out) * sig(a_out, True)
        d_hidden_two = np.dot(self.output_weights.T, d_output) * sig(hidden_state_two, True)
        d_hidden_one = np.dot(self.hidden_weights_one.T, d_hidden_two) * sig(hidden_state_one, True)

        #update weights
        self.output_weights -= alpha * np.dot(d_output, hidden_state_two_sig.T)
        self.hidden_weights_one -= alpha * np.dot(d_hidden_two, hidden_state_one_sig.T)
        self.input_weights -= alpha * np.dot(d_hidden_one, t_in.T)

        self.output_bias -= alpha * d_output
        self.hidden_bias_one -= alpha * d_hidden_one
        self.hidden_bias_two -= alpha * d_hidden_two


print('--Feed Forward Neural Network solving MNIST--')
#1) Load MNIST
data_file = open('MNIST\mnist_train.csv', 'r')
dataset = data_file.readlines()
data_file.close()

#2) Initialise neural network class
input_size = 784
output_size = 10
hidden_size = 500

training_iterations = 80000

nn = neural_network(input_size,output_size,hidden_size)
#training loop
print('Training:')
for _ in tqdm(range(training_iterations)):
    #select random sample from training set
    sample_no = random.randint(1,len(dataset))
    all_values = dataset[sample_no - 1].split(',')

    #split training sample into input and output
    t_in = np.zeros((input_size, 1))
    t_in = ((np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01).reshape(input_size, 1)
    t_out = np.zeros((output_size, 1)) + 0.01
    t_out[int(all_values[0])] = 0.99

    #backpropogate partial derivatives and update weights using training sample
    nn.back(t_in,t_out, 0.001)

#test loop
print('')
print('Evaluating Network:')
test_file = open('MNIST\mnist_test.csv', 'r')
testset = test_file.readlines()
test_file.close()
correct_count = 0
for _ in tqdm(range(len(testset))):
    all_values = testset[_-1].split(',')
    test_in = np.zeros((input_size, 1))
    test_in = ((np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01)
    test_out = np.zeros((output_size, 1)) + 0.01
    test_out[int(all_values[0])] = 0.99

    y = nn.forward(test_in)
    if np.argmax(y) == np.argmax(test_out): #if prediction matches target then increment count
        correct_count+=1

print('Network accuracy {0}%'.format((correct_count/len(testset)) * 100)) #calculate percentage correct
