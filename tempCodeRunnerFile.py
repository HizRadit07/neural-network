#main driver

from tkinter.messagebox import NO
from algorithms import sumproduct
from layer import Layer
from neuralnetwork import NeuralNetwork
from neuron import Neuron


nn = NeuralNetwork()

#create input layer
input_layer = Layer(2, None, False, "input layer")


#create output layer
output_layer = Layer(1, "sigmoid", False, "output layer", 0)
output_layer.getNeuronAtIndex(0).set_weight([-30, 20, 20])

#create hidden layer 1
hidden_layer1 = Layer(2, "sigmoid", True, "hidden layer 1")
hidden_layer1.getNeuronAtIndex(0).set_weight([-10, 20, 20])
hidden_layer1.getNeuronAtIndex(1).set_weight([30, -20, -20])

nn.add_layer(input_layer)
nn.add_layer(hidden_layer1)
nn.add_layer(output_layer)

arr1 = [1,2,3]
arr2 = [1,2,3]
#input bias value in the input
# res = nn.batch_predict([[0.0,0.0], [0.0,1.0], [1.0,0.0], [1.0,1.0]])
res = nn.predict([1.0, 0.0])
#res = sumproduct(arr1, arr2)
print(res)

#err = nn.layers[3].find_error_term([0.0, 0.0], [0,0])
# print(err)