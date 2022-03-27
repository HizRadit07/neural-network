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
output_layer.getNeuronAtIndex(0).set_weight([0, 0, 0])

#create hidden layer 1
hidden_layer1 = Layer(2, "sigmoid", True, "hidden layer 1")
hidden_layer1.getNeuronAtIndex(0).set_weight([-10, 20, 20])
hidden_layer1.getNeuronAtIndex(1).set_weight([30, -20, -20])

nn.add_layer(input_layer)
nn.add_layer(hidden_layer1)
nn.add_layer(output_layer)

res = nn.predict([1.0, 0.0])
print(res)
# nn.printAllNeuronInput()
#nn.printAllNeuronWeight()
#nn.printAllNeuronOutput()

for i in range (1000):
    delta = nn.backprop([1.0])

    nn.update_weights(delta, 0.7)
    res2 = nn.predict([1.0, 0.0])
# for layer in nn.layers:
#     for neuron in layer.neurons:
#         print(neuron.weight)
# nn.printAllNeuronInput()
#nn.printAllNeuronWeight()
#nn.printAllNeuronOutput()

print(res2)