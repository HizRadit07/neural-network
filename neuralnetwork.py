

from re import I
from unittest import result
from algorithms import sumIntAndArray, sumproduct
from layer import Layer
from neuron import Neuron

class NeuralNetwork:
    #attributes
    layers: list[Layer] = None
    #current backpropErrTerms
    errTerm: list[float] = None

    #constructor
    def __init__(self) -> None:
        self.layers = []
    
    #add a layer
    def add_layer(self, newlayer):
        self.layers.append(newlayer)

    #predict output from a given input
    def predict(self, input: list[float]) -> list[float]:
        result = input
        for layer in self.layers:
            result = layer.calculate_outputs(result)
        
        return result

    #predict output via batch predict
    def batch_predict(self, input: list[list[float]]) -> list[list[float]]:
        result = []
        for data in input:
            result.append(self.predict(data))
        #returns a list of result of predict
        #nanti di akhir, utk perhitungan, biasanya di average
        return result

    #find unit gradient of neural network
    #output is an array that corresponds to all weight in the neural network
    def backprop(self, target):
        grad = []
        backPropErrTerm = [] #init awal
        nextlayer = None
        finalErrTermArr = []

        for layer in reversed(self.layers):
            #first check is always output layer, so its fine to init nextlayer in the bottom
            if layer.name == "output layer": #for output error
                i = 0
                for neuron in layer.neurons:
                    for x in neuron.x:
                        grad.append((target[i]-neuron.output) * neuron.count_derivative(neuron.output) * x)
                        backPropErrTerm.append((target[i]-neuron.output) * neuron.count_derivative(neuron.output))
                    i+=1
                finalErrTermArr.extend(backPropErrTerm)
            else: #hidden or input layer
                #pass
                tempBackPropErrArr = []
                if layer.name == "input layer": #pass input layer
                    break
                for neuron in layer.neurons:
                    i = 0
                    for x in neuron.x:
                        nextLayerWeights = nextlayer.get_all_neuron_weight_at_index(i)
                        grad.append(neuron.count_derivative(neuron.output) * sumIntAndArray(backPropErrTerm[i], nextLayerWeights) * x)
                        tempBackPropErrArr.append(neuron.count_derivative(neuron.output) * sumIntAndArray(backPropErrTerm[i], nextLayerWeights))
                    i+=1
                backPropErrTerm = tempBackPropErrArr
                finalErrTermArr.extend(backPropErrTerm)
            #set the next layer = now layer, for the next iteration
            nextlayer = layer
        #save errTerm array in attribute
        self.errTerm = finalErrTermArr
        #print(finalErrTermArr)
        return finalErrTermArr

    #print all neuron inputs
    def printAllNeuronInput(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.printInput()
        
    #print neuron weights
    def printAllNeuronWeight(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                print(neuron.weight)
    
    #print neuron weights
    def printAllNeuronOutput(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                print(neuron.output)
    #update all weights by input
    #layers is reversed, because input should be a reversed array containing negative gradient
    def update_weights(self, input, learn_rate):
        for layer in reversed(self.layers):
            for neuron in layer.neurons:
                for i in range(len(neuron.weight)):
                    neuron.weight[i] += learn_rate * input[i] * neuron.x[i]
