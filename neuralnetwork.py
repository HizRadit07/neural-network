

from unittest import result
from layer import Layer


class NeuralNetwork:
    #attributes
    layers: list[Layer] = None
    
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

    def get_neuron_values_per_layer(self,):
        output = []
        for layer in self.layers:
            output.append(layer.get_neuron_output_values())
        return output