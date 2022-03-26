#definition of layer class


from select import select
from neuron import Neuron


class Layer:
    #attributes
    neurons: list[Neuron] = None
    algorithm: str = None
    trainable: bool = None
    name: str = None
    layer_bias: float = 1
    layer_outputs: list[float] = None

    def __init__(self, numNeurons, algorithm, trainable, name, layer_bias=1) -> None:
        self.algorithm = algorithm
        self.trainable = trainable
        self.name = name
        self.layer_bias = layer_bias
        #initiate with neurons with empty weights
        self.neurons = [Neuron(algorithm,[]) for i in range (numNeurons)]

    # getter setter individual neurons
    def getNeuronAtIndex(self, idx: int) -> Neuron:
        return self.neurons[idx]

    def setNeuronAtIndex(self, new_neuron: Neuron, idx: int):
        self.neurons[idx] = new_neuron

    # methods for neurons weight setting

    # set all neuron on neuron list with the same weight
    def setAllNeuronWeights(self, set_weight: list[float]):
        for neuron in self.neurons:
            neuron.setWeight(set_weight)

    def setNeuronsWeights(self, neuron_weight: list[list[float]]):
        for i in range(len(self.neurons)):
            self.neurons[i].setWeight(neuron_weight[i])

    # set the weight of individual weight at index
    def setNeuronWeightAtIndex(self, set_weight: list[float], idx: int):
        self.neurons[idx].setWeight(set_weight)
    
    #calculate output
    def calculate_outputs(self, inputArr: list[float]) -> list[float]:
        output = [self.layer_bias] if self.layer_bias else []
        
        if self.name =="input layer":
            output.extend(inputArr)
        else:
            for neuron in self.neurons:
                output.append(neuron.calculate_output(inputArr))

        self.layer_outputs = output

        return output

    #return neuron weight at idx, misal idx = 1, and neuron ada 5
    # foreach neuron, return array of neuron.weights[1] gitu
    def get_all_neuron_weight_at_index(self, idx):
        output = []
        for neuron in self.neurons:
            output.append(neuron.weight[idx])
        return output