#definition of layer class


from neuron import Neuron


class Layer:
    #attributes
    neurons: list[Neuron] = None
    algorithm: str = None
    trainable: bool = None
    name: str = None
    layer_bias: float = 1

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
        
        if self.name == "input layer":
            output.extend(inputArr)
        else:
            for neuron in self.neurons:
                output.append(neuron.calculate_output(inputArr))

        return output
    