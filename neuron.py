#definition of a neuron class

from algorithms import derLinear, derRelu, derSigmoid, relu, sigmoid, sumproduct


class Neuron:
    #attributes
    algorithm: str = None
    #weight from previous
    weight: list[float] = None
    #output value
    output: float = None
    # x value / input value
    x: list[float] = None

    #constructor
    def __init__(self, algorithm: str, weight: list[float]) -> None:
        self.algorithm = algorithm
        self.weight = weight
    #weight getter setter
    def get_weight(self,) -> list[float]:
        return self.weight
    def set_weight(self, new_weights:list[float]):
        self.weight = (new_weights)
    #calculate output given an inputArr
    #bias is some int, didefinisiin di layer
    def calculate_output(self, inputArr, ):
        # print(inputArr)

        if self.algorithm is None: #input layer
            return inputArr

        sumweights = sumproduct(self.weight, inputArr)
        self.x = inputArr
        
        if self.algorithm.lower() == "linear":
            self.output = sumweights
            return sumweights
        elif self.algorithm.lower() == "sigmoid":
            self.output = sigmoid(sumweights)
            return sigmoid(sumweights)
        elif self.algorithm.lower() == "relu":
            self.output = relu(sumweights)
            return relu(sumweights)
        else:
            return inputArr

    #count derivative
    def count_derivative(self, output):
        if self.algorithm.lower() == "linear":
            return derLinear(output)
        elif self.algorithm.lower() == "sigmoid":
            return derSigmoid(output)
        elif self.algorithm.lower() == "relu":
            return derRelu(output)
        else:
            return output
    
    
    
    