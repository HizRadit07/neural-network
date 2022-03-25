import math

#sumproduct of equivalent length arrs
def sumproduct(arr1: list[float], arr2: list[float]) -> float:
    sum = 0
    for i in range (len(arr1)):
        sum += arr1[i] * arr2[i]
    return sum

def sigmoid(x):
    return 1/(1+math.exp(-1*x))

def relu(x):
    return max(0,x)

def softmax(outputs):
    map(lambda x: math.exp(x), outputs)
    tot = sum(outputs)
    for i in range(len(outputs)):
        outputs[i] = outputs[i]/tot
    return outputs

# linear derivative
def derLinear(x):
    return 1

# relu derivative
def derRelu(x):
    return 1 if x >=0 else 0

# sigmoid derivative
def derSigmoid(x):
    return 1/(1+math.exp(-x)) * (1 - 1/(1+math.exp(-x)))

# softmax derivative
def derSoftmax(x, j, targetClass):
    return x if j == targetClass else -1+x