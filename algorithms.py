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