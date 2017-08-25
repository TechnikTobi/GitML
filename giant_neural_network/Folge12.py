#WICHTIG: pythonw -i 12.py verwenden um Dokument im Interactive Mode zu oeffnen

#matplotlib inline
#Wird nur benutzt für jupyter

from matplotlib import pyplot as plt    #Zum Darstellen der Graphen
import numpy as np                      #Fuer mathematisches Zeug
import os


# Verteilung der Daten
data = [
    [3  , 1.5, 1],
    [3  , 1  , 0],
    [4  , 1.5, 1],
    [2  , 1  , 0],
    [3.5, 0.5, 1],
    [2  , 0.5, 0],
    [5.5, 1  , 1],
    [1  , 1  , 0]
    ]

mystery_flower = [4.5, 1]


# Verteilt Zufallswerte an w1, w2 und b
w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()


# Funktionen
def sigmoid(x):                         #Zum quetschen von Zahlen auf einen Wert zwischen 0 und 1
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1-sigmoid(x))

def graph_plot(achse1, achse2):         #Fuer die graphische Darstellung von Sachen
    plt.show(plt.plot(achse1, achse2))


# Training Loop

learningRate = 0.1
costs = []

for i in range(50000):
    randomPointIndex = np.random.randint(len(data))
    point = data[randomPointIndex]

    z = point[0] * w1 + point[1] * w2 + b
    prediction = sigmoid(z)

    target = point[2]
    cost = (prediction - target) ** 2

    derivativeOfCostWithRespectToPrediction = 2 * (prediction - target)
    derivativeOfPredictionWithRespectToZ = sigmoid_derivative(z)
    derivativeOfZWithRespectToW1 = point[0]
    derivativeOfZWithRespectToW2 = point[1]
    derivativeOfZWithRespectToB = 1

    derivativeOfCostWithRespectToW1 = derivativeOfCostWithRespectToPrediction * derivativeOfPredictionWithRespectToZ * derivativeOfZWithRespectToW1
    derivativeOfCostWithRespectToW2 = derivativeOfCostWithRespectToPrediction * derivativeOfPredictionWithRespectToZ * derivativeOfZWithRespectToW2
    derivativeOfCostWithRespectToB = derivativeOfCostWithRespectToPrediction * derivativeOfPredictionWithRespectToZ * derivativeOfZWithRespectToB

    #Nicht vergessen derivativeofCostWithRespect zum RICHTIGEN Parameter nehmen ;)
    w1 = w1 - learningRate * derivativeOfCostWithRespectToW1
    w2 = w2 - learningRate * derivativeOfCostWithRespectToW2
    b = b - learningRate * derivativeOfCostWithRespectToB

    costSum = 0
    for point in data:

        prediction = sigmoid(point[0] * w1 + point[1] * w2 + b)
        costSum += (prediction - point[2]) ** 2

    costs.append(costSum)

#plt.plot(costs)
#plt.show()


# Predict

def which_flower(length, width):
    z = sigmoid(length * w1 + width * w2 +b)
    if z < .5:
        os.system("say blau")
    else:
        os.system("say rot")
