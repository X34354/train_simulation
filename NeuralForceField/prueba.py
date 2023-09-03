import pandas
import numpy as np

def promedio(lista):

    return np.mean(lista)


numeros = [10, 20, 30, 40, 50]
promedio_numeros = promedio(numeros)
print("El promedio de los números es:", promedio_numeros)


np.mean( [10, 20, 30, 40, 50])