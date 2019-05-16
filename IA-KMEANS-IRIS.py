from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from copy import deepcopy
from matplotlib import pyplot as plt 
import pandas as pd
import numpy as np
import random


def taxaDeAcerto(real, obtido):
    kill = 0
    miss = 0
    for i in range(len(real)):
        if real[i] == obtido[i]:
            print("[",real[i]," : ",obtido[i],"]", "Acertou  ")
            kill+=1
        else:
            miss+=1
            print ("[",real[i]," : ",obtido[i],"]","Errou  ")
    
    return ((kill-miss)/150)*100

if __name__ == "__main__":
    pass

    iris = load_iris()

    data = iris.data

    x = pd.DataFrame(iris.data, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
    y = pd.DataFrame(iris.target, columns=['Target'])
    
    modelo = KMeans(n_clusters=3, n_jobs = 4, random_state=21)
    modelo.fit(x)
    new_labels = modelo.labels_

    # PLOT 
    plt.figure(figsize=(12,3))
    
    colors = np.array(['red', 'green', 'blue'])


    predictedY = np.choose(modelo.labels_, [1, 0, 2]).astype(np.int64)
    precisao = taxaDeAcerto(y['Target'], predictedY)
    print (precisao)
    
    plt.subplot(1, 2, 1)
    plt.scatter(x['Petal Length'], x['Petal Width'], c=colors[y['Target']], s=40)
    plt.xlabel("Petal Length (cm)")
    plt.ylabel("Petal Width (cm)")
    plt.title('Resultados esperados')
    
    plt.subplot(1, 2, 2)
    plt.scatter(x['Petal Length'], x['Petal Width'], c=colors[predictedY], s=40)
    plt.scatter(modelo.cluster_centers_[:,2] , modelo.cluster_centers_[:,3], marker='*', s=200, c='m')
    plt.xlabel("Petal Length (cm)")
    plt.ylabel("Petal Width (cm)")
    plt.title("Resultados obtidos")

    plt.show()


    plt.subplot(1, 2, 1)
    plt.scatter(x['Sepal Length'], x['Sepal Width'], c=colors[y['Target']], s=40)
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Sepal Width (cm)")
    plt.title('Resultados esperados')
    
    plt.subplot(1, 2, 2)
    plt.scatter(x['Sepal Length'], x['Sepal Width'], c=colors[predictedY], s=40)
    plt.scatter(modelo.cluster_centers_[:,0] , modelo.cluster_centers_[:,1], marker='*', s=200, c='m')
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Sepal Width (cm)")
    plt.title("Resultados obtidos")
    plt.show()
    