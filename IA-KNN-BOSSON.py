from sklearn.datasets import load_boston #Carrega o dataset de dados do boston 
from random import randint
from math import sqrt
from operator import itemgetter
from numpy import array
from numpy import arange
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

rp_global = []
teste_global = []

def targetClassificador(data, target):
    tamanho = len(data)
    t = target
    t.sort()
    baixo =  t[tamanho//3]
    medio =  t[2*(tamanho//3)]
    print("-->",baixo,"medio", medio)


    for i in range(len(target)):
        if target[i] <= baixo:
            #print("Baixo: ", baixo)
            #print(target[i] , " < " ,baixo)
            target[i] = 0
        elif target[i] <= medio:
            #print("Medio: ", medio)
            #print(target[i] , " < " ,medio)
            target[i] = 1
        elif target[i] >= medio:
            #print("Alto: ")
            #print(target[i])
            target[i] = 2
    return target


def calculaPrecisao(previsao):
    kill = 0
    miss = 0
    for i in range (len(previsao)):
        if previsao[i] == rp_global[i]:
            kill = kill + 1
        else:  miss = miss + 1
    taxaPrecisao = (kill/(kill + miss))*100
    return taxaPrecisao


def processaResultados(results):
    conclusao = []
    for x in range (len(results)):
        v1 = 0
        v2 = 0
        v3 = 0
        for y in range(len(results[x])):
                if results[x][y][13] == 0:
                    v1 = v1 + 1
                elif results[x][y][13] == 1:
                    v2 = v2 + 1
                elif results[x][y][13] == 2:
                    v3 =  v3 + 1
        if v2 >= v1 and v2 >= v3:
            conclusao.append(1)
        elif v3>=v1 and v3 >= v2:
            conclusao.append(2)
        elif v1 >= v2 and v1 >= v3:
            conclusao.append(0)
    return conclusao

def controleTestes(treinos,testes,k):
    resultados = []
    for i in range(len(testes)):
        resultados.append(KNN(treinos,testes[i],k))
    return resultados

def distanciaEuclidiana(i1, i2, lenght): #Calcula a distância euclidiana entre i1 e i2 
	distance = 0
	for x in range(lenght):
		distance += pow((i1[x] - i2[x]), 2) #Pow vem de potenciação, assim, i1 - i2 está sendo elevado ao quadrado 
	return sqrt(distance)#Retorna a raiz do quadrado da subtração de i1 e i2 (valor da distância euclidiana)

def KNN(classificados,teste,k):
    teste_global.append(teste) 
    distancias = []
    lenght = len(teste) - 1
    for x in range(len(classificados)):
        dist = distanciaEuclidiana(teste,classificados[x],lenght)
        distancias.append((classificados[x], dist))
    distancias.sort(key = itemgetter(1))
    knn = []
    for x in range(k):
        knn.append(distancias[x][0])
    return knn

def selecionaDadosTreino(data,target,k,x):
    data = data.tolist()
    target = target.tolist()
    n=0
    treino = []
    tipos=[]
    t = len(data)-1
    for i in range (x):
        c = randint(0,t-n)
        linha=[]
        for j in range (13):
            linha.append(data[c][j])
        linha.append(target[c])
        treino.append(linha)
        del data[c]
        del target[c]
        n=n+1
    for i in range((t+1)-n):
        rp_global.append(target[i])
    return controleTestes(treino,data,k)


if __name__ == "__main__":
    pass

    mediaPrecisao = []
    
    k=5 #Quantidade de individuos próximos
    x= 400 #Quantidade de individuos para treino
    votos = []
    target = []
    boston = load_boston()
    print(boston.feature_names)
    #Carrega o dataset com os dados da boston
    target = targetClassificador(boston['data'], boston['target'])
    resultados = selecionaDadosTreino(boston['data'],target,k,x)
    #divide treino e teste e passa os resultados no algoritmo knn retornando os k mais proximos
    votos = processaResultados(resultados)
    #avalia os mais proximos e decide de qual tipo são as flores
    precisao = calculaPrecisao(votos)
    #mediaPrecisao.append(precisao)
    #print(precisao)
    #print(teste_global)
    #print("MEDIA DE ACERTO: ", sum(mediaPrecisao)/100)

    x = pd.DataFrame(teste_global, columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO'
    ,'B','LSTAT'])
    y = pd.DataFrame(rp_global, columns = ['target'])
    x2 = pd.DataFrame(teste_global, columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO'
    ,'B','LSTAT'])
    y2 = pd.DataFrame(votos, columns = ['target'])
    c0 = []
    c1 = []#'ZN'
    c2 = []#'INDUS'
    c3 = []#'CHAS'
    c4 = []#'NOX'
    c5 = []#'RM'
    c6 = []#'AGE'
    c7 = []#'DIS'
    c8 = []#'RAD'
    c9 = []#'TAX'
    c10 =[]#'PTRATIO'
    c11 =[]#'B'
    c12 =[]#'LSTAT'
    tag = []
    tag2 = []
    for i in teste_global:
        c0.append(i[0])
        c1.append(i[1])
        c2.append(i[2])
        c3.append(i[3])
        c4.append(i[4])
        c5.append(i[5])
        c6.append(i[6])
        c7.append(i[7])
        c8.append(i[8])
        c9.append(i[9])
        c10.append(i[10])
        c11.append(i[11])
        c12.append(i[12])
    for i in y['target']:
        tag.append(i)
    for i in y2['target']:
        tag2.append(i)
    
    plt.figure(figsize=(12,3))
    colors = array(['r','g','b'])
    plt.subplot(1, 2, 1)
    t = [round(i) for i in tag]
    t2 =[round(i) for i in tag2]
    plt.scatter(c5, c6, c=colors[t], s=40, alpha=0.8)
    plt.title('Resultados esperados')
    plt.xlabel("RM")
    plt.ylabel('AGE')
    plt.subplot(1,2,2)
    plt.scatter(c5, c6,  c=colors[t2], s=40, alpha=0.8)
    plt.xlabel("RM")
    plt.ylabel('AGE')
    plt.title('Resultados obtidos')
    plt.show()
    #print (len(resultados))
    #print("Taxa de precisao: ",precisao)
    #avalia a taxa de acerto
    