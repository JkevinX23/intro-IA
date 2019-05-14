from sklearn.datasets import load_boston #Carrega o dataset de dados do boston 
from random import randint
from math import sqrt
from operator import itemgetter
from numpy import array

rp_global = []

def targetClassificador(data, target):
    tamanho = len(data)
    t = target
    t.sort()
    baixo =  t[tamanho//3]
    medio =  t[2*(tamanho//3)]


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
    k=5 #Quantidade de individuos próximos
    x= 400 #Quantidade de individuos para treino
    votos = []
    target = []
    boston = load_boston() 
    #Carrega o dataset com os dados da boston
    target = targetClassificador(boston['data'], boston['target'])
    resultados = selecionaDadosTreino(boston['data'],target,k,x)
    #divide treino e teste e passa os resultados no algoritmo knn retornando os k mais proximos
    votos = processaResultados(resultados)
    #avalia os mais proximos e decide de qual tipo são as flores
    precisao = calculaPrecisao(votos)
    print(precisao)
    #print (len(resultados))
    #print("Taxa de precisao: ",precisao)
    #avalia a taxa de acerto