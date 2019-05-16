from sklearn.datasets import load_iris #Carrega o dataset de dados da iris
from random import randint
from math import sqrt
from operator import itemgetter
from numpy import array
from numpy import choose
import  numpy as np
import matplotlib.pyplot as plt
import pandas as pd

rp_global = []
testes_global = []


'''
Kill é a quantidade de acertos
Miss é a quantidade de erros
previsão, são as classificações geradas pelo algoritmo
rp_global é a classificação correta 
se a previsão for igual a classificação correta, incremento kill (acertos)
se não, incremento miss(erro)

a função retorna a porcentagem de acertos, que é dada por kill(acertos) /kill+miss(total) *100 (para devolver a porcentagem)
'''
def calculaPrecisao(previsao):
    kill = 0
    miss = 0
    for i in range (len(previsao)):
        if previsao[i] == rp_global[i]:
            kill = kill + 1
        else:  miss = miss + 1
    taxaPrecisao = (kill/(kill + miss))*100
    return taxaPrecisao

'''
Função Processa resultados
recebe o retorno de controleTestes(uma lista de lista com todos os k individuos mais próximos para cada teste)
faz uma votação para classificar os testes. Ou seja: 
é observado a classificação de cada um dos k individuos próximos
se for 0, v1 é incrementado
se for 1, v2 é incrementado
se for 2, v3 é incrementado
No final, os individuos são classificados de acordo com o tipo que mais apareceu (O mais votado)
'''
def processaResultados(results):
    conclusao = []
    for x in range (len(results)):
        v1 = 0
        v2 = 0
        v3 = 0
        for y in range(len(results[x])):
                if results[x][y][4] ==  0 :
                    v1 = v1 + 1
                elif results[x][y][4] == 1:
                    v2 = v2 + 1
                elif results[x][y][4] == 2:
                    v3 =  v3 + 1
        if v2 > v1 and v2 > v3:
            conclusao.append(1)
        elif v3>v1 and v3 > v2:
            conclusao.append(2)
        elif v1 > v2 and v1 > v3:
            conclusao.append(0)
    return conclusao
'''
Função controleTestes
    a função controleTestes, passa todos os testes na função KNN e armazena seu resultado

retorno, uma lista de lista com todos os k individuos mais próximos para cada teste
'''
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

'''
Função KNN
Classificados são os dados de treino, e teste os dados a serem classifiicados
e k é a quantidade de elementos proximos/classificados que serão retornados.

o retorno é uma lista com os k elementos proximos para cada individuo
'''
def KNN(classificados,teste,k):
    testes_global.append(teste)
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
'''
SelecionaDadosTreino irá separar os dados em treino e teste. 
SelecionaDadosTreino recebe o ['data'] que são os dados na variavel iris
e typeFlor são as classificações desses dados (tipo da flor para cada individuo de data)
x é a quantidade de elementos que irão pro treinamento, o restante servirá para os testes
iris e typeflor são arrays, assim o tolist() transforma-o em lista (Não consigo remover elementos de um array)
n é uma variavel de controle
a função seleciona os x individuos aleatoriamente, insere o tipo dela no final da linha e 
deleta da lista para que não possa ser escolhido novamente

os valores que sobram na lista serão os dados de teste, assim, não é anexado a resposta.. estas serão armazenadas
em rp_global para que a taxa de acertos seja gerada futuramente.

'''
def selecionaDadosTreino(iris,typeFlor,k,x):
    iris = iris.tolist()
    typeFlor = typeFlor.tolist()
    n=0
    treino = [] # Treino recebe 'x' individuos 
    tipos=[]
    t = len(iris)-1
    for i in range (x):
        c = randint(0,t-n)
        linha=[]
        for j in range (4):
            linha.append(iris[c][j])
        linha.append(typeFlor[c])
        treino.append(linha)
        del iris[c]
        del typeFlor[c]
        n=n+1
    for i in range((t+1)-n):
        rp_global.append(typeFlor[i])
    return controleTestes(treino,iris,k)
    '''
    os treinos serão chamados na função  controleTestes(), pois a função KNN recebe todos os dados de treino
    e um dado de teste para classifica-lo
    '''

def printPrecisao(real, esperado):
    
    for i in range (len(real)):
        if real[i] == esperado [i]:
            print("[",real[i]," : ",esperado[i],"]", "Acertou  ")
        else: print ("[",real[i]," : ",esperado[i],"]","Errou  ")
    

if __name__ == "__main__":
    pass
    mediaPrecisao = []
    k=5 #Quantidade de individuos próximos
    x= 100 #Quantidade de individuos para treino - 100/150 -> 66.67%
    votos = []
    iris = load_iris() #Carrega o dataset com os dados da iris 
    resultados = selecionaDadosTreino(iris['data'],iris['target'],k,x) 
    #divide treino e teste e passa os resultados no algoritmo knn retornando os k mais proximos
    votos = processaResultados(resultados)
    #avalia os mais proximos e decide de qual tipo são as flores
    precisao = calculaPrecisao(votos)
    printPrecisao(votos,rp_global)
    #print("Taxa de acerto: ", precisao)
    #avalia a taxa de acerto 
    total = 0
    #for i in range(100):
    #total+=mediaPrecisao[i]
    #print("PRECISAO MEDIA: ",total/100 )
    #print(array(testes_global[0][3]))
    #c =testes_global[:len(testes_global)][0]
   
    print(len(testes_global))
    x = pd.DataFrame(testes_global, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
    y = pd.DataFrame(votos, columns=['Target'])
    x2 = pd.DataFrame(testes_global, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
    y2 = pd.DataFrame(rp_global, columns=['Target'])
    plt.figure(figsize=(12,3))
    colors = np.array(['red', 'green', 'blue'])
    #nrows=1, ncols=2, plot_number=1
    plt.subplot(1, 2, 1)
    plt.scatter(x['Sepal Length'], x['Sepal Width'], c=colors[y['Target']], s=40, alpha=0.8)
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Sepal Width (cm)")
    plt.title('Resultado obtido')

    plt.subplot(1,2,2)
    plt.scatter(x2['Sepal Length'], x2['Sepal Width'], c= colors[y2['Target']], s=40, alpha= 0.8)
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Sepal Width (cm)")
    plt.title('Resultado esperado')
    
    plt.show()

    colors = np.array(['red', 'green', 'blue'])
    #nrows=1, ncols=2, plot_number=1
    plt.subplot(1, 2, 1)
    plt.scatter(x['Petal Length'], x['Petal Width'], c=colors[y['Target']], s=40, alpha= 0.8)
    plt.xlabel("Petal Length (cm)")
    plt.ylabel("Petal Width (cm)")
    plt.title('Resultados obtidos ')

    plt.subplot(1,2,2)
    plt.scatter(x2['Petal Length'], x2['Petal Width'], c= colors[y2['Target']], s=40, alpha= 0.8)
    plt.xlabel("Petal Length (cm)")
    plt.ylabel("Petal Width (cm)")
    plt.title('Resultados esperados')

    plt.show()