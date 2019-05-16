from sklearn.datasets import load_boston
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from copy import deepcopy
from matplotlib import pyplot as plt 
import pandas as pd
import numpy as np
import random
def targetClassificador(data, target):
    tamanho = len(data)
    t = target
    t.sort()
    baixo =  t[tamanho//3]
    medio =  t[2*(tamanho//3)]
    for i in range(len(target)):
        if target[i] <= baixo:
            target[i] = 0
        elif target[i] <= medio:
            target[i] = 1
        elif target[i] >= medio:
            target[i] = 2
    return target

boston = load_boston()
bost = pd.DataFrame(boston.data)
bost.columns = boston.feature_names
bost['TARGET'] = targetClassificador(bost,boston['target'])
X = bost.drop('TARGET', axis= 1)
Y = bost['TARGET']
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 5)

modelo = KMeans(n_clusters=3, n_jobs = 4, random_state=21)
modelo.fit(X)
Y_pred = np.choose(modelo.labels_, [1, 0, 2]).astype(np.int64)

plt.figure(figsize=(12,3))
colors = np.array(['r','g','b'])
plt.subplot(1, 2, 1)
t = [round(i) for i in bost['TARGET']]
t2 =[round(i) for i in Y_pred]
plt.scatter(X['RM'], X['AGE'], c=colors[t], s=40, alpha=0.8)
plt.title('Resultados esperados')
plt.xlabel("RM")
plt.ylabel('AGE')
plt.subplot(1,2,2)
plt.scatter(X['RM'], X['AGE'], c=colors[t2], s=40, alpha=0.8)
plt.xlabel("RM")
plt.ylabel('AGE')
plt.title('Resultados obtidos')
plt.scatter(modelo.cluster_centers_[:,5] , modelo.cluster_centers_[:,6], marker='*', s=200, c='black')
plt.show()

'''x = pd.DataFrame(teste_global, columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO'
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
'''
    

    