#El programa es un ejemplo del libro Machine Learning in Action
#primera parte :Importacion de datos en python 
from numpy import *
import operator
#import matplotlib.pyplot as plt
#la funcion crea variables en este caso goup y label
#cada uno tendra 2 atributos uno es group y el otro es label
#Para usar la funcion se pone en linea de python
#>>>group,labels=kNN.createDataSet()
def createDataSet():
	group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels = ['A','A','B','B']
	return group, labels
#kNN classification algorithm
#Se calculara la distancia de X al punto siguiente
#Se ordenan por distancia (en modo creciente)
#para inX se e agregan los k puntos mas cernanos
#busca la mayor coincidencia en la clase
#regresa como resultado la case conmayor coincidencia como prediccion de inX 
#Para usar la funcion:
#>>>kNN.classify0([n,n], group, labels, k) donde n es el par de numeros a clasificar
#NOTA: se usa la funcion anterior
def classify0(inX, dataSet, labels, k):
	dataSetSize= dataSet.shape[0]
	diffMat= tile(inX,(dataSetSize,1))-dataSet
	sqDiffMat= diffMat**2
	sqDistances= sqDiffMat.sum(axis=1)
	distances=sqDistances**0.5
	sortedDistIndices= distances.argsort()
	classCount={}
	for i in range(k):
		voteIlabel= labels[sortedDistIndices[i]]
		classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
	sortedClassCount= sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]
#abre un rachivo con n datos para hacer una clasificacion
#todo con respecto a label que se construye de la siguiente manera:
#>>>datingDataMat,datingLabels = kNN.file2matrix('datingTestSet.txt')
def file2matrix(filename):
	fr=open(filename)
	numberOfLines=len(fr.readlines())
	returnMat=zeros((numberOfLines,3))
	classLabelVector=[]
	fr=open(filename)
	index=0
	for line in fr.readlines():
		line=line.strip()
		listFromLine= line.split('\t')
		returnMat[index,:]=listFromLine[0:3]
		classLabelVector.append(listFromLine[-1])
		index+=1
	return returnMat,classLabelVector

