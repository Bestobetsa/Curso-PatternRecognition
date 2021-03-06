#carga de DataSet()
#creacion de vectores de X y W(coeficientes)
import numpy as np
from numpy import *
import operator
#import matplotlib.pyplot as plt
def loadDataSet():
	dataMat=[]
	labelMat=[]
	fr=open('testSet.txt')
	for line in fr.readlines():
		lineArr=line.strip().split()
		dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
		labelMat.append(int(lineArr[2]))
	return dataMat,labelMat
#calculo de funcion sigmoid
def sigmoid(inX):
	return 1.0/(1+exp(-inX))
#calculo del gradiente Ascendente
def gradAscent(dataMatIn,classLabels):
	dataMatrix=np.mat(dataMatIn)
	labelMat=np.mat(classLabels).transpose()
	m,n=shape(dataMatrix)
	alpha=0.001
	maxCycles=500
	weights=ones((n,1))
	for k in range(maxCycles):
		h=sigmoid(dataMatrix*weights)
		error=(labelMat-h)
		weights=weights+alpha*dataMatrix.transpose()*error
	return weights
#visualizacion de datos en un plano x,y
#def plotBestFit(wei):
#	weights=wei.getA()
#	dataMat,labelMat=loadDataSet()
#	dataArr=array(dataMat)
#	n=shape(dataMat)[0]
#	xcord1=[];ycord1=[]
#	xcord2=[];ycord2=[]
#	for i in range(n):
#		if int(labelMat[i])==1:
#			xcord1.append(dataArr[i,1])
#			ycord1.append(dataArr[i,2])
#		else:
#			xcord2.append(dataArr[i,1])
#			ycord2.append(dataArr[i,2])
#	fig=plt.figure()
#	ax=fig.add_subplot(111)
#	ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
#	ax.scatter(xcord2, ycord2, s=30, c='green')
#	x = arange(-3.0, 3.0, 0.1)
#	y = (-weights[0]-weights[1]*x)/weights[2]
#	ax.plot(x,y)
#	plt.xlabel('X1')
#	plt.ylabe('X2')
#	plt.show()
#Gradiente ascendente estocastico
def stocGradAscent0(dataMatrix,classLabels):
	m,n=shape(dataMatrix)
	alpha=0.01
	weights=ones(n)
	for i in range(m):
		h=sigmoid(sum(dataMatrix[i]*weights))
		error=classLabels[i]-h
		weights=weights+alpha*error*dataMatrix[i]
		return weights
#se cambia el alpha por cada iteracion
#se actualiza el vector por random selection
def stocGradAscent1(dataMatrix,classLabels,numIter=150):
	m,n=shape(dataMatrix)
	weights=ones(n)
	for j in range(numIter):
		dataIndex=range(m)
		for i in range(m):
			alpha=4/(1.0+j+i)+0.01
			randIndex=int(random.uniform(0,len(dataIndex)))
			h = sigmoid(sum(dataMatrix[randIndex]*weights))
			error = classLabels[randIndex] - h
			weights = weights + alpha * error * dataMatrix[randIndex]
			del(dataIndex[randIndex])
	return weights
#clasificacion en accion
def classifyVector(inX, weights):
	prob = sigmoid(sum(inX*weights))
	if prob > 0.5: 
		return 1.0
	else: 
		return 0.0

def colicTest():
	frTrain = open('horseColicTraining.txt')
	frTest = open('horseColicTest.txt')
	trainingSet = []
	trainingLabels = []
	for line in frTrain.readlines():
		currLine = line.strip().split('\t')
		lineArr =[]
		for i in range(21):
			lineArr.append(float(currLine[i]))
		trainingSet.append(lineArr)
		trainingLabels.append(float(currLine[21]))
	trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
	errorCount = 0; numTestVec = 0.0
	for line in frTest.readlines():
		numTestVec += 1.0
		currLine = line.strip().split('\t')
		lineArr =[]
		for i in range(21):
			lineArr.append(float(currLine[i]))
		if int(classifyVector(array(lineArr), trainWeights))!=int(currLine[21]):
			errorCount += 1
	errorRate = (float(errorCount)/numTestVec)
	print ("the error rate of this test is: {0}".format(errorRate))
	return errorRate

def multiTest():
	numTests = 10
	errorSum=0.0
	for k in range(numTests):
		errorSum += colicTest()
	print "after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests))
#para usarlo por linea de comando de terminal se ingresa
#dataArr,labelMat=LogiR.loadDataSet()
#LogiR.gradAscent(dataArr,labelMat)
#dataArr,labelMat=LogiR.loadDataSet()
#weights=LogiR.stocGradAscent0(array(dataArr),labelMat)
#weights=LogiR.stocGradAscent1(array(dataArr),labelMat)
#LogiR.multiTest()
#LogiR.plotBestFit(weights)
#LogiR.plotBestFit(weights.getA())
