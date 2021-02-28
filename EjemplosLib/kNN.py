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
#Aqui vamos a usar Matplotlib todo debe ser directo
#en la linea de comando de python una vez creado:
#datingDataMat
#fig=plt.figure()
#ax=fig.add_subplot(111)
#ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
#plt.show()
#ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0*array(datingLabels,dtype=numpy.int32), 15.0*array(datingLabels,dtype=numpy.int32))
#Formula a utilizar para sacar la normalizacion: newValue=(oldValue-min)/(max-min)
#normMat, ranges, minVals = kNN.autoNorm(datingDataMat)
def autoNorm(dataSet):
	minVals=dataSet.min(0)
	maxVals=dataSet.max(0)
	ranges=maxVals-minVals
	normDataSet=zeros(shape(dataSet))
	m=dataSet.shape[0]
	normDataSet=dataSet-tile(minVals,(m,1))
	normDataSet=dataSet/tile(ranges,(m,1))
	return normDataSet,ranges,minVals
#en lasiguiente funcion se evaluara el clasificdor
#0 significa que funciona bien 1.0 que el error es 
#absoluto y siempre tendra errores al clasificar
#kNN.datingClassTest()
def datingClassTest():
	hoRatio=0.10
	datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
	normMat,ranges, minVals=autoNorm(datingDataMat)
	m=normMat.shape[0]
	numTestVecs=int(m*hoRatio)
	errorCount=0
	for i in range(numTestVecs):
		classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
		print("the classifier came back with: {0}, the real answer is: {1}".format(classifierResult, datingLabels[i])) 
		if(classifierResult!=datingLabels[i]): errorCount+=1.0
	print("the total error rate is: %f" % (errorCount/float(numTestVecs)))
#ahora en esta seccion se va a predecir cuanto puede agraddarle una persona
#raw_input le permite al usuario ingresar datos y los regresa 
#kNN.classifyPerson()
def classifyPerson():
	resultList=['not at all','in small does', 'in large does']
	percentTast=float(input("percentage of time spent playing video games?"))
	ffMiles=float(input("frequent flimer miles earned per year?"))
	iceCream=float(input("litters of ice cream consumed per year?"))
	datingDataMat, datingLabels=file2matrix('datingTestSet2.txt')
	normMat,ranges,minVals=autoNorm(datingDataMat)
	inArr=array([ffMiles,percentTast,iceCream])
	classifierResult=classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
	print("You Will probably like this person: ",resultList[classifierResult -1])
#Para este ejemplo haremos un clasificador para HANDWRINTING con kNN
def handwritingClasssTest():
	hwLabels=[]
	trainingFileList=listdir('trainingDigits')
	m=len(trainingFileList)
	trainingMat=zeros((m,1024))
	for i in range(m):
		fileNameStr=trainingFileList[i]
		fileStr=fileNameStr.split('.')[0]
		classNumStr=int(fileStr.split('_')[0])
		hwLabels.append(classNumStr)
		trainingMat[i,:]=img2vector('trainingDigits/%s'%fileNameStr)
	testFileList=listdir('testDigits')
	errorCount=0.0
	mTest=len(testFileList)
	for i in range(mTest):
		fileNameStr=testFileList[i]
		fileStr=fileNameStr.split('.')[0]
		classNumStr=int(fileStr.split('_')[0])
		vectorUnderTest=img2vector('testDigits/%'%fileNameStr)
		classifierResult=classify0(vectorUnderTest,\trainingMat,hwLabels,3)
		print("the classifier came back with: {0}, the real answer is: {1}".format(classifierResult,classNumStr))
		if(classifierResult!=classNumStr):errorCount+=1.0
	print("\n The total number of errors is:{0}".format(errorCount))
	print("\n The total error rate is:{0}".format(errorCount/float(mTest)))
