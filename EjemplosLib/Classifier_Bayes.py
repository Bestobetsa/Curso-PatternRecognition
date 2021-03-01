#El texto fue ya divido por etiquetas, aui ya se hizo el tratamieto de datos
#hay dos clases (abusivo o no) pero recordemos que laprobalilidad puede
#ser binaria como 1 o 0 
#para usar en terminal con python3 
#>>import Classifier_Bayes
#>>listOPosts,listClasses = bayes.loadDataSet()
#>>myVocabList = bayes.createVocabList(listOPosts)
#>>myVocabList
from numpy import *
import operator
def loadDataSet():
	postingList=[['my', 'dog', 'has', 'flea', \
				  'problems', 'help', 'please'],
				 ['maybe', 'not', 'take', 'him', \
				  'to', 'dog', 'park', 'stupid'],
				 ['my', 'dalmation', 'is', 'so', 'cute', \
				  'I', 'love', 'him'],
				 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
				 ['mr', 'licks', 'ate', 'my', 'steak', 'how',\
				  'to', 'stop', 'him'],
				 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
	classVec=[0,1,0,1,0,1]	#1 is abusive 0 not
	return postingList,classVec
#Ahora haremos la creacion de un datasetvacio y la union de dos
#para poder trabajar estas como si fueran los conjuntos de probabilidad
def createVocabList(dataSet):
	vocabSet=set([])
	for document in dataSet:
		vocabSet=vocabSet | set(document)
	return list(vocabSet)
def setOfWords2Vect(vocabList,inputSet):
	returnVec=[0]*len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)]=1
		else: 
			print("The Word: {0} is not my Vocabulary!".format(word))
	return returnVec
#ahora vamos a hacer el entranamiento
def trainNB0(trainMatrix,trainCategory):
	numTrainDocs=len(trainMatrix)
	numWords=len(trainMatrix[0])
	pAbusive=sum(trainCategory)/float(numTrainDocs)
	p0Num = ones(numWords)
	p1NUm=	ones(numWords)
	p0Denom=2.0
	p1Denom=2.0
	for i in range(numTrainDocs):
		if trainCategory[i]==1:
			p1NUm+=trainMatrix[i]
			p1Denom+=sum(trainMatrix[i])
		else:
			p0Num+=trainMatrix[i]
			p0Denom+=sum(trainMatrix[i])
	p1Vect=log(p1NUm/p1Denom) #change to log()
	p0Vect=log(p0Num/p0Denom) #change to log()
	return p0Vect,p1Vect,pAbusive
#hacer la diferencia entre clases Abusiva "1"  NO "0"
def classifyNB(vect2Classify,p0Vec,p1Vec,pClass1):
	p1=sum(vect2Classify*p1Vec)+log(pClass1)
	p0=sum(vect2Classify*p0Vec)+log(1.0-pClass1)
	if p1>p0:
		return 1
	else:
		return 0
#probar la clasificacion NB
def testingNB():
	listOPosts,listClasses=loadDataSet()
	myVocabList=createVocabList(listOPosts)
	trainMat=[]
	for postinDoc in listOPosts:
		trainMat.append(setOfWords2Vect(myVocabList,postinDoc))
	p0V,p1V,pAb=trainNB0(array(trainMat),array(listClasses))
	testEntry=['love','my','dalmation']
	thisDoc=array(setOfWords2Vect(myVocabList,testEntry))
	print(testEntry,"Classified as:",classifyNB(thisDoc,p0V,p1V,pAb))
	testEntry=['stupid','garbage']
	thisDoc=array(setOfWords2Vect(myVocabList,testEntry))
	print(testEntry,"Classified as:",classifyNB(thisDoc,p0V,p1V,pAb))
#vamos a darle una mejora a setOFWords2Vect
#ahora iremos incrementando el vector word
def bagOfWords2VectMN(vocabList,inputSet):
	returnVec=[0]*len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)]+=1
	return returnVec


#para probar todo en a linea de comandos de Python3 o Python2
#from numpy import *
#import Classifier_Bayes
#listOPosts,listClasses = Classifier_Bayes.loadDataSet()
#myVocabList = Classifier_Bayes.createVocabList(listOPosts)
#myVocabList
#Classifier_Bayes.setOfWords2Vect(myVocabList, listOPosts[0])
#trainMat=[]
#for postinDoc in listOPosts: 
#...
#	trainMat.append(Classifier_Bayes.setOfWords2Vect(myVocabList, postinDoc))
#p0V,p1V,pAb=Classifier_Bayes.trainNB0(trainMat,listClasses)
#Classifier_Bayes.testingNB()
#Para capturar texto y separarlo utilizaremos tambien expresiones reguares
#ya que puntuacion "." se consdera en la palabra
#import re
#mySent='This book is the best book on Python or M.L. I have ever laid eyes upon'
#con la siguiente linea eliminamos los string que no sean mayor que 0
#[tok for tok in listOfTokens if len(tok) > 0]
#ya que queremos que todo sea parejo ya sea Mayusculas o MInusculas
#debemos utilizar las funciones (.lowe()) o (.upper())
#[tok.lower() for tok in listOfTokens if len(tok) > 0]
#regEx=re.compile('\\W*')
#listOfTokens=regEx.split(mySent)
#emailText = open('email/ham/6.txt').read()
#listOfTokens=regEx.split(emailText)
#el archivo 6.txt se encuentra dentro de ham en la capeta email (email.zip)
