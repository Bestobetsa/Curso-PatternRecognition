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
	p0Num=zeros(numWords)
	p1NUm=zeros(numWords)
	p0Denom=0.0
	p1Denom=0.0
	for i in range(numTrainDocs):
		if trainCategory[i]==1:
			p1NUm+=trainMatrix[i]
			p1Denom+=sum(trainMatrix[i])
		else:
			p0Num+=trainMatrix[i]
			p0Denom+=sum(trainMatrix[i])
	p1Vect=p1NUm/p1Denom #change to log()
	p0Vect=p0Num/p0Denom #change to log()
	return p0Vect,p1Vect,pAbusive
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
			 
