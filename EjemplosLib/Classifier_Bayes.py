#El texto fue ya divido por etiquetas, aui ya se hizo el tratamieto de datos
#hay dos clases (abusivo o no) pero recordemos que laprobalilidad puede
#ser binaria como 1 o 0 
#para usar en terminal con python3 
#>>import Classifier_Bayes
#>>listOPosts,listClasses = bayes.loadDataSet()
#>>myVocabList = bayes.createVocabList(listOPosts)
#>>myVocabList
def loadDataSet():
	postingList=[['my','dog','has','flea','problems','help','please'],
				 ['maybe','not','take','him','to','dog','park','stupid'],
				 ['my','dalmation','is','so','cute','I','love','him'],
				 ['stop','posting','stupid','worthless','garbage'],
				 ['mr','licks','ate','my','steak','how','to','stop','him'],
				 ['quit','buying','worthless','dog','food','stupid']]
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
	for word in vocabList:
		if word in vocabList:
			returnVec[vocabList.index(word)]=1
		else: print("The Word: %s is not my Vocabulary!"%word)
	return returnVec



			 