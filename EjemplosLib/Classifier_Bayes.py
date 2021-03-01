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
#Vamos a hacer funcion de Parseo y de testeo
def textParse(bigString):
	import re
	listOfTokens=re.split(r'\W*', bigString)
	return [tok.lowe() for tok in listOfTokens if len(tok)>2]
#Classifier_Bayes.spamTest()
def spamTest():
	docList=[]
	classList=[]
	fullText=[]
	for i in range(1,26):
		wordList=textParse(open('email/spam/%d.txt' % i).read())
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(1)
		wordList=textParse(open('email/ham/%d.txt' % i).read())
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(0)
	vocabList=createVocabList(docList)
	trainingSet=range(50)
	testSet=[]
	for i in range(10):
		randIndex=int(random.uniform(0,len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del(trainingSet[randIndex])
	trainMat=[]
	trainClasses=[]
	for docIndex in trainingSet:
		trainMat.append(setOfWords2Vect(vocabList,docList[docIndex]))
	p0V,p1V,pSpam=trainNB0(array(trainMat),array(trainClasses))
	errorCount=0
	for docIndex in testSet:
		wordVector=setOfWords2Vect(vocabList,docList[docIndex])
		if classifyNB(array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
			errorCount+=1
	print("The error rate is:",float((errorCount)/len(testSet)))	
def calcMostFreq(vocabList,fullText):
import operator
freqDict = {}
for token in vocabList:
	freqDict[token]=fullText.count(token)
sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1),\
reverse=True)
return sortedFreq[:30]
def localWords(feed1,feed0):
	import feedparser
	docList=[]; classList = []; fullText =[]
	minLen = min(len(feed1['entries']),len(feed0['entries']))
	for i in range(minLen):
		wordList = textParse(feed1['entries'][i]['summary'])
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(1)
		wordList = textParse(feed0['entries'][i]['summary'])
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(0)
	vocabList = createVocabList(docList)
	top30Words = calcMostFreq(vocabList,fullText)
	for pairW in top30Words:
		if pairW[0] in vocabList: vocabList.remove(pairW[0])
	trainingSet = range(2*minLen); testSet=[]
	for i in range(20):
		randIndex = int(random.uniform(0,len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del(trainingSet[randIndex])
	trainMat=[]; trainClasses = []
	for docIndex in trainingSet:
		trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
		trainClasses.append(classList[docIndex])
	p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
	errorCount = 0
	for docIndex in testSet:
		wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
		if classifyNB(array(wordVector),p0V,p1V,pSpam) != \
		   classList[docIndex]:
		   errorCount += 1
print ("the error rate is:",float(errorCount)/len(testSet))
return vocabList,p0V,p1V
#Classifier_Bayes.getTopWords(ny,sf)
def getTopWords(ny,sf):
	import operator
	vocabList,p0V,p1V=localWords(ny,sf)
	topNY=[]; topSF=[]
	for i in range(len(p0V)):
		if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
		if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))
	sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
	print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**") 
	for item in sortedSF:
		print item[0]
	sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
	print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY **") 
	for item in sortedNY:
		print item[0]

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
#ny=feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
#sf=feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
#vocabList,pSF,pNY=Classifier_Bayes.localWords(ny,sf)
#vocabList,pSF,pNY=Classifier_Bayes.localWords(ny,sf)
