import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import *
from sklearn.naive_bayes import *
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import normalize
import svm as mysvm
import knn
import log_reg
import decision_trees

class featureextractor():

        def extract_features(self,sents,stoplist,stoppunc,ngram=1,vocab=None,lowercase=True,stopwords=True,nfeats=100,min_df=1,lemmatize=False):


                stop = (set.union(stoppunc,stoplist) if stopwords else stoppunc)
                #sents = self.lemmatize(sents) if lemmatize==True else sents
                #print("stop->",stop)
                vectorizer = TfidfVectorizer(lowercase=lowercase,stop_words=stop,vocabulary=vocab,min_df=min_df,max_features=None,ngram_range=(1,ngram))
                csrmatrix = vectorizer.fit_transform(sents)
                self.X = csrmatrix
                self.mapping = vectorizer.get_feature_names()
                self.vocab = vectorizer.vocabulary_
                self.vectorizer = vectorizer
                return csrmatrix.toarray()

        def get_features(self,sents):
                csrmatrix = self.vectorizer.transform(sents)
                return csrmatrix.toarray()

class models():
	def __init__(self,stoplist,stoppunc,ngram,lowercase,stopwords,nfeats,min_df):
		self.stoplist=stoplist
		self.stoppunc=stoppunc
		self.ngram=ngram
		self.lowercase=lowercase
		self.stopwords=stopwords
		self.nfeats=nfeats
		self.min_df=min_df
		vocab=pd.read_csv("../datasets/train_in_lemmafeatures.csv",nrows=nfeats+1)
		self.vocab=np.ravel(pd.DataFrame(vocab["word"]).as_matrix())
		#trainextra = pd.read_csv("../datasets/train_in_posfeatures.csv")
		#del trainextra["id"]
		#self.trainextra = trainextra.as_matrix()
		#testextra = pd.read_csv("../datasets/test_in_posfeatures.csv")
		#del testextra["id"]		
		#self.testextra = testextra.as_matrix()
	
	def kfoldCV(self,sentences,Y,k,modelname,classes,loops):
		datasets = np.array_split(sentences,k,axis=0)
		testsets = np.array_split(Y,k,axis=0) 
		#extradata = np.array_split(self.trainextra,k,axis=0)
		CM_model=np.zeros((classes,classes)).astype(float)
		model = eval(modelname)
		self.modelname = modelname
		error=float(0)
        	nooft=0
		noofv=0
		FE = featureextractor()
		for i in range(0,k):
#generate datasets
			learnon = np.concatenate((datasets[i % k], datasets[(i + 1) % k], datasets[(i + 2) % k], datasets[(i + 3) % k]),
                             axis=0)
			learny = np.concatenate((testsets[i % k], testsets[(i + 1) % k], testsets[(i + 2) % k], testsets[(i + 3) % k]),
                            axis=0)
			#extrax = np.concatenate((extradata[i % k], extradata[(i + 1) % k], extradata[(i + 2) % k], extradata[(i + 3) % k]),
                           # axis=0)
			teston = datasets[(i + 4) % k]
                	testy = testsets[(i + 4) % k]
			#extratest = extradata[(i+4)%k]
#			print(learnon)
#extract features and learn model			
			
			learnon = FE.extract_features(learnon,self.stoplist,self.stoppunc,ngram=self.ngram,lowercase=self.lowercase,stopwords=self.stopwords,nfeats=self.nfeats,min_df=self.min_df,vocab=self.vocab)
			#learnon=np.concatenate((learnon,extrax),axis=1) 
#               	print(learnon)
			teston = FE.get_features(teston)
			normalize(learnon,axis=0,copy=False)
			normalize(teston,axis=0,copy=False) 
			#teston=np.concatenate((teston,extratest),axis=1)
			model.fit(learnon,learny)
			#mysvm=svm.svm()
			#mysvm.fit(learnon,learny)
			#print(model.n_support_)
			#print("examples",learnon.shape[0])
#make predictions
			model_pred=model.predict(teston)
			model_trainpred = model.predict(learnon)
			for m in range(0,testy.shape[0]):
                                CM_model[testy[m]-1][model_pred[m]-1]=CM_model[testy[m]-1][model_pred[m]-1]+1
				noofv = noofv+1	
			for m in range(0,learny.shape[0]):
                        	if(learny[m]!=model_trainpred[m]):
					error = error+1
                        	nooft=nooft+1
			if loops==1:
				break
			print("Validation Stage: ",k)
#calculate errors
		recall=[]
		precision=[]
		f1=[]
		for i in range(0,classes):
			precision.append(float(CM_model[i][i])/float(np.sum(CM_model[:,i])))
			recall.append(float(CM_model[i][i])/float(np.sum(CM_model[i])))
			f1.append(2*precision[i]*recall[i]/(precision[i]+recall[i]))
		error_valid = 1-np.trace(CM_model)/noofv
		error_train = error/nooft
		self.trainingError = error_train
		self.validationError = error_valid
		self.CM = CM_model
		return error_train,error_valid,precision,recall,f1
	def setpred(self,XP):
		self.XP=XP
	def predict(self,sentences,Y,XP,modelname):
		FE = featureextractor()
		X = FE.extract_features(sentences,self.stoplist,self.stoppunc,ngram=self.ngram,lowercase=self.lowercase,stopwords=self.stopwords,nfeats=self.nfeats,min_df=self.min_df,vocab=self.vocab)
		#print(FE.mapping)
		model = eval(modelname)
		model.fit(X,Y)
		
		#print(model.coef_.shape)
		XP = FE.get_features(XP)
		return model.predict(XP)
