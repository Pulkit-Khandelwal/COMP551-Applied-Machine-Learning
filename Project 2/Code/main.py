import numpy as np
from nltk.corpus import stopwords
import string
import support
import models as md
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
from sklearn.decomposition import PCA, FactorAnalysis

mapper = {"math":1,"cs":2,"stat":3,"physics":4}
reversemapper={1:"math",2:"cs",3:"stat",4:"physics"}
classes=4
sentences, Y, idstrain = support.read_data('train',mapper)
sentences_test, ids = support.read_data('test',mapper)
sentences,Y = shuffle(sentences,Y,random_state=17)

#sentences = (support.lemmatize(sentences))
#support.write_lemma(sentences,idstrain,"train")
#print("Training lemmatization done")
#sentences_test = support.lemmatize(sentences_test)
#print("Testing Lemmatization done")
#support.write_lemma(sentences_test,ids,"test")
stoplist = set(stopwords.words('english'))
stoppunc = set(string.punctuation)
#make this true to generate plots
plot=False
models = []
#Explanation for models respectively 
# Name of the Graph or file, unigram(1) or unigram and bigram(2), convert to lowercase boolean, include stop words boolean, Number of features, run only a single cross validation iteration (for time speedup), boolean for writing test_out.csv, class name for creating the object

#models.append(["MultinomialNB Smoothing=0.5",2,True,True,57000,0,True,"MultinomialNB(alpha=0.5)"])
#models.append(["MultinomialNB Smoothing=1",2,True,True,60000,0,False,"MultinomialNB(alpha=1)"])
#models.append(["MultinomialNB Smoothing=1.5",2,True,True,60000,0,False,"MultinomialNB(alpha=1.5)"])
#models.append(["MultinomialNB Smoothing=2",2,True,True,60000,0,False,"MultinomialNB(alpha=2)"])
#models.append(["MultinomialNB Smoothing=2.5",2,True,True,35000,0,True,"MultinomialNB(alpha=2.5)"])
#models.append(["SVM Linear",2,True,True,100,0,False,"mysvm.svm()"])
#models.append(["KNN Minkowski (p = 3) k=101",2,True,True,3000,1,False,"knn.knn(201,p=4,dist='mi')"])
#models.append(["KNN RFB (Sigma 2) k=101",2,True,True,3000,1,False,"knn.knn(201,sigma=2,dist='rbf')"])
#models.append(["KNN Euclidean k=101",2,True,True,3000,1,False,"knn.knn(201)"])
models.append(["Logistic Regression Lamda=1e-4",2,True,True,57000,0,False,"log_reg.log_reg(epsilon=0.0001,alpha=0.5, max_iterations=1000,lamda=0.125)"])
#models.append(["Neural Network",2,True,True,50000,0,True,"MLPClassifier(solver='lbfgs', random_state=1)"])
#models.append(["RandomForest",2,True,True,500,1,False,"decision_trees.decision_trees(ntrees=10,nfeat_min=2,nfeat_max=5)"])
#models.append(["RandomForest",2,True,True,5000,1,False,"decision_trees.decision_trees(ntrees=5000,nfeat_min=4,nfeat_max=5)"])

for i in range(0,len(models)):
	terrors=[]
	verrors=[]
	feats=[]
	precisions=[]
	f1s=[]
	recalls=[]
	minfeat=models[i][4]
	maxfeat=models[i][4]+1
	step=1000
	for j in range(minfeat,maxfeat,step):
		models[i][4]=j
	
		model = md.models(stoplist,stoppunc,ngram=models[i][1],lowercase=models[i][2],stopwords=models[i][3],nfeats=models[i][4],min_df=models[i][5])
		
		sentences = sentences[:10000]
		Y = Y[:10000]
		model.setpred(sentences_test)
		Terror,Verror,precision,recall,f1 = model.kfoldCV(sentences,Y,5,models[i][7],classes,1)
		print(models[i])
		print("Training Error = ",Terror)
		print("Validation Error = ",Verror)
		print(models[i][4])
		print("Precision",precision)
		print("Recall",recall)
		print("F1",f1)
		terrors.append(Terror)
		verrors.append(Verror)
		precisions.append(precision)
		recalls.append(recall)
		f1s.append(f1)
		feats.append(models[i][4])
		
#generate tes result or not
		if models[i][6]:
			predictions=model.predict(sentences,Y,sentences_test,models[i][7])

			#print(predictions.shape)
			support.write_data(predictions,ids,reversemapper)	
	if plot:
		#validation error plot
		plt.plot(feats, terrors)
		plt.plot(feats,verrors)
		plt.xlabel("Number of Feats")
		plt.ylabel("Error")
		plt.legend(['Training', 'Validation'], loc='upper right')
		plt.title(models[i][0])
		plt.savefig("../Fig/"+models[i][0]+'_feats.png', format='png')
		plt.clf()
		precisions = np.array(precisions)
		recalls = np.array(recalls)
		f1s = np.array(f1s)
		#precision recall plot
		plt.plot(precisions[:,0].tolist(),recalls[:,0].tolist())
		plt.plot(precisions[:,1].tolist(),recalls[:,1].tolist())
		plt.plot(precisions[:,2].tolist(),recalls[:,2].tolist())
		plt.plot(precisions[:,3].tolist(),recalls[:,3].tolist())
		plt.legend(['math',"cs","stat", 'physics'], loc='lower right')
		plt.xlabel("Recall")
		plt.ylabel("Precision")
		plt.title(models[i][0])
		plt.savefig("../Fig/"+models[i][0]+'_precall.png', format='png')
		plt.clf()
		#F1 plot
		plt.plot(feats,f1s[:,0])
		plt.plot(feats,f1s[:,1])
		plt.plot(feats,f1s[:,2])
		plt.plot(feats,f1s[:,3])
		plt.legend(['math',"cs","stat", 'physics'], loc='lower left')
		plt.xlabel("Number of Features")
		plt.ylabel("F1 Score")
		plt.title(models[i][0])
		plt.savefig("../Fig/"+models[i][0]+'_f1.png', format='png')
		plt.clf()
		#write to file
		data=pd.DataFrame([feats,terrors,verrors]).transpose()
		data.columns = ["features","Training Error","Validation Error"]
		precisions=pd.DataFrame(precisions)
		precisions.columns = ["p math","p cs","p stat","p physics"]
		recalls=pd.DataFrame(recalls)
		recalls.columns = ["r math","r cs","r stat","r physics"]
		f1s=pd.DataFrame(f1s)
		f1s.columns = ["f1 math","f1 cs","f1 stat","f1 physics"]
		data=pd.concat([data,precisions,recalls,f1s],axis=1)
		data.to_csv("../Fig/"+models[i][0]+"_feats.csv")

	
