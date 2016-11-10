import support
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import *
import pandas as pd
from nltk.corpus import stopwords
import string
from sklearn.decomposition import TruncatedSVD, PCA, FactorAnalysis
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize
from sklearn.feature_selection import chi2
mapper = {"math":1,"cs":2,"stat":3,"physics":4}
reversemapper={1:"math",2:"cs",3:"stat",4:"physics"}
classes=4
stoplist = set(stopwords.words('english'))
stoppunc = set(string.punctuation)
stopwords=True

def minmaxnorm(matrix):
    for column in matrix: 
        minimum = (matrix[column]).min()
        maximum = (matrix[column]).max()
        matrix[column] = pd.DataFrame(((matrix[column])-minimum)/(maximum-minimum))
    return matrix
def zscorenorm(matrix):
    for column in matrix:
        if column!="zvalue":
            std = (matrix[column]).std()
            mean = (matrix[column]).mean()
            matrix[column] = pd.DataFrame(((matrix[column])-mean)/(std))
    return matrix

def selectfeatures_pca(n_features,step,X,Y):
	pca = PCA(svd_solver='full') 
	fa = FactorAnalysis()
	n_components = np.arange(0, n_features, step)
	pca_scores, fa_scores = [], []
	for n in n_components:
		pca.n_components = n
		fa.n_components = n
		pca_scores.append(np.mean(cross_val_score(pca, X,Y)))
		fa_scores.append(np.mean(cross_val_score(fa, X,Y)))
		print(pca_scores)
	plt.figure()
	plt.plot(n_components, pca_scores, 'b', label='PCA scores')
	plt.plot(n_components, fa_scores, 'r', label='FA scores')
	plt.xlabel('nb of components')
	plt.ylabel('CV scores')
	plt.legend(loc='lower right')
	plt.title('Feature Selection using PCA and FA')
	plt.savefig('FeatureSelectionPCAnFA.png',format='png')
#sentences, Y, idstrain = support.read_data('train',mapper,tag="")
#sentences_test, ids = support.read_data('test',mapper,tag="")

###############################################################
#create file for lemmatized output

#sentenceslem = (support.lemmatize(sentences))
#support.write_lemma(sentenceslem,idstrain,"train")
#print("Training lemmatization done")
#sentences_testlem = support.lemmatize(sentences_test)
#print("Testing Lemmatization done")
#support.write_lemma(sentences_testlem,ids,"test")

################################################################
#create file which contains pos tags for each word

#sentences = (support.pos(sentences))
#support.write_pos(sentences,idstrain,"train")
#print("Training pos done")
#sentences_test = support.pos(sentences_test)
#print("Testing pos done")
#support.write_pos(sentences_test,ids,"test")

###############################################################
#create features from pos count
'''
sentences, Y, idstrain = support.read_data('train',mapper,tag="_pos")
sentences_test, ids = support.read_data('test',mapper,tag="_pos")

vectorizer = CountVectorizer(lowercase=True,ngram_range=(1,1))
csrmatrix=vectorizer.fit_transform(sentences)
data = pd.DataFrame(csrmatrix.toarray())
data.columns = vectorizer.get_feature_names()
#data = zscorenorm(data)
#data = data.as_matrix()
#pca=PCA()
#data=pd.DataFrame(pca.fit_transform(data))
data.to_csv("train_in_posfeatures.csv")
csrmatrix = vectorizer.transform(sentences_test)
data = pd.DataFrame(csrmatrix.toarray())
data.columns = vectorizer.get_feature_names()
#data = zscorenorm(data)
#data = data.as_matrix()
#data=pd.DataFrame(pca.transform(data))
ids = pd.DataFrame([ids]).transpose()
ids.columns = ["id"]
data = pd.concat([ids,data],axis=1)
data.to_csv("test_in_posfeatures.csv",index=False)
'''

################################################################
#create features from lemmatized output
sentences, Y, idstrain = support.read_data('train',mapper,tag="_lemma")
sentences_test, ids = support.read_data('test',mapper,tag="_lemma")
stop = (set.union(stoppunc,stoplist) if stopwords else stoppunc)

vectorizer = TfidfVectorizer(lowercase=True,min_df=1,stop_words=stop,ngram_range=(1,2))

csrmatrix=vectorizer.fit_transform(sentences)
featname=vectorizer.get_feature_names()
print(len(featname))
print("csr created")
#data = csrmatrix.toarray()
#del (csrmatrix)
#data.columns = vectorizer.get_feature_names()
#print("zscore started")
#data = zscorenorm(data)
#data = data.as_matrix()
#normalize(csrmatrix,axis=0,copy=False)
#print("converting to array")

#data = stats.zscore(data,axis=0,ddof=1)
#print("pca started")
#svd=TruncatedSVD(n_components=20000)
#csrmatrix=pd.DataFrame(svd.fit_transform(csrmatrix))
#print("pca finished")

chi2,pval = chi2(csrmatrix, Y)
featname = pd.DataFrame(featname)
chi2 = pd.DataFrame(chi2)
pval = pd.DataFrame(pval)

data=pd.concat([featname,chi2,pval],axis=1)
data.columns = ["word","chi2","pval"]
data=data.sort_values("pval",axis=0)
#print(data)
data.to_csv("../datasets/train_in_lemmafeatures.csv")
#csrmatrix = vectorizer.transform(sentences_test)
#data = pd.DataFrame(csrmatrix.toarray())
#data.columns = vectorizer.get_feature_names()
#data = zscorenorm(data)
#data = data.as_matrix()
#data=pd.DataFrame(pca.transform(data))
#ids = pd.DataFrame([ids]).transpose()
#ids.columns = ["id"]
#data = pd.concat([ids,data],axis=1)
#data.to_csv("test_in_lemmafeatures.csv",index=False)

#selectfeatures_pca(data.shape[1],2,data,Y)
#csrmatrix=vectorizer.transform(sentences_test)
#data_test = pd.DataFrame(csrmatrix.toarray())
#data_test.columns = vectorizer.get_feature_names()
#idstrain = pd.DataFrame([idstrain])
#idstrain = idstrain.transpose()
#idstrain.columns = ["id"]
#data=pd.concat([idstrain,data],axis=1)
#data.to_csv("../datasets/train_in_posfeatures.csv",index=False)
#ids = pd.DataFrame([ids])
#ids = ids.transpose()
#ids.columns = ["id"]
#data_test = pd.concat([ids,data_test],axis=1)
#data_test.to_csv("../datasets/test_in_posfeatures.csv",index=False)
##############################################################
#Create features with ngrams and bigrams
'''

stop = (set.union(stoppunc,stoplist) if stopwords else stoppunc)

sentences, Y, idstrain = support.read_data('train',mapper,tag="_lemma")
#sentences_test, ids = support.read_data('test',mapper,tag="_lemma")
#sentences,Y = shuffle(sentences,Y,random_state=17)
vectorizer = CountVectorizer(lowercase=True,stop_words=stop,ngram_range=(1,2))
csrmatrix=vectorizer.fit_transform(sentences)
data2 = pd.DataFrame(csrmatrix.toarray())
data2.columns = vectorizer.get_feature_names()
'''
#data = pd.concat([data,data2],axis=1)

#data.columns = vectorizer.get_feature_names()
#idstrain = pd.DataFrame([idstrain])
#idstrain = idstrain.transpose()
#idstrain.columns = ["id"]
#data=pd.concat([idstrain,data],axis=1)
#data.to_csv("../datasets/train_in_lemmafeatures.csv",index=True)
#sentences_test, ids = support.read_data('test',mapper,tag="_lemma")
#csrmatrix=vectorizer.transform(sentences_test)
#data_test = pd.DataFrame(csrmatrix.toarray())
#data_test.columns = vectorizer.get_feature_names()
#data = pd.concat([data,data2],axis=1)
#data_test = pd.concat([data_test,data2_test],axis=1)


#print(data)
#print(data_test)
#data.to_csv("../datasets/train_in_features.csv",index=False)
#data_test.to_csv("../datasets/test_in_features.csv",index=False)


