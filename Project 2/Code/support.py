from itertools import izip
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag


def pos(sentences):
        newsent=[]
        i=0    
        for sents in sentences:

                sents = sent_tokenize(sents)
                sentence=""
                for sent in sents:
                        tokens = word_tokenize(sent)
                        tokens = pos_tag(tokens)
                        for token in tokens:
                                #if(penntown(token[1])!='x'):
                                        word = token[1]
                                        sentence = " ".join((sentence, word))

                newsent.append(sentence.strip().encode('ascii','ignore'))
                print(i)
                #if i>2:
                #       break
                #print(newsent[i])
                i=i+1
        return newsent

def lemmatize(sentences):
	newsent=[]
	i=0	
	for sents in sentences:

                sents = sent_tokenize(sents)
                sentence=""
                for sent in sents:
                        tokens = word_tokenize(sent)
                        tokens = pos_tag(tokens)
                        for token in tokens:
                                #if(penntown(token[1])!='x'):
                                        word = WordNetLemmatizer().lemmatize(token[0],penntown(token[1]))
                                        sentence = " ".join((sentence, word))

                newsent.append(sentence.strip().encode('ascii','ignore'))
		print(i)
		#if i>2:
		#	break
		#print(newsent[i])
		i=i+1
	return newsent

def penntown(tag):
                if tag[0]=="J":
                        return wn.ADJ
                elif tag[0]=="N":
                        return wn.NOUN
                elif tag[0]=="R":
                        return wn.ADV
                elif tag[0]=="V":
                        return 'v'
                else:
                        return 'n'
                return None





def read_data(mode,mapper,lemmat=True,tag=""):
	if mode=="train":	
    		xpath = '../datasets/train_in'+tag+'.csv'
    		ypath = '../datasets/train_out.csv'
		X, Y = [], []
		X=pd.read_csv(xpath)
		Y=pd.read_csv(ypath)
		merged=(pd.merge(X,Y,how='inner',on='id'))
		merged=(merged[merged["category"]!="category"])
		merged["category"]=(merged["category"].map(mapper))
		Y=merged["category"].as_matrix()
		X=merged["abstract"].as_matrix()
		ids=merged["id"].as_matrix()
		return X,Y,ids
	else:
		xpath = '../datasets/test_in'+tag+'.csv'
		X=[]
		X=pd.read_csv(xpath)
		ids=X["id"].as_matrix()
		X=X["abstract"].as_matrix()
		
		return X,ids

def write_data(pred,ids,mapping):
	filepath = '../datasets/test_out.csv'
	data = pd.DataFrame([ids,pred]).transpose()
	data.columns = ["id","category"]
	data["category"] = data["category"].map(mapping)
	data.to_csv(filepath,index=False)

def write_lemma(sents,ids,mode):
	if mode=="train":
		filepath = '../datasets/train_in_lemma.csv'
	else:
		filepath = '../datasets/test_in_lemma.csv'
	data = pd.DataFrame([ids,sents]).transpose()
	data.columns = ["id","abstract"]
	data.to_csv(filepath,index=False)

def write_pos(sents,ids,mode):
        if mode=="train":
                filepath = '../datasets/train_in_pos.csv'
        else:
                filepath = '../datasets/test_in_pos.csv'
        data = pd.DataFrame([ids,sents]).transpose()
        data.columns = ["id","abstract"]
        data.to_csv(filepath,index=False)
