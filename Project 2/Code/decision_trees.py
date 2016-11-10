import pandas as pd;
import numpy as np;
from generic_fns import *;

class node:
    def __init__(self,feat=-1,test_val=None,results=None,true_branch=None,false_branch=None):
        self.feat=feat;
        self.test_val=test_val;
        self.results=results;
        self.true_branch=true_branch;
        self.false_branch=false_branch;
    
            
class decision_trees():
    def __init__(self,ntrees=10,nfeat_min=2,nfeat_max=5):
    #option variables
    	self.n_tree_sets_in_forest = ntrees
    	self.n_features_min = nfeat_min
    	self.n_features_max = nfeat_max
    
    def printtree(self,tree,indent=''):
        # Is this a leaf node?
        if tree.results!=None:
            print("y0,y1:",tree.results)
        else:
            print(str(tree.feat)+':'+str(tree.test_val)+'? ')
            # Print the branches
            print(indent+'T->')
            self.printtree(tree.true_branch,indent+'  ')
            print(indent+'F->')
            self.printtree(tree.false_branch,indent+'  ')

            
    def count_ys (self,Y):
        y0=len(Y[Y['category']==0].index.tolist());
        y1=len(Y[Y['category']==1].index.tolist());
        return y0,y1;
    
    def r_transform(self,p):
        if(p>0.5):
            r_p=(0.5)*(1+((2*p-1)**(0.5)));
        else:
            r_p=(0.5)*(1-((1-2*p)**(0.5)));
        #print "r_p:",r_p;
        return r_p;
    
    def xlogx(self,x):
        xlogx=x*np.log(x);
        #print "xlogx:",xlogx;
        return xlogx;
    
    def g_calc(self,p):
        r_p = self.r_transform(p);
        g_p = -self.xlogx(r_p)-self.xlogx(1-r_p);
        #print "g_p:",g_p;
        return g_p;  
    
    def return_new_p(self,p):
        if p==0:
            pn=0.00000000001;
        elif p==1:
            pn=0.99999999999;
        else:
            pn=p
        return pn;
    
    def m_entropy_cost(self,Y1,Y2):
        y00,y01=self.count_ys(Y1);
        y10,y11=self.count_ys(Y2);
        y00=float(y00);
        y01=float(y01);
        y10=float(y10);
        y11=float(y11);
        p_1=y01/(y00+y01);
        p_2=y11/(y10+y11);
        p = (y00+y01)/(y00+y01+y10+y11);
        #print "y00 y01 y10 y11:",y00, y01, y10, y11;
        #print "p_1,p_2,p",p_1,p_2,p;
        p_1=self.return_new_p(p_1);
        p_2=self.return_new_p(p_2);
        cost = p*self.g_calc(p_1)+(1-p)*self.g_calc(p_2);
        
        return cost;
        
    def divide_df(self,X,Y,feature,value):
        #returns 2 sets, broken at value.
        #print ("X");
        #print X;
        #print ("Y");
        #print Y;
        X_tmp = X.copy();
        X_tmp['category']=Y[Y.columns[0]];
        X_tmp = X_tmp.sort_values(X_tmp.columns[feature],ascending=True);
        
        X1=X_tmp.loc[X_tmp[X_tmp.columns[feature]]<=value];
        X2=X_tmp.loc[X_tmp[X_tmp.columns[feature]]>value];
        Y1 = X1[['category']];
        Y2 = X2[['category']];
        X1=X1.drop('category',axis=1);
        X2=X2.drop('category',axis=1);
        
        return X1,Y1,X2,Y2;
    
    def grow_tree(self,X,Y):
        #Function:grows a tree of Y w.r.t. X. 
        #Recursive function that grows a tree and returns the tree.
        
        #Return if dataset is empty.
        if(len(X.index)==0):
            return node();
        
        #Variables to track cost.
        cost_lowest=1.0;
        lowest_crit=[];
        lowest_sets=[];
        
        for i in range(0,len(X.columns)):
            #for each feature in df,find all possible splits.
            uqlist = X[X.columns[i]].unique();
            #print "uqlist:",uqlist;
            maxuqlist = max(uqlist);
            #print maxuqlist;
            search_list = np.arange(0,maxuqlist+0.1,0.1);
            for j in search_list:
                #print j;
                X1,Y1,X2,Y2=self.divide_df(X,Y,i,j);
                if(len(Y1)>0 and len(Y2)>0):
                    cost = self.m_entropy_cost(Y1,Y2);
                    #print cost;
                    if(cost<cost_lowest):
                        cost_lowest=cost;
                        lowest_crit=[i,j];
                        lowest_sets=[X1,Y1,X2,Y2];
        
        if(cost_lowest<1.0):
            branch_true = self.grow_tree(lowest_sets[0],lowest_sets[1]);
            branch_false =self.grow_tree(lowest_sets[2],lowest_sets[3]);
            return node(feat=X.columns[lowest_crit[0]],test_val=lowest_crit[1],true_branch=branch_true,false_branch=branch_false);
        else:
            y0,y1=self.count_ys(Y);
            return node(results=[y0,y1]);
        
    def prune_tree(self,tree,X,Y):
        return tree;

    def make_tree(self,X,Y):
        #Function:makes a classification tree of Y w.r.t. X
        #Calls grow_tree that grows tree to data, and then prunes the tree
        #A random number of features are chosen between n_features_min and n_features_max
        n_features = np.random.randint(self.n_features_min,self.n_features_max);
        n_features_total = len(X.columns);
        index = [0]*(n_features);
        i=0;
        while (i<n_features):
            tmp_idx = np.random.randint(0,n_features_total);
            if((tmp_idx in index)==0):
                index[i]=tmp_idx;
                i=i+1;
     
        X_tmp = X.copy();
        X_tmp = X_tmp[X_tmp.columns[index]];
        Y_tmp = Y.copy();
        Y_tmp = Y_tmp[['category']];
        
        #print "X:"
        #print X_tmp
        #print "Y:"
        #print Y_tmp
        
        tree = self.grow_tree(X_tmp,Y_tmp);
        tree = self.prune_tree(tree,X_tmp,Y_tmp);
        
        return tree;
             
                
    
    def make_trees(self,X_list,Y_list,cat_uniq):
        #Function:Returns N number of trees for N categories in Y.
        #args
        #------
        #X_list - Input features for each category.
        #Y_list - List of Ys for each category.
        
        N = len(cat_uniq);
        tree_list = range(0,N);
        #make a list of trees. one for each category
        for i in range(0,N):
			tree_list[i]=self.make_tree(X_list[i],Y_list[i]);
			#print "tree list",i,"done";
			#self.printtree(tree_list[i]);
    
        return tree_list;
    
    def fit(self,X,Y):
		print "start fitting"
		df_X = pd.DataFrame(np.copy(X))
		df_Y = pd.DataFrame(np.copy(Y))
		df_Y.columns = ["category"]
		#print df_Y;
		cat_uniq=(df_Y['category'].unique())
		#print cat_uniq;
		N=len(cat_uniq)
		self.cat_uniq = cat_uniq
		#for make new Y's for each category.
		Y_list = range(0,N)
		Y_list,X_list = return_out_tree_lists(N,df_X,df_Y,cat_uniq)
		forest=range(0,self.n_tree_sets_in_forest);
		for i in range(0,self.n_tree_sets_in_forest):
			forest[i]=self.make_trees(X_list,Y_list,cat_uniq);
			print "tree",i,"done";
            
		self.forest=forest;
    
    def predict_tree(self,tree,X):
        if tree.results!=None:
            #print tree.results;
            if(tree.results[0]>tree.results[1]):
                return 0;
            else:
                return 1;
        else:
            value=X.iloc[0][tree.feat];
            
            #print tree.feat;
            #print value;
            #print str(tree.feat);
            #print X;
            
            branch=None;
            if value<=tree.test_val: 
                nb=tree.true_branch;
            else: 
                nb=tree.false_branch;
                
        return self.predict_tree(nb,X);
    
    
    def predict_set(self,X,tree_list):
        n_trees = len(tree_list);
        predictset = range(0,n_trees);
        
        for i in range(0,n_trees):
            predictset[i]=self.predict_tree(tree_list[i],X);
        
        return predictset;
    
    def prediction_forest_obs(self,forest,X_obs,cat_uniq):
        n_tree_sets = len(forest);
        n_cat = len(cat_uniq);
        
        prediction = range(0,len(forest));
        
        for i in range(0,n_tree_sets):
            prediction[i]=self.predict_set(X_obs,forest[i]);
        
        #print prediction;
        final_pred_list=[sum(i) for i in zip(*prediction)];
        pred_category = final_pred_list.index(max(final_pred_list))+1;
        #print "final_pred_list:",final_pred_list;
        return pred_category;

    def predict(self,X):
		print "start prediction";
		X_tmp = pd.DataFrame(np.copy(X));
		X_df = pd.DataFrame(np.copy(X));
		cat_uniq = self.cat_uniq
		#X_tmp = X.copy();
		X_tmp['category']=0;
		nrows = len(X_tmp.index);
		for i in range(0,nrows):
			X_tmp['category'].iloc[[i]] = self.prediction_forest_obs(self.forest,X_df.iloc[[i]],cat_uniq);
		
		predictions = X_tmp[['category']];
		pred2 = predictions.as_matrix();
		return pred2;
        
        

#df_X = pd.read_csv("in.csv",",",None,0);
#df_Y = pd.read_csv("out.csv",",",None,0);
#cat_uniq=(df_Y['category'].unique());
#N=len(cat_uniq);
#for make new Y's for each category.
#Y_list = range(0,N);
#Y_list,X_list = return_out_tree_lists(N,df_X,df_Y,cat_uniq);

#dt = decision_trees();

#forest = dt.make_forests(X_list,Y_list,cat_uniq);
#predictions = dt.prediction_forest(forest,df_X,cat_uniq);
#print predictions;






