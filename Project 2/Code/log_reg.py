import pandas as pd
import numpy as np

#j column is the

class log_reg:


    def __init__(self,epsilon,alpha, max_iterations,lamda):
        self.err_matrix = np.zeros([2, 2])
        self.epsilon = epsilon
        self.alpha = alpha
        self.max_iterations= max_iterations
	self.lamda=lamda
#wk has to be a vector = to number of features, keeps updating



    def logistic (self,w,x): #w is a vector, x is a vector
	dot_product = np.sum(np.dot(w.transpose(),x))
	#print(dot_product)
        neg_dot = -dot_product
        result = 1/(1+np.exp(neg_dot))
        return result


    def curr_gradient(self,x,y,w): #y is a single value, x is a single row in the matrix, w is current vec of weights
        x=np.array([x]).transpose()
	log = self.logistic(w,x)
        grad = x*(y-log)
        return grad



    def fit(self,x,y): #returns a vector of weights
	#handle multiclass
	y1=pd.DataFrame(np.copy(y))
	y1.columns=["Y"]
	y2=y1.copy()
	y3=y1.copy()
	y4=y1.copy()
	y1[y1["Y"]!=1]=0
	y2[y2["Y"]!=2]=0
	y3[y3["Y"]!=3]=0
	y4[y4["Y"]!=4]=0
	y2[y2["Y"]==2]=1
	y3[y3["Y"]==3]=1
	y4[y4["Y"]==4]=1
	y1 = (y1.as_matrix())
	y2 = (y2.as_matrix())
	y3 = (y3.as_matrix())
	y4 = (y4.as_matrix())
	temp = np.ones((x.shape[0], x.shape[1]+1))
	temp[:, 1:] = x
	x=temp
	nrow_x = x.shape[0]  #number rows of x
        ncol_x = x.shape[1]
	self.w1=self.findw(ncol_x,nrow_x,x,y1)
	self.w2=self.findw(ncol_x,nrow_x,x,y2)
	self.w3=self.findw(ncol_x,nrow_x,x,y3)
	self.w4=self.findw(ncol_x,nrow_x,x,y4)	

    def findw(self,ncol_x,nrow_x,x,y):
        prev_w = np.random.rand(ncol_x,1) #initalize to random numbers
        #prev_w = prev_w
        curr_w = prev_w #producing a double array for some reason
        #curr_w = curr_w[:,0]

        it_count = 0
	currerror=1
	preverror=0
        while abs(currerror-preverror)>=self.epsilon: #and it_count<self.max_iterations:
            gsum = np.zeros((ncol_x,1))
            prev_w = curr_w
	    preverror = np.sum(np.square(x.dot(prev_w)-y))/x.shape[0]
	    #print(prev_w)
	    neg_dot=-1*x.dot(prev_w)
	    log=1/(1+np.exp(neg_dot))
	    grad=x.transpose().dot(y-log)
            #for i in range(0,nrow_x):
            #    gsum = gsum+self.curr_gradient(x[i,:],np.sum(y[i]),prev_w)
	    #print(it_count)
	    it_count +=1
            self.alpha = float(1)/(float(it_count)+float(1))
            curr_w = prev_w + self.alpha*grad + 2*self.lamda*prev_w
            currerror = np.sum(np.square(x.dot(curr_w)-y))/x.shape[0]
	    #print("c",curr_w)
	return curr_w

    def predict(self,x):
	temp = np.ones((x.shape[0], x.shape[1]+1))
	temp[:, 1:] = x
	x=temp
	w1=self.w1
	w2=self.w2
	w3=self.w3
	w4=self.w4
        pred = list()
	p1=1/(1+np.exp(-1*x.dot(w1)))
        p2=1/(1+np.exp(-1*x.dot(w2)))
	p3=1/(1+np.exp(-1*x.dot(w3)))
	p4=1/(1+np.exp(-1*x.dot(w4)))
	pred=1+np.argmax(np.concatenate((p1,p2,p3,p4),axis=1),axis=1)
	#for i in range (0,x.shape[0]):
	#    z=np.array([x[i,:]]).transpose()
        #    prediction = [self.logistic(w1,z),self.logistic(w2,z),self.logistic(w3,z),self.logistic(w4,z)]
        #    pred.append(np.argmax(prediction)+1)
        #print((pred))
	return pred

    def error(self,x,y,w): #y is a value, x and w are vectors
        prediction = self.logistic(w,x)
        if(prediction>0.5):prediction = 1
        else: prediction = 0
        self.err_matrix[y,prediction]+=1 #updates the matrix of cumulative errors
        err = np.abs(y-prediction)
        return err #either 0 if model correct, 1 otherwise

#y is vector of 1s and 0s
    def nfold_validation(self):

        x = self.x
        y = self.y


        nrow_x = x.shape[0]
        ncol_x = x.shape[1]

        sum_err = 0

        for i in range (0,nrow_x):

            test_set = x[i, :]
            test_val = y[i,0]


            new_x = np.delete(x, (i), axis=0)
            new_y = np.delete(y,(i), axis=0)

            weights = self.learn(new_x,new_y)

            error = self.error(test_set,test_val,weights)
            sum_err = sum_err+error #keep cumulating the errors

        return sum_err/nrow_x #average error of the model










