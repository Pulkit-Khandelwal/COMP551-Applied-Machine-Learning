def map_category(df,cat_is):
        #Function: Returns a dataframe after mapping categories as follows
        #if category of each row is cat_is, it is mapped to 1, else 0
        ilist1= df[df['category']==cat_is].index.tolist();
        ilist0= df[df['category']!=cat_is].index.tolist();
        df['category'].iloc[ilist1] = 1;
        df['category'].iloc[ilist0] = 0;
        return df;
def return_out_tree_lists(N,X,Y,cat_uniq):
        #Function: Returns a list of output dataframes for each category
        # N - number of unique categories
        # Y - ouput dataframe.
        Y_list = range(0,N);
        X_list = range(0,N);
        
        for i in range(0,N):
            cat_is = cat_uniq[i];
            Y_tmp=Y.copy();
            Y_tmp=map_category(Y_tmp,cat_is);
            Y_list[i]=Y_tmp;
            X_list[i]=X.copy();
        return Y_list,X_list;