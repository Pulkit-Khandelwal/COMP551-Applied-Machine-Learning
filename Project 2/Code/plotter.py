from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("Final Results_NB.csv",',',None,0) #Read data frame from csv.

uniqalpha = df['Alpha'].unique();
uniqfeats = df['features'].unique();
uniqalpha = uniqalpha[:-1];
uniqfeats = uniqfeats[:-1];
uniqalpha.sort();
uniqfeats.sort();
print uniqalpha;
print uniqfeats;
#make plot of alpha vs pavg,ravg,f1avg
pavg =range(0,len(uniqalpha));
ravg = range(0,len(uniqalpha));
f1avg = range(0,len(uniqalpha));

for i in range(0,len(uniqalpha)):
    pavglist = df[df['Alpha']==uniqalpha[i]]['pavg'];
    pavgarr = pavglist.as_matrix();
    pavg[i] = pavgarr.mean();
    ravglist = df[df['Alpha']==uniqalpha[i]]['ravg'];
    ravgarr = ravglist.as_matrix();
    ravg[i] = ravgarr.mean();
    f1avglist = df[df['Alpha']==uniqalpha[i]]['f1 avg'];
    f1avgarr = f1avglist.as_matrix();
    f1avg[i] = f1avgarr.mean();

pavg_feats =range(0,len(uniqfeats));
ravg_feats = range(0,len(uniqfeats));
f1avg_feats = range(0,len(uniqfeats));

for j in range (0,len(uniqfeats)):
    pavglist_feats = df[df['features']==uniqfeats[j]]['pavg'];
    pavgarr_feats = pavglist_feats.as_matrix();
    pavg_feats[j] = pavgarr_feats.mean();
    ravglist_feats = df[df['features']==uniqfeats[j]]['ravg'];
    ravgarr_feats = ravglist_feats.as_matrix();
    ravg_feats[j] = ravgarr_feats.mean();
    f1avglist_feats = df[df['features']==uniqfeats[j]]['f1 avg'];
    f1avgarr_feats = f1avglist_feats.as_matrix();
    f1avg_feats[j] = f1avgarr_feats.mean();


pavghd,=plt.plot(uniqalpha,pavg,'ro-',label="precision avg");
ravg_hd,=plt.plot(uniqalpha,ravg,'go-',label="recall avg");
f1avg_hd,=plt.plot(uniqalpha,f1avg,'bo-',label="f1 score avg");
plt.xlabel('Compression (C)')
plt.ylabel('Average precision, recall, f1 score')
plt.title('SVM Alpha vs precision,recall,f1 score')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0,handles=[pavghd,ravg_hd,f1avg_hd]);
plt.savefig('svm_c_vs_prf1.png', bbox_inches='tight');
#plt.show()



pavghd,=plt.plot(uniqfeats,pavg_feats,'ro-',label="precision avg");
ravg_hd,=plt.plot(uniqfeats,ravg_feats,'go-',label="recall avg");
f1avg_hd,=plt.plot(uniqfeats,f1avg_feats,'bo-',label="f1 score avg");
plt.xlabel('Number of Features')
plt.ylabel('Average precision, recall, f1 score')
plt.title('SVM Nfeats vs precision,recall,f1 score')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0,handles=[pavghd,ravg_hd,f1avg_hd]);
plt.savefig('svm_feats_vs_prf1.png', bbox_inches='tight');
#plt.show()



