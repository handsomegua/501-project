import requests
import json
import numpy
import pandas as pd
import seaborn as sns
import nltk
from sklearn import preprocessing
import sklearn
import re  
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import os
from sklearn.model_selection import train_test_split
import random as rd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz 
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem.porter import PorterStemmer
import string 
import pydotplus
import six
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC

# Text data 
champion_data = pd.read_csv('champion_infor_501.csv',encoding='latin1')
champion_data_text = champion_data[['name','tags','blurb']]

STEMMER=PorterStemmer()
def MY_STEMMER(str_input):
    words = re.sub(r"[^A-Za-z\-]", " ", str_input).lower().split()
    words = [STEMMER.stem(word) for word in words]
    return words
MyVect_TFIDF=CountVectorizer(input='content',
                        analyzer = 'word',
                        stop_words='english',
                        lowercase = True,
                        #binary=True
                        )

MyVect_TFIDF_STEM=CountVectorizer(input='content',
                        analyzer = 'word',
                        stop_words='english',
                        tokenizer=MY_STEMMER,
                        #strip_accents = 'unicode', 
                        lowercase = True,
                        #binary=True
                        )
X1=MyVect_TFIDF.fit_transform(champion_data_text['blurb'])
X2=MyVect_TFIDF_STEM.fit_transform(champion_data_text['blurb'])

column_names_x1 = MyVect_TFIDF.get_feature_names()
column_names_x2 = MyVect_TFIDF_STEM.get_feature_names()

df_x1 = pd.DataFrame(X1.toarray(),columns=column_names_x1)
df_x2 = pd.DataFrame(X2.toarray(),columns=column_names_x2)


#add labels
df_x1['labels'] = champion_data_text['tags']
df_x2['labels'] = champion_data_text['tags']


FinalDF_TFIDF=pd.DataFrame()
FinalDF_TFIDF_STEM=pd.DataFrame()


FinalDF_TFIDF= FinalDF_TFIDF.append(df_x1)
FinalDF_TFIDF_STEM = FinalDF_TFIDF_STEM.append(df_x2)

FinalDF_TFIDF=FinalDF_TFIDF.fillna(0)
FinalDF_TFIDF_STEM=FinalDF_TFIDF_STEM.fillna(0)

def RemoveNums(SomeDF):
    #print(SomeDF)
    print("Running Remove Numbers function....\n")
    temp=SomeDF
    MyList=[]
    for col in temp.columns:
        #print(col)
        #Logical1=col.isdigit()  ## is a num
        Logical2=str.isalpha(col) ## this checks for anything
        ## that is not a letter
        if(Logical2==False):# or Logical2==True):
            #print(col)
            MyList.append(str(col))
            #print(MyList)       
    temp.drop(MyList, axis=1, inplace=True)
            #print(temp)
            #return temp
       
    return temp

FinalDF_TFIDF=RemoveNums(FinalDF_TFIDF)
FinalDF_TFIDF_STEM=RemoveNums(FinalDF_TFIDF_STEM)


from sklearn.model_selection import train_test_split
import random as rd
rd.seed(1234)

TrainDF1, TestDF1 = train_test_split(FinalDF_TFIDF, test_size=0.3)
TrainDF2, TestDF2 = train_test_split(FinalDF_TFIDF_STEM, test_size=0.3)

Test1Labels=TestDF1['labels']
Test2Labels=TestDF2["labels"]
TestDF1 = TestDF1.drop(['labels'], axis=1)
TestDF2 = TestDF2.drop(['labels'], axis=1)


Train1Labels=TrainDF1['labels']
Train2Labels=TrainDF2['labels']
TrainDF1 = TrainDF1.drop(['labels'], axis=1)
TrainDF2 = TrainDF2.drop(['labels'], axis=1)


####################################################################
########################### Naive Bayes ############################
####################################################################
from sklearn.naive_bayes import MultinomialNB

MyModelNB1= MultinomialNB()
MyModelNB2= MultinomialNB()

MyModelNB1.fit(TrainDF1, Train1Labels)
MyModelNB2.fit(TrainDF2, Train2Labels)


Prediction1 = MyModelNB1.predict(TestDF1)
Prediction2 = MyModelNB2.predict(TestDF2)
print("\nThe prediction from NB is:")
print(Prediction1)
print("\nThe actual labels are:")
print(Test1Labels)

print("\nThe prediction from NB is:")
print(Prediction2)
print("\nThe actual labels are:")
print(Test2Labels)
from sklearn.metrics import confusion_matrix

cnf_matrix1 = confusion_matrix(Test1Labels, Prediction1)
print("\nThe confusion matrix is:")
print(cnf_matrix1)

cnf_matrix2 = confusion_matrix(Test2Labels, Prediction2)
print("\nThe confusion matrix is:")
print(cnf_matrix2)


from sklearn.decomposition import PCA

ymap = Train1Labels
ymap = ymap.replace('Assassin',1)
ymap = ymap.replace('Fighter',2)
ymap = ymap.replace('Mage',3)
ymap = ymap.replace('Marksman',4)
ymap = ymap.replace('Support',5)
ymap = ymap.replace('Tank',6)

pca = PCA(n_components=6)
proj = pca.fit_transform(TrainDF1)
plt.scatter(proj[:, 0], proj[:, 1], c=ymap, cmap="Paired")
plt.colorbar()


#############################################
###########  SVM ############################
#############################################


from sklearn.svm import LinearSVC


TRAIN= TrainDF1   ## As noted above - this can also be TrainDF2, etc.
TRAIN_Labels= Train1Labels
TEST= TestDF1
TEST_Labels= Test1Labels

SVM_Model1=LinearSVC(C=1)
SVM_Model1.fit(TRAIN, TRAIN_Labels)



SVM_matrix = confusion_matrix(Test1Labels, SVM_Model1.predict(TEST))
print("\nThe confusion matrix is:")
print(SVM_matrix)
print("\n\n")


import matplotlib.pyplot as plt
def plot_coefficients(MODEL=SVM_Model1, COLNAMES=TrainDF1 .columns, top_features=5):
    ## Model if SVM MUST be SVC, RE: SVM_Model=LinearSVC(C=10)
    coef = MODEL.coef_.ravel()
    top_positive_coefficients = np.argsort(coef,axis=0)[-top_features:]
    print(top_positive_coefficients)
    top_negative_coefficients = np.argsort(coef,axis=0)[:top_features]
    print(top_negative_coefficients)
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(15, 5))
    colors = ["red" if c < 0 else "blue" for c in coef[top_coefficients]]
    plt.bar(  x=  np.arange(2 * top_features)  , height=coef[top_coefficients], width=.5,  color=colors)
    feature_names = np.array(COLNAMES)
    feature_names_final = []
    for i in feature_names:
        for j in range(6):
            feature_names_final.append(i)
    
    plt.xticks(np.arange(0, (2*top_features)), np.array(feature_names_final)[top_coefficients], rotation=60, ha="right")
    plt.show()
    

plot_coefficients()



SVM_Model2 = SVC(C=1,kernel='poly',degree = 2,gamma = 'auto',verbose=True)
SVM_Model2.fit(TRAIN, TRAIN_Labels)

SVM_matrix2 =  confusion_matrix(Test1Labels, SVM_Model2.predict(TEST))
print("\nThe confusion matrix is:")
print(SVM_matrix2)
print("\n\n")
label = list(set(TRAIN_Labels))
sns.heatmap(SVM_matrix2,square=True,annot=True,fmt='d',cbar=False,cmap='GnBu',
            xticklabels=label,yticklabels=label)

#Record Data
Recordata = pd.read_csv('Clean_df.csv')
Recordata.drop(['bans','region','gameId','teamId','time','gold',"dominionVictoryScore",'firstBlood','firstTower','firstInhibitor','firstBaron','firstDragon','firstRiftHerald'],axis=1,inplace=True)
# Recordata
Recordata_train, Recordata_test = train_test_split(Recordata,test_size=0.3)



Recordata_test_labls = Recordata_test['win']
Recordata_test.drop(['win'],axis=1,inplace=True)

Recordata_train_labels  = Recordata_train['win']
Recordata_train.drop(['win'],axis=1,inplace=True)

####################################################################
########################### Naive Bayes ############################
####################################################################

x = Recordata_train.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
Recordata_train_s = pd.DataFrame(x_scaled)

x_test =  Recordata_test.values
x_test_scaled = min_max_scaler.fit_transform(x_test)
Recordata_test_s = pd.DataFrame(x_test_scaled)

MymodelNB_record = MultinomialNB()
MymodelNB_record.fit(Recordata_train,Recordata_train_labels)
PredictonNB_record = MymodelNB_record.predict(Recordata_test)


cnf_matrix_record = confusion_matrix(Recordata_test_labls,PredictonNB_record)
print("\nThe confusion matrix for naive bayse is:")
print(cnf_matrix_record)

from sklearn.decomposition import PCA

ymap = Recordata_train_labels
ymap = ymap.replace("Win",1)
ymap = ymap.replace("Fail",0)

pca = PCA(n_components=2)
proj = pca.fit_transform(Recordata_train)
plt.scatter(proj[:, 0], proj[:, 1], c=ymap, cmap="Paired")
plt.colorbar()



#############################################
###########  SVM ############################
#############################################

svm_model_record = LinearSVC(C=1)

svm_model_record.fit(Recordata_train_s,Recordata_train_labels)
svm_matrix_record = confusion_matrix(Recordata_test_labls,svm_model_record.predict(Recordata_test_s))


print("\nThe confusion matrix of svm doing record data is:")
print(svm_matrix_record)
print("\n\n")
# label = list(set(Recordata_train_labels))
# sns.heatmap(svm_matrix_record,square=True,annot=True,fmt='d',cbar=False,cmap='GnBu',
#             xticklabels=label,yticklabels=label)






svm_model_record = LinearSVC(C=0.1)

svm_model_record.fit(Recordata_train_s,Recordata_train_labels)
svm_matrix_record = confusion_matrix(Recordata_test_labls,svm_model_record.predict(Recordata_test_s))


print("\nThe confusion matrix of svm doing record data is:")
print(svm_matrix_record)
print("\n\n")
label = list(set(Recordata_train_labels))
sns.heatmap(svm_matrix_record,square=True,annot=True,fmt='d',cbar=False,cmap='GnBu',
            xticklabels=label,yticklabels=label)