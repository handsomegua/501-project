import requests
import json
import numpy
import pandas as pd
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
champion_data = pd.read_csv('champion_infor_501.csv',encoding='latin1')
champion_data_text = champion_data[['name','tags','blurb']]

STEMMER=PorterStemmer()
def MY_STEMMER(str_input):
    words = re.sub(r"[^A-Za-z\-]", " ", str_input).lower().split()
    words = [STEMMER.stem(word) for word in words]
    return words
MyVect_TFIDF=TfidfVectorizer(input='content',
                        analyzer = 'word',
                        stop_words='english',
                        lowercase = True,
                        #binary=True
                        )

MyVect_TFIDF_STEM=TfidfVectorizer(input='content',
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

MyDT=DecisionTreeClassifier(criterion='gini', ##"entropy" or "gini"
                            splitter='random',  ## or "random" or "best"
                            max_depth=None, 
                            min_samples_split=5, 
                            min_samples_leaf=5, 
                            min_weight_fraction_leaf=0.0, 
                            max_features=None, 
                            random_state=None, 
                            max_leaf_nodes=None, 
                            min_impurity_decrease=0.0, 
                            min_impurity_split=None, 
                            class_weight=None)
MyDT_1=DecisionTreeClassifier(criterion='entropy', ##"entropy" or "gini"
                            splitter='random',  ## or "random" or "best"
                            max_depth=None, 
                            min_samples_split=5, 
                            min_samples_leaf=5, 
                            min_weight_fraction_leaf=0.0, 
                            max_features=None, 
                            random_state=None, 
                            max_leaf_nodes=None, 
                            min_impurity_decrease=0.0, 
                            min_impurity_split=None, 
                            class_weight=None)

for i in [1,2]:
    temp1=str("TrainDF"+str(i))
    temp2=str("Train"+str(i)+"Labels")
    temp3=str("TestDF"+str(i))
    temp4=str("Test"+str(i)+"Labels")
    
    ## perform DT
    MyDT.fit(eval(temp1), eval(temp2))
    ## plot the tree
    tree.plot_tree(MyDT)
    plt.savefig(temp1)
    feature_names=eval(str(temp1+".columns"))
    dot_data = tree.export_graphviz(MyDT, out_file=None,
                    ## The following creates TrainDF.columns for each
                    ## which are the feature names.
                      feature_names=eval(str(temp1+".columns")),  
                      #class_names=MyDT.class_names,  
                      filled=True, rounded=True,  
                      special_characters=True)                                    
    graph = graphviz.Source(dot_data) 
  
    ## Create dynamic graph name
    tempname=str("Graph" + str(i))
    graph.render(tempname) 
    ## Show the predictions from the DT on the test set
    print("\nActual for DataFrame: ", i, "\n")
    print(eval(temp2))
    print("Prediction\n")
    DT_pred=MyDT.predict(eval(temp3))
    print(DT_pred)
    ## Show the confusion matrix
    bn_matrix = confusion_matrix(eval(temp4), DT_pred)
    print("\nThe confusion matrix is:")
    print(bn_matrix)
    FeatureImp=MyDT.feature_importances_   
    indices = np.argsort(FeatureImp)[::-1]
    ## print out the important features.....
    for f in range(TrainDF2.shape[1]):
        if FeatureImp[indices[f]] > 0:
            print("%d. feature %d (%f)" % (f + 1, indices[f], FeatureImp[indices[f]]))
            print ("feature name: ", feature_names[indices[f]])




for i in [1,2]:
    temp1=str("TrainDF"+str(i))
    temp2=str("Train"+str(i)+"Labels")
    temp3=str("TestDF"+str(i))
    temp4=str("Test"+str(i)+"Labels")
    
    ## perform DT
    MyDT_1.fit(eval(temp1), eval(temp2))
    ## plot the tree
    tree.plot_tree(MyDT_1)
    plt.savefig(temp1)
    feature_names_1=eval(str(temp1+".columns"))
    dot_data_1 = tree.export_graphviz(MyDT_1, out_file=None,
                    ## The following creates TrainDF.columns for each
                    ## which are the feature names.
                      feature_names=eval(str(temp1+".columns")),  
                      #class_names=MyDT.class_names,  
                      filled=True, rounded=True,  
                      special_characters=True)                                    
    graph_1 = graphviz.Source(dot_data_1) 
  
    ## Create dynamic graph name
    tempname=str("Graph_entropy" + str(i))
    graph_1.render(tempname) 
    ## Show the predictions from the DT on the test set
    print("\nActual for DataFrame: ", i, "\n")
    print(eval(temp2))
    print("Prediction\n")
    DT_pred_1=MyDT_1.predict(eval(temp3))
    print(DT_pred)
    ## Show the confusion matrix
    bn_matrix_1 = confusion_matrix(eval(temp4), DT_pred_1)
    print("\nThe confusion matrix is:")
    print(bn_matrix_1)
    FeatureImp=MyDT_1.feature_importances_   
    indices = np.argsort(FeatureImp)[::-1]


from six import StringIO
from IPython.display import Image  
dot_data2 = StringIO()
export_graphviz(MyDT, out_file=dot_data2,  
                filled=True, rounded=True,
                special_characters=True,
                feature_names = TrainDF2.columns)

                #class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data2.getvalue())  
graph.write_png('Dt_2.png')
Image(graph.create_png())







#Random forest for text data
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
RF = RandomForestClassifier(min_samples_leaf=3)

RF.fit(TrainDF1,Train1Labels)
RF_pred=RF.predict(TestDF1)
bn_matrix_RF_text = confusion_matrix(Test1Labels, RF_pred)  
bn_matrix_RF_text
FeaturesT=TrainDF1.columns
figT, axesT = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4),dpi=1000)
tree.plot_tree(RF.estimators_[0],
               feature_names = FeaturesT, 
               #class_names=Targets,
               filled = True)
FeaturesT=TrainDF1.columns
#Targets=StudentTestLabels_Num

##save it
figT.savefig('RF_Tree_Text')  ## creates png

RF.fit(TrainDF1,Train1Labels)
RF_pred=RF.predict(TestDF1)
bn_matrix_RF_text = confusion_matrix(Test1Labels, RF_pred)  
bn_matrix_RF_text
FeaturesT=TrainDF1.columns
figT, axesT = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4),dpi=1000)
tree.plot_tree(RF.estimators_[0],
               feature_names = FeaturesT, 
               #class_names=Targets,
               filled = True)
# FeaturesT=TrainDF1.columns
# #Targets=StudentTestLabels_Num

# ##save it
# figT.savefig('RF_Tree_Text')  ## creates png

FeatureImpRF=RF.feature_importances_   
indicesRF = np.argsort(FeatureImpRF)[::-1]
## print out the important features.....
for f2 in range(TrainDF1.shape[1]):   ##TrainDF1.shape[1] is number of columns
    if FeatureImpRF[indicesRF[f2]] >= 0.01:
        print("%d. feature %d (%.2f)" % (f2 + 1, indicesRF[f2], FeatureImpRF[indicesRF[f2]]))
        print ("feature name: ", FeaturesT[indicesRF[f2]])


