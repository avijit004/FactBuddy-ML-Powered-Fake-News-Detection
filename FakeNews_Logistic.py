import pandas as pd
import numpy as np
import re
import matplotlib as plt
import matplotlib.pyplot as pyplot
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,precision_score,recall_score,f1_score,roc_curve,auc
import nltk
nltk.download('stopwords')
#print(stopwords.words('english'));
from sklearn import preprocessing 
import os

print("OK")
input_dir='C:/Users/User01/Desktop/python idle/Data/'
df_train=pd.read_csv(input_dir+'FakeNews.csv')
#df_train=df_train.iloc[0:20]
col_name=list(df_train)
#print(df_train.isnull().sum())  #Check for missing values in dataset #
#df_train=df_train.dropna()    #drop the entire row with at least one missing/nan value
df_train=df_train.fillna(' ')
df_train['content']=df_train['Headline']+' '+df_train['Body']
'''
for i in col_name:
        if df_train[i].dtype==object:    #if data is object type(ie Y/N) encode it
                scaler=preprocessing.OrdinalEncoder()
                df_train[i]=scaler.fit_transform(df_train)
'''
#print(df_train)


#print(df_train.dtypes)
#X=df_train.iloc[:, :-1].values  #column to be discarded in X (variables)
#y=df_train.iloc[:, -1].values     #column to taken for y (target)
X=df_train.drop(['Label'],axis=1)    # Alternative for above two lines
y=df_train['Label']

# stemming

port_stem=PorterStemmer()
def stemming(content):
        stemmed_content=re.sub('[^a-zA-Z]',' ',content)
        stemmed_content=stemmed_content.lower()
        stemmed_content=stemmed_content.split()
        '''
        for word in (stemmed_content):
                if word not in stopwords.words('english'):
                        stemmed_content=port_stem.stem(word)
        '''            
        #stemmed_content=[port_stem.stem(word) for word in (stemmed_content) if word not in stopwords.words('english')]
        stemmed_content=' '.join(stemmed_content)
        #print(stemmed_content)
        return stemmed_content
df_train['content']=df_train['content'].apply(stemming)
#print("Ok1")
#print(df_train['content'])
X=df_train['content'].values
Y=df_train['Label'].values            
#print(X);
#Vectorization

vectorizer=TfidfVectorizer()
vectorizer.fit(X)
X=vectorizer.fit_transform(X)
#print(X)

# Training and testing set of data       
X_train,X_test,y_train,y_test=train_test_split( X,y,test_size=0.25,shuffle=True,random_state=0)

print('Train Data')
print(X_train,y_train)
print('Test Data')
print(X_test,y_test)
'''
scaler=preprocessing.LabelEncoder()
print(X_train.columns)

y_train=scaler.fit_transform(y_train)
y_test=scaler.fit_transform(y_test)
'''
print('After encoding')
#print(X_train,y_train)
#print(X_test,y_test)

print("\nDecision Tree")
clf=DecisionTreeClassifier(random_state=10)
clf.fit(X_train,y_train)
res=clf.predict(X_test)
correct_results = sum(y_test == res)   #how many target result from model matches actual supplied target data y_train
print('Accuracy score: %0.3f' % (correct_results/float(y_test.size)))

print("\nRandom Forest")
clf=RandomForestClassifier(random_state=10 )
clf.fit(X_train,y_train)
res=clf.predict(X_test)
correct_results = sum(y_test == res)   #how many target result from model matches actual supplied target data y_train
print('Accuracy score: %0.3f' % (correct_results/float(y_test.size)))
'''
print("\nClustering")
clf=KMeans(n_clusters=2, n_init=10, random_state=0)
clf.fit(X_train,y_train)
res=clf.predict(X_test)
correct_results = sum(y_test == res)   #how many target result from model matches actual supplied target data y_train
print('Accuracy score: %0.3f' % (correct_results/float(y_test.size)))
'''
print("\nSVM")
clf=svm.LinearSVC()
clf.fit(X_train,y_train)
res=clf.predict(X_test)
correct_results = sum(y_test == res)   #how many target result from model matches actual supplied target data y_train
print('Accuracy score: %0.3f' % (correct_results/float(y_test.size)))

print("\nKNN")
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=4 ) 
clf.fit(X_train,y_train)
res=clf.predict(X_test)
correct_results = sum(y_test == res)   #how many target result from model matches actual supplied target data y_train
print('Accuracy score: %0.3f' % (correct_results/float(y_test.size)))

print("\nNaive Bayes")
from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB() 
clf.fit(X_train,y_train)
res=clf.predict(X_test)
correct_results = sum(y_test == res)   #how many target result from model matches actual supplied target data y_train
print('Accuracy score: %0.3f' % (correct_results/float(y_test.size)))

print("\nLogisticRegression")
clf=LogisticRegression()
clf.fit(X_train,y_train)
#Model analysis
#print('score=',clf.score(X_train,y_train))
res=clf.predict(X_test)
#print('Test result',res)
y_pred_proba = clf.predict_proba(X_test)[:, 1]
correct_results = sum(y_test == res)   #how many target result from model matches actual supplied target data y_train

print(pd.DataFrame({'Actual': y_test, 'Predicted':res}))
precision = precision_score(y_test, res)
recall = recall_score(y_test, res)
f1 = f1_score(y_test, res)
print("Result: %d out of %d samples were correctly labeled." % (correct_results, y_test.size))
print('Accuracy score: %0.3f' % (correct_results/float(y_test.size)))

print("Precision score:", precision)
print("Recall score:", recall)
print("F1 Score:", f1)

import pickle

with open('model.bm1', 'wb') as f:
    pickle.dump(clf, f)

with open('vectorizer.bm1', 'wb') as f:
    pickle.dump(vectorizer, f)

# for external data
'''
choice=input("Do you want to test news (Y for yes   N for no) : ")
while choice=="Y" or choice=="y":
        
        news=input("Enter News to be Tested :\n")
        testing_news={"text":[news]}
        new_def_test=pd.DataFrame(testing_news)
        new_def_test["text"]=new_def_test["text"].apply(stemming)
        new_x_test=new_def_test["text"]
        #print (new_x_test)
        new_xv_test=vectorizer.transform(new_x_test)
        test_res=clf.predict(new_xv_test)
        if test_res[0]==0:
                print("Fake News")
        else:
                print("Authentic News")
        choice=input("Any more checking (Y/N) : ")
'''       
# Plot
'''
pyplot.figure(figsize=(10,10))
for i in range(0,50,10):
        tree.plot_tree(clf.estimators_[i].fit(X_test,y_test))
        pyplot.show()

'''

cm = confusion_matrix(y_test, res, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
disp.plot()
pyplot.show()

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba) 
roc_auc = auc(fpr, tpr)
# Plot the ROC curve
pyplot.figure()  
pyplot.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pyplot.plot([0, 1], [0, 1], 'k--', label='No Skill')
pyplot.xlim([0.0, 0.5])
pyplot.ylim([0.0, 1.05])
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.title('ROC Curve for Fake News Detection')
pyplot.legend()
pyplot.show()

