import os, sys
import sklearn
from sklearn.datasets import load_files
import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
import re
import nltk
import string
from nltk.corpus import stopwords
from os.path import join as pjoin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

    
os.makedirs( './data/' );
os.makedirs( './data/+1/' );
os.makedirs( './data/-1/' );

#load data
df = pd.read_csv("test.csv", names = ['class','heading','date','text'])
stop_words = set(stopwords.words('english'))
print()

ps = PorterStemmer()

#pre-processing
def clean_str(istring):
    istring = re.sub(r"\n", "", istring)    
    istring = re.sub(r"\r", "", istring) 
    istring = re.sub(r"[0-9]", "digit", istring)
    istring = re.sub(r"\'", "", istring)    
    istring = re.sub(r"\"", "", istring)
    istring = re.sub('['+string.punctuation+']', ' ', istring)
    return istring.strip().lower()


print(df.shape[0])
for i in range(df.shape[0]):
    name = str (i)+ ".txt"    
    if df.iloc[i,3] == "(NO TEXT)":
        df.iloc[i,3] = df.iloc[i,1]
    
    if df.iloc[i,0] == +1:
        path_to_file = "./data/+1/" + name
        FILE = open(path_to_file, "w")
        df.iloc[i,3] = clean_str(df.iloc[i,3])
        words = word_tokenize(df.iloc[i,3])
        for w in words: 
            w = ps.stem(w)
            if w not in stop_words: 
                FILE.write("%s " %w)
        FILE.close()
    elif df.iloc[i,0] == -1:
        path_to_file = "./data/-1/" + name
        FILE = open(path_to_file, "w")
        df.iloc[i,3] = clean_str(df.iloc[i,3])
        words = word_tokenize(df.iloc[i,3])
        for w in words: 
            w = ps.stem(w)
            if w not in stop_words:
                FILE.write("%s " %w)
        FILE.close()
    
    
dir = r'./data'
# loading all files as training data. 
text_data = load_files(dir, shuffle=True)
# target names ("classes") are automatically generated from subfolder names

# Initialize a CoutVectorizer to use NLTK's tokenizer instead of its 
# default one (which ignores punctuation and stopwords). 
# initialize data_vector object, and then turn movie train data into a vector
data_vec = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize)  
data_counts = data_vec.fit_transform(text_data.data)

tfidf_transformer = TfidfTransformer()
data_tfidf = tfidf_transformer.fit_transform(data_counts)
# Same dimensions, now with tf-idf values instead of raw frequency count
#print(data_tfidf.shape)


#Uncomment below section to test with different classifiers and test accuracies
"""
docs_train, docs_test, y_train, y_test = train_test_split(data_tfidf, text_data.target, test_size = 0.33, random_state = 12)

print('DecisionTreeClassifier')
clf = tree.DecisionTreeClassifier().fit(docs_train, y_train)
y_pred = clf.predict(docs_test)
sklearn.metrics.accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(cm)
# 10 fold CV
clf = tree.DecisionTreeClassifier()
scores = cross_val_score(clf, data_tfidf, text_data.target, cv=10)
print (scores) # Results for all the folds
print ("10CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(),scores.std()*2))

print('NeighborsClassifier')
clf = neighbors.KNeighborsClassifier(n_neighbors=1).fit(docs_train, y_train)
y_pred = clf.predict(docs_test)
sklearn.metrics.accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(cm)
# 10 fold CV
clf = neighbors.KNeighborsClassifier(n_neighbors=1) 
scores = cross_val_score(clf, data_tfidf, text_data.target, cv=10)
print (scores) # Results for all the folds
print ("10CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(),scores.std()*2))

print('MultinomialNB')
clf = MultinomialNB().fit(docs_train, y_train)
y_pred = clf.predict(docs_test)
sklearn.metrics.accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(cm)
# 10 fold CV
clf = MultinomialNB()
scores = cross_val_score(clf, data_tfidf, text_data.target, cv=10)
print (scores) # Results for all the folds
print ("10CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(),scores.std()*2))

print('SVM')
clf = svm.SVC(kernel='linear', C=1,gamma=1).fit(docs_train, y_train)
y_pred = clf.predict(docs_test)
sklearn.metrics.accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(cm)
# 10 fold CV
clf = svm.SVC(kernel='linear', C=1,gamma=1)
scores = cross_val_score(clf, data_tfidf, text_data.target, cv=10)
print (scores) # Results for all the folds
print ("10CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(),scores.std()*2))
"""


clf = svm.SVC(kernel='linear', C=1,gamma=1).fit(data_tfidf, text_data.target)
df_test = pd.read_csv("testsetwithoutlabels.csv", names = ['heading','date','text'])

for i in range(df_test.shape[0]):    
    if df_test.iloc[i,2] == "(NO TEXT)":
        df_test.iloc[i,2] = df_test.iloc[i,0]
        
df_test_new_counts = data_vec.transform(df_test["text"])
df_test_new_tfidf = tfidf_transformer.transform(df_test_new_counts)
pred = clf.predict(df_test_new_tfidf)
# print out results
file = open("e14240.txt","w") 
for text_new, category in zip(df_test["text"], pred):
    #print('%s => %r' % ( text_data.target_names[category] , text_new ))
    print('%s' % ( text_data.target_names[category]))
    file.write(text_data.target_names[category]+'\n') 
file.close()



