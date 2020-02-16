#%%
import pandas as pd
import numpy as np
import os
import glob
import csv
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import nltk
from nltk import word_tokenize
from nltk.collocations import BigramCollocationFinder
from sklearn.feature_extraction.text import CountVectorizer
#%%
# sınıflandırıcı işlemleri
def classifier(df):

    # feature vektörünün son sütuna kadar feature(x1), son sutunu label(y1)
    x1 = df.iloc[:,:-1]
    y1 = df['label']

    # kfold validation 5 yapıldı
    kfold = model_selection.KFold(n_splits=5, random_state=100)
    # sınıflandırıcı modeli logistic regression olarak ayarlı, parametreleri değişebilir
    model_kfold = LogisticRegression(solver='lbfgs')
    results_kfold = model_selection.cross_val_score(model_kfold, x1, y1, cv=kfold)
    print("Logistic Regression Accuracy: %.2f%%" % (results_kfold.mean()*100.0))

#%%
def extractFeatures(df):

    # 'fDf': feature array olacak
    fDf = pd.DataFrame()

    length = [] # 1. feature: tweetteki kelime sayısı(normalize edilmemiş)
    mention = [] # 2. feature: tweetteki mention sayısı

    
    # tweetleri sırasıyla oku
    for text in df['tweet']:

        length += [len(text.split())]
        mention += [text.count('@')]

    print("Toplam tweet sayısı: {}".format(len(df)))
    """
    # burada min_df değerini en az kaç tweette bulunan unigramları kullanmak için ayarla; örneğin en az 10 tweette geçen unigramlar; ngram_range ise neye bakıldığını belirler unigram için (1,1), bigram için (2,2)
    vectorizerUnigram = CountVectorizer(ngram_range=(1, 1),min_df=10,token_pattern=u"(?u)\\b\\w+\\b") 
    vectorizerUnigram.fit_transform(df['tweet'].values.astype('U'))     #tüm unigramları çıkarır
    print("Kullanılacak unigram sayısı: {}".format(len(vectorizerUnigram.get_feature_names())))
    unigrams = vectorizerUnigram.get_feature_names()            #kullanılacak unigram listesi
    vectorizer = CountVectorizer(ngram_range=(1, 1),token_pattern=u"(?u)\\b\\w+\\b")    #bunu kullanarak her tweet tekrar unigramlara ayrılır ve önceden bulduklarımız içinde var mı kontrol edilir
    
    unigramF = np.zeros(shape=(len(df),len(unigrams)))  # tweet sayısı * unigram sayısı kadar 
    for lineIdx in range(0,len(df.tweet)):
            vectorizer.fit_transform([str(df.tweet[lineIdx])])
            tweetUni = vectorizer.get_feature_names()       #bir tweetteki unigramlar
            for uni in unigrams :
                if (uni in tweetUni):
                    unigramF[lineIdx][unigrams.index(uni)]=1
                    """



    # aynı süreç bigram için
    vectorizerBigram = CountVectorizer(ngram_range=(2, 2),min_df=10,token_pattern=u"(?u)\\b\\w+\\b") 
    vectorizerBigram.fit_transform(df['tweet'].values.astype('U'))
    print("Kullanılacak bigram sayısı: {}".format(len(vectorizerBigram.get_feature_names())))
    bigrams = vectorizerBigram.get_feature_names()
    vectorizer = CountVectorizer(ngram_range=(2, 2),token_pattern=u"(?u)\\b\\w+\\b")    

    bigramF = np.zeros(shape=(len(df),len(bigrams)))  
    for lineIdx in range(0,len(df.tweet)):
            vectorizer.fit_transform([str(df.tweet[lineIdx])])
            tweetBi = vectorizer.get_feature_names()       
            for bi in bigrams :
                if (bi in tweetBi):
                    bigramF[lineIdx][bigrams.index(bi)]=1           
    bigramDf = pd.DataFrame(bigramF)
    print(bigramDf)

    # featureları arraye yerleştir
    fDf['length']=length
    fDf['mention']=mention
    
    
    # !!! kullanmak istenilen vektörü ana feature vektore ekle
    fDf = pd.concat([fDf, bigramDf], axis=1)
    

    # train dataframeden labelları ekle
    fDf['label']=df['subtask']

    print(fDf[:10])

    # hazırlanan dataframei sınıflandırıcıya ver
    classifier(fDf)


# %%
# train dosyasını oku ve dataframe hazırla
def readTrainFiles():
    columns = ['id', 'tweet', 'subtask']
    trainFiles = []
    ext=".tsv"
    for file in glob.glob( "./train/*" + ext): trainFiles.append(file)
    print("Train dosyasi: {}".format(trainFiles))

    dataDf = pd.read_csv(file, delimiter = '\t', encoding = 'utf-8', names = columns, header=0)

    print("Train dosyasi dataframe ornegi:")
    print(dataDf[0:10])

    print("Dataframede OFF ve NOT yerine label olarak 1 ve 0 yazıldı :")
    dataDf.loc[dataDf.subtask=='OFF', 'subtask']=1
    dataDf.loc[dataDf.subtask=='NOT', 'subtask']=0
    print(dataDf[0:10])
    extractFeatures(dataDf)


# %%

if __name__ == "__main__":
    readTrainFiles()


# %%
