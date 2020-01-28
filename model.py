#%%
import pandas as pd
import numpy as np
import os
import glob
import csv
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

#%%
# sınıflandırıcı işlemleri
def classifier(df):

    # feature vektörünün ilk iki sutunu feature(x1), son sutunu label(y1)
    x1 = df.iloc[:,:2]
    y1 = df['label']

    # kfold validation 5 yapıldı
    kfold = model_selection.KFold(n_splits=5, random_state=100)
    # sınıflandırıcı modeli logistic regression olarak ayarlı, parametreleri değişebilir
    model_kfold = LogisticRegression(solver='lbfgs')
    results_kfold = model_selection.cross_val_score(model_kfold, x1, y1, cv=kfold)
    print("Logistic Regression Accuracy: %.2f%%" % (results_kfold.mean()*100.0))

#%%
# Burada öznitelikler oluşturulur
def extractFeatures(df):

    # 'fDf': feature array olacak
    fDf = pd.DataFrame()

    length = [] # 1. feature: tweetteki kelime sayısı(normalize edilmemiş)
    mention = [] # 2. feature: tweetteki mention sayısı

    # tweetleri sırasıyla oku
    for text in df['tweet']:

        length += [len(text.split())]
        mention += [text.count('@')]
    
    # featureları arraye yerleştir
    fDf['length']=length
    fDf['mention']=mention
    # train dataframeden labelları ekle
    fDf['label']=df['subtask']
    
    # 2 feature içeren train array hazır
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
