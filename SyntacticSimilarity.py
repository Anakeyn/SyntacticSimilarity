# -*- coding: utf-8 -*-
"""
Created on Tue Aug  17 10:32:15 2021

@author: Pierre
"""
#See more details here :
#https://towardsdatascience.com/overview-of-text-similarity-metrics-3397c4601f50
#https://towardsdatascience.com/calculating-document-similarities-using-bert-and-other-models-b2c1a29c9630
#or in french
#https://ichi.pro/fr/calcul-des-similitudes-de-documents-a-l-aide-de-bert-et-d-autres-modeles-196544726603312

#import needed librarie
import pandas as pd #for dataframes
import numpy as np #for arrays
#import nltk stopwords
from nltk.corpus import stopwords  #nltk (Natural Language ToolKit) normaly comes With Anaconda
#import stopwords in french
stopWords = set(stopwords.words('french'))

import re #for regular expressions
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer #for vectorization

#needed for lemmatization
#Install it before in Anaconda console :
#conda install -c conda-forge spacy
#python -m spacy download fr_core_news_md

import spacy #nltk alternative. Lemmatization in French is not available with nltk
nlp = spacy.load('fr_core_news_md')  #Spacy pre-train model in French

def SpacyLemmatizer(doc):
    myDoc = nlp(doc)
    myLemmatizeDoc = " ".join([token.lemma_ for token in myDoc])
    print(myLemmatizeDoc)             
    return myLemmatizeDoc          
    

#################################################
# Loading and Cleaning  Data
#################################################
#Read my company file  (french version)
dfCompaniesFrenchDesc = pd.read_excel("CompaniesFrenchDesc.xlsx")
dfCompaniesFrenchDesc.dtypes


dfCompanies=pd.DataFrame(dfCompaniesFrenchDesc,columns=['Name', 'Description'])
#dfCompanies.rename(columns={"Description": "Description"}, inplace=True)

# removing special characters and stop words from the text
stop_words_l=stopwords.words('french')
#Keep numbers and accents but remove stop words
dfCompanies['Description_Cleaned']=dfCompanies.Description.apply(lambda x: " ".join(re.sub(r'[^a-zA-Z0-9À-ÖØ-öø-ÿœ]',' ',w).lower() for w in x.split() if re.sub(r'[^a-zA-Z0-9À-ÖØ-öø-ÿœ]',' ',w).lower() not in stopWords) )
#lemmatize
dfCompanies['Description_Cleaned'] = dfCompanies.Description_Cleaned.apply(lambda x: SpacyLemmatizer(x) )

#Check
#dfCompanies.Description_Cleaned.shape





################## instantiate the Count vectorizer object (for jaccard and Dice)
countvectoriser = CountVectorizer()
count_vectors = countvectoriser.fit_transform(dfCompanies.Description_Cleaned)
#transform sparse matrix in array
count_vectors_array = count_vectors.toarray()
#Checks :
#count_vectors_array.shape  #1321 documents / 62228 words
#np.unique(count_vectors_array[0], return_counts=True) #What we get in the first document : 14 words found, 6214 not found



############ instantiate the TF-IDF vectorizer object
tfidfvectoriser=TfidfVectorizer() 
tfidfvectoriser.fit(dfCompanies.Description_Cleaned)  #Fit tfidfvectoriser
#Calculate TF-IDF sparse matrix
tfidf_vectors=tfidfvectoriser.transform(dfCompanies.Description_Cleaned)

#Checks (Norm calculation)
#tfidf_vectors_array = tfidf_vectors.toarray()
#np.linalg.norm(tfidf_vectors_array[0]) # = 1



############ instantiate the BM25 vectorizer object
##########################  BM25 implementation by Koreyou
#see here https://gist.github.com/koreyou/f3a8a0470d32aa56b32f198f49a9f2b8
""" Implementation of OKapi BM25 with sklearn's TfidfVectorizer
Distributed as CC-0 (https://creativecommons.org/publicdomain/zero/1.0/)
"""

#import numpy as np  #already done
#from sklearn.feature_extraction.text import TfidfVectorizer  #already done
from scipy import sparse


class BM25(object):
    def __init__(self, b=0.75, k1=1.6):
        self.vectorizer = TfidfVectorizer(norm=None, smooth_idf=False)
        self.b = b
        self.k1 = k1

    def fit(self, X):
        """ Fit IDF to documents X """
        self.vectorizer.fit(X)
        y = super(TfidfVectorizer, self.vectorizer).transform(X)
        self.avdl = y.sum(1).mean()

    def transform(self, q, X):
        """ Calculate BM25 between query q and documents X """
        b, k1, avdl = self.b, self.k1, self.avdl

        # apply CountVectorizer
        X = super(TfidfVectorizer, self.vectorizer).transform(X)
        len_X = X.sum(1).A1
        q, = super(TfidfVectorizer, self.vectorizer).transform([q])
        assert sparse.isspmatrix_csr(q)

        # convert to csc for better column slicing
        X = X.tocsc()[:, q.indices]
        denom = X + (k1 * (1 - b + b * len_X / avdl))[:, None]
        # idf(t) = log [ n / df(t) ] + 1 in sklearn, so it need to be coneverted
        # to idf(t) = log [ n / df(t) ] with minus 1
        idf = self.vectorizer._tfidf.idf_[None, q.indices] - 1.
        numer = X.multiply(np.broadcast_to(idf, X.shape)) * (k1 + 1)                                                          
        return (numer / denom).sum(1).A1

bm25vectoriser=BM25() 
bm25vectoriser.fit(dfCompanies.Description_Cleaned)  #Fit bm25vectoriser
#no need of a matrix to calculate BM25 Scores (scores provided in the class)


##############  Scores Calculators 

#Jaccard Scores Manually
def Jaccard_Scores(myArray, i) :
    JaccardScores = np.zeros([myArray.shape[0]]) #myArray.shape[0]
    for j in range(0, myArray.shape[0]):
        CountArray1 = myArray[i].sum()
        CountArray2 = myArray[j].sum()
        Intersection = sum(myArray[i]*myArray[j]) 
        JaccardScores[j] = float(Intersection/(CountArray1+CountArray2 - Intersection))
    return JaccardScores


#Dice Scores Manually
def Dice_Scores(myArray, i) :
    DiceScores = np.zeros([myArray.shape[0]]) #myArray.shape[0]
    for j in range(0, myArray.shape[0]):
        CountArray1 = myArray[i].sum()
        CountArray2 = myArray[j].sum()
        Intersection = sum(myArray[i]*myArray[j]) 
        DiceScores[j] = float(2*Intersection/(CountArray1+CountArray2))
    return DiceScores     


#For TF-IDF (cosine similarity)
#all similarity scores for TF-IDF are calculated using the dot product of array and array.T
#np.dot Dot product of Two arrays : dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])
TFIDF_Scores=np.dot(tfidf_vectors,tfidf_vectors.T).toarray()
#TFIDF_Scores.shape


#For BM25
#Already Implemented in the BM25 class -> transform
#Calculate BM25 scores (example with 0 to others)  
BM25_Scores0=bm25vectoriser.transform(dfCompanies.Description_Cleaned[0], dfCompanies.Description_Cleaned)



################################################
# create dataframe for the Results : dfResults 
################################################

column_names = ["Index", 
                "Name", 
                "Description", 
                "Description_Cleaned",
                "Jaccard_Index", 
                "Jaccard_Name", 
                "Jaccard_Description", 
                "Jaccard_Description_Cleaned",
                "Jaccard_Score",
                "Dice_Index", 
                "Dice_Name", 
                "Dice_Description", 
                "Dice_Description_Cleaned",
                "Dice_Score",
                "TFIDF_Index", 
                "TFIDF_Name", 
                "TFIDF_Description", 
                "TFIDF_Description_Cleaned",
                "TFIDF_Score",
                "BM25_Index", 
                "BM25_Name", 
                "BM25_Description", 
                "BM25_Description_Cleaned",
                "BM25_Score"]

dfResults = pd.DataFrame(columns = column_names) #All Results

MaxSize = dfCompanies.shape[0] #1329
#MaxSize = 100 #if it's too long or to test different values.

for i in range(0,MaxSize):
    print("i:",i)      
    #Compare i to others
    
    
    ###### Jaccard Sore
    #Create Current dataframe containing my jaccard_Scores  for row i       
    dfCurrent_Jaccard_Scores=pd.DataFrame(Jaccard_Scores(count_vectors_array,i), columns=['Jaccard_Score'])

    #index in column
    dfCurrent_Jaccard_Scores['Jaccard_Index']=dfCurrent_Jaccard_Scores.index
    #Remove the current row
    dfCurrent_Jaccard_Scores.drop(i, inplace=True)
    #Sort 
    dfCurrent_Jaccard_Scores.sort_values(by=['Jaccard_Score'], ascending=False, ignore_index=True, inplace=True)
    #dfCurrent_Jaccard_Scores
    Jaccard_Index= int(dfCurrent_Jaccard_Scores.loc[0].Jaccard_Index) #make sure you have an index
    print("Jaccard_Index:",Jaccard_Index)
    Jaccard_Score = dfCurrent_Jaccard_Scores.loc[0].Jaccard_Score
    print("Jaccard_Score:",Jaccard_Score)   
    
   
 
    ###### Dice Sore
    #Create Current dataframe containing my dice scores  for row i       
    dfCurrent_Dice_Scores=pd.DataFrame(Dice_Scores(count_vectors_array,i), columns=['Dice_Score'])

    #index in column
    dfCurrent_Dice_Scores['Dice_Index']=dfCurrent_Dice_Scores.index
    #Remove the current row
    dfCurrent_Dice_Scores.drop(i, inplace=True)
    #Sort 
    dfCurrent_Dice_Scores.sort_values(by=['Dice_Score'], ascending=False, ignore_index=True, inplace=True)
    #dfCurrent_Dice_Scores
    Dice_Index= int(dfCurrent_Dice_Scores.loc[0].Dice_Index) #make sure you have an index
    print("Dice_Index:",Dice_Index)
    Dice_Score = dfCurrent_Dice_Scores.loc[0].Dice_Score
    print("Dice_Score:",Dice_Score)  
    
    
    ####################################################
    ######  For TF-IDF
    #Create Current dataframe containing similarities from  TFIDF_Scores for row i          
    dfCurrent_TFIDF_Scores=pd.DataFrame(TFIDF_Scores[i][:], columns=['TFIDF_Score'])
    #index in column
    dfCurrent_TFIDF_Scores['TFIDF_Index']=dfCurrent_TFIDF_Scores.index
    #Remove the current row
    dfCurrent_TFIDF_Scores.drop(i, inplace=True)
    #Sort 
    dfCurrent_TFIDF_Scores.sort_values(by=['TFIDF_Score'], ascending=False, ignore_index=True, inplace=True)
    #dfCurrent_TFIDF_Scores
    TFIDF_Index= int(dfCurrent_TFIDF_Scores.loc[0].TFIDF_Index) #make sure you have an index
    print("TFIDF_Index:",TFIDF_Index)
    TFIDF_Score = dfCurrent_TFIDF_Scores.loc[0].TFIDF_Score
    print("TFIDF_Score:",TFIDF_Score)
    

    ####################################################
    ######  For OKAPI BM25
    #Create Current dataframe containing similarities from  bm25 scores for row i          
    dfCurrent_BM25_Scores=pd.DataFrame(bm25vectoriser.transform(dfCompanies.Description_Cleaned[i], dfCompanies.Description_Cleaned), columns=['BM25_Score'])
    #index in column
    dfCurrent_BM25_Scores['BM25_Index']=dfCurrent_BM25_Scores.index
    #Remove the current row
    dfCurrent_BM25_Scores.drop(i, inplace=True)
    #Sort 
    dfCurrent_BM25_Scores.sort_values(by=['BM25_Score'], ascending=False, ignore_index=True, inplace=True)
    #dfCurrent_BM25_Scores
    BM25_Index= int(dfCurrent_BM25_Scores.loc[0].BM25_Index) #make sure you have an index
    print("BM25_Index:",BM25_Index)
    BM25_Score = dfCurrent_BM25_Scores.loc[0].BM25_Score
    print("BM25_Score:",BM25_Score)

    
    dfCurrentResults = pd.DataFrame({
                
                "Index" : i, 
                "Name" : dfCompanies.loc[i].Name,                     
                "Description" : dfCompanies.loc[i].Description, 
                "Description_Cleaned" : dfCompanies.loc[i].Description_Cleaned, 
                
                
                "Jaccard_Index" : Jaccard_Index,  
                "Jaccard_Name" : dfCompanies.loc[Jaccard_Index].Name,
                "Jaccard_Description" : dfCompanies.loc[Jaccard_Index].Description, 
                "Jaccard_Description_Cleaned" : dfCompanies.loc[Jaccard_Index].Description_Cleaned, 
                "Jaccard_Score" : Jaccard_Score,
                
                
                "Dice_Index" : Dice_Index,  
                "Dice_Name" : dfCompanies.loc[Dice_Index].Name,
                "Dice_Description" : dfCompanies.loc[Dice_Index].Description, 
                "Dice_Description_Cleaned" : dfCompanies.loc[Dice_Index].Description_Cleaned, 
                "Dice_Score" : Dice_Score,

                "TFIDF_Index" : TFIDF_Index,  
                "TFIDF_Name" : dfCompanies.loc[TFIDF_Index].Name,
                "TFIDF_Description" : dfCompanies.loc[TFIDF_Index].Description, 
                "TFIDF_Description_Cleaned" : dfCompanies.loc[TFIDF_Index].Description_Cleaned, 
                "TFIDF_Score" : TFIDF_Score,
                
                "BM25_Index" : BM25_Index,  
                "BM25_Name" : dfCompanies.loc[BM25_Index].Name,
                "BM25_Description" : dfCompanies.loc[BM25_Index].Description, 
                "BM25_Description_Cleaned" : dfCompanies.loc[BM25_Index].Description_Cleaned, 
                "BM25_Score" : BM25_Score},
        
                index=[0])
    
    
    
    dfResults = pd.concat([dfResults,dfCurrentResults]) #add to global results
    dfResults.reset_index(inplace=True, drop=True)  #reset index




#####################################################################
# Save Results in Excel
#####################################################################

#All Results    
dfResults.shape
dfResults.to_excel("SyntaxResults"+str(MaxSize)+".xlsx", sheet_name='SyntaxFResults', index=False)   


#All Similar results
dfJDTBResults = dfResults.loc[(dfResults['Jaccard_Index']  == dfResults['Dice_Index']) &
                              (dfResults['Jaccard_Index'] == dfResults['TFIDF_Index']) & 
                              (dfResults['Jaccard_Index'] ==  dfResults['BM25_Index']) ]
dfJDTBResults.shape
dfJDTBResults.to_excel("JDTBResults"+str(MaxSize)+".xlsx", sheet_name='JDTBResults', index=False)  

########  Jaccard vs Dice
#Jaccard Dice  Same  results
dfJDResults = dfResults.loc[(dfResults['Jaccard_Index']  == dfResults['Dice_Index'])]
dfJDResults.drop(columns=["TFIDF_Index", "TFIDF_Name", "TFIDF_Description", "TFIDF_Description_Cleaned", "TFIDF_Score",
                "BM25_Index", "BM25_Name", "BM25_Description", "BM25_Description_Cleaned", "BM25_Score"], inplace=True)
dfJDResults.shape
dfJDResults.to_excel("JDResults"+str(MaxSize)+".xlsx", sheet_name='JDResults', index=False) 

#Jaccard Dice  Different  results
dfNotJDResults = dfResults.loc[(dfResults['Jaccard_Index']  != dfResults['Dice_Index'])]
dfNotJDResults.drop(columns=["TFIDF_Index", "TFIDF_Name", "TFIDF_Description", "TFIDF_Description_Cleaned", "TFIDF_Score",
                "BM25_Index", "BM25_Name", "BM25_Description", "BM25_Description_Cleaned", "BM25_Score"], inplace=True)
dfNotJDResults.shape
dfNotJDResults.to_excel("NotJDResults"+str(MaxSize)+".xlsx", sheet_name='NotJDResults', index=False) 

########  Jaccard vs TFIDF
#Jaccard TFIDF  Same  results
dfJTResults = dfResults.loc[(dfResults['Jaccard_Index']  == dfResults['TFIDF_Index'])]
dfJTResults.drop(columns=["Dice_Index", "Dice_Name", "Dice_Description", "Dice_Description_Cleaned", "Dice_Score",
                "BM25_Index", "BM25_Name", "BM25_Description", "BM25_Description_Cleaned", "BM25_Score"], inplace=True)
dfJTResults.shape
dfJTResults.to_excel("JTResults"+str(MaxSize)+".xlsx", sheet_name='JTResults', index=False) 

#Jaccard TFIDF  Different  results
dfNotJTResults = dfResults.loc[(dfResults['Jaccard_Index']  != dfResults['TFIDF_Index'])]
dfNotJTResults.drop(columns=["Dice_Index", "Dice_Name", "Dice_Description", "Dice_Description_Cleaned", "Dice_Score",
                "BM25_Index", "BM25_Name", "BM25_Description", "BM25_Description_Cleaned", "BM25_Score"], inplace=True)
dfNotJTResults.shape
dfNotJTResults.to_excel("NotJTResults"+str(MaxSize)+".xlsx", sheet_name='NotJTResults', index=False) 

########  Jaccard vs BM25
#Jaccard BM25  Same  results
dfJBResults = dfResults.loc[(dfResults['Jaccard_Index']  == dfResults['BM25_Index'])]
dfJBResults.drop(columns=["Dice_Index", "Dice_Name", "Dice_Description", "Dice_Description_Cleaned", "Dice_Score",
                "TFIDF_Index", "TFIDF_Name", "TFIDF_Description", "TFIDF_Description_Cleaned", "TFIDF_Score"], inplace=True)
dfJBResults.shape
dfJBResults.to_excel("JBResults"+str(MaxSize)+".xlsx", sheet_name='JBResults', index=False) 

#Jaccard BM25  Different  results
dfNotJBResults = dfResults.loc[(dfResults['Jaccard_Index']  != dfResults['BM25_Index'])]
dfNotJBResults.drop(columns=["Dice_Index", "Dice_Name", "Dice_Description", "Dice_Description_Cleaned", "Dice_Score",
                "TFIDF_Index", "TFIDF_Name", "TFIDF_Description", "TFIDF_Description_Cleaned", "TFIDF_Score"], inplace=True)
dfNotJBResults.shape
dfNotJBResults.to_excel("NotJBResults"+str(MaxSize)+".xlsx", sheet_name='NotJBResults', index=False) 


########  TFIDF vs BM25
#TFIDF BM25  Same results
dfTBResults = dfResults.loc[(dfResults['TFIDF_Index']  == dfResults['BM25_Index'])]
dfTBResults.drop(columns=["Jaccard_Index", "Jaccard_Name", "Jaccard_Description", "Jaccard_Description_Cleaned", "Jaccard_Score",
                "Dice_Index", "Dice_Name", "Dice_Description", "Dice_Description_Cleaned", "Dice_Score"], inplace=True)
dfTBResults.shape
dfTBResults.to_excel("TBResults"+str(MaxSize)+".xlsx", sheet_name='TBResults', index=False)  

#TFIDF BM25  Different results
dfNotTBResults = dfResults.loc[(dfResults['TFIDF_Index']  != dfResults['BM25_Index'])]
dfNotTBResults.drop(columns=["Jaccard_Index", "Jaccard_Name", "Jaccard_Description", "Jaccard_Description_Cleaned", "Jaccard_Score",
                "Dice_Index", "Dice_Name", "Dice_Description", "Dice_Description_Cleaned", "Dice_Score"], inplace=True)
dfNotTBResults.shape
dfNotTBResults.to_excel("NotTBResults"+str(MaxSize)+".xlsx", sheet_name='NotTBResults', index=False)  