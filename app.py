import pandas as pd
import numpy as np
df=pd.read_csv('mail_data.csv')
df.sample(5)
df.shape
df.info()
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
df['Category']=encoder.fit_transform(df['Category'])
df.isnull().sum()
df.duplicated().sum()
df=df.drop_duplicates(keep='first')
df.shape
df['Category'].value_counts()
import matplotlib.pyplot as plt
plt.pie(df['Category'].value_counts(),labels=['ham','spam'],autopct="%0.2f")
plt.show()
import nltk
pip install nltk
nltk.download('punkt')
df['num_characters']=df['Message'].apply(len)
df.head()
df['num_words']=df['Message'].apply(lambda x:len(nltk.word_tokenize(x)))
df['num_sents']=df['Message'].apply(lambda x:len(nltk.sent_tokenize(x)))
df.head()
df[['num_characters','num_words','num_sents']].describe()
df[df['Category']==0][['num_characters','num_words','num_sents']].describe()
df[df['Category']==1][['num_characters','num_words','num_sents']].describe()
import seaborn as sns
sns.histplot(df[df['Category']==0]['num_characters'])
sns.histplot(df[df['Category']==1]['num_characters'],color='red')
sns.histplot(df[df['Category']==0]['num_words'])
sns.histplot(df[df['Category']==1]['num_words'],color='red')
sns.pairplot(df,hue='Category')
import string
nltk.download('popular')
transform_text("LMAO where's your fish memory when I need it")
df['Message'][2000]
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
ps.stem('loving')
df['transformed_text']=df['Message'].apply(transform_text)
pip install wordcloud
df.head()
from wordcloud import WordCloud
wc=WordCloud(width=500,height=500,min_font_size=10,background_color='white')
spam_wc=wc.generate(df[df['Category']==1]['transformed_text'].str.cat(sep=" "))
plt.imshow(spam_wc)
ham_wc=wc.generate(df[df['Category']==0]['transformed_text'].str.cat(sep=" "))
plt.imshow(ham_wc)
df.head()
spam_corpus=[]
for msg in df[df['Category']==1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)
spam_corpus
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv=CountVectorizer()
tfidf=TfidfVectorizer(max_features=3000)
X=tfidf.fit_transform(df['transformed_text']).toarray()
X
y=df['Category'].values
y
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)
from sklearn.naive_bayes import GaussianNB, MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
gnb=GaussianNB()
mnb=MultinomialNB()
bnb=BernoulliNB()
gnb.fit(X_train,y_train)
y_pred1=gnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))
mnb.fit(X_train,y_train)
y_pred2=mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))
bnb.fit(X_train,y_train)
y_pred3=bnb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))
def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)

    y=[]
    for i in text:
         if i.isalnum():
             y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in string.punctuation and i not in stopwords.words('english'):
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return mnb.predict(" ".join(y))
msg = "[Update] Congratulations Nile Yogesh, You account is activated for investment in Stocks. Click to invest now:"

if transform_text(msg):
    print("spam")
else:
    print("not spam") 
