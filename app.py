import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import string
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score

# Load data
df = pd.read_csv('mail_data.csv')

# Data preprocessing
df['Category'] = LabelEncoder().fit_transform(df['Category'])
df = df.drop_duplicates(keep='first')

# Visualize category distribution
plt.pie(df['Category'].value_counts(), labels=['ham', 'spam'], autopct="%0.2f")
plt.show()

# Add text features
df['num_characters'] = df['Message'].apply(len)
df['num_words'] = df['Message'].apply(lambda x: len(nltk.word_tokenize(x)))
df['num_sents'] = df['Message'].apply(lambda x: len(nltk.sent_tokenize(x)))

# Visualize data distribution
sns.histplot(df[df['Category'] == 0]['num_characters'])
sns.histplot(df[df['Category'] == 1]['num_characters'], color='red')
sns.histplot(df[df['Category'] == 0]['num_words'])
sns.histplot(df[df['Category'] == 1]['num_words'], color='red')
sns.pairplot(df, hue='Category')

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('popular')

# Text transformation function
ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [i for i in text if i.isalnum()]
    text = [i for i in text if i not in string.punctuation and i not in stopwords.words('english')]
    text = [ps.stem(i) for i in text]
    return " ".join(text)

df['transformed_text'] = df['Message'].apply(transform_text)

# Generate WordClouds
wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
spam_wc = wc.generate(df[df['Category'] == 1]['transformed_text'].str.cat(sep=" "))
plt.imshow(spam_wc)
plt.show()
ham_wc = wc.generate(df[df['Category'] == 0]['transformed_text'].str.cat(sep=" "))
plt.imshow(ham_wc)
plt.show()

# Create corpus
spam_corpus = [word for msg in df[df['Category'] == 1]['transformed_text'] for word in msg.split()]

# Feature extraction
X = TfidfVectorizer(max_features=3000).fit_transform(df['transformed_text']).toarray()
y = df['Category'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Train Naive Bayes models
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

gnb.fit(X_train, y_train)
mnb.fit(X_train, y_train)
bnb.fit(X_train, y_train)

# Evaluate models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))

evaluate_model(gnb, X_test, y_test)
evaluate_model(mnb, X_test, y_test)
evaluate_model(bnb, X_test, y_test)

# Predict spam or not
def is_spam(message):
    transformed = transform_text(message)
    prediction = mnb.predict([transformed])[0]
    return "spam" if prediction == 1 else "not spam"

# Test message
msg = "[Update] Congratulations Nile Yogesh, You account is activated for investment in Stocks. Click to invest now:"
print(is_spam(msg))
