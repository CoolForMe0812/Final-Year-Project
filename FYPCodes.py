#Necessary libraries
import os
import pathlib
import typing
import pandas as pd
from datetime import date
from typing import Optional
import snscrape.modules.twitter as sntwitter
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer  
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
import spacy, re
import emoji
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob

#Create a new file directory
folder='../FinalYearProject'
if not os.path.isdir(folder):
  pathlib.Path('../FinalYearProject').mkdir(parents=True, exist_ok=True)

# Sub-File path directories for retrieving data
folder1='../FinalYearProject/Datasets'
if not os.path.isdir(folder1):
  pathlib.Path('../FinalYearProject/Datasets').mkdir(parents=True, exist_ok=True) 

# Sub-File path directories for retrieving results
folder1='../FinalYearProject/Results'
if not os.path.isdir(folder1):
  pathlib.Path('../FinalYearProject/Results').mkdir(parents=True, exist_ok=True) 

#install required libraries
os.system ('pip install -r ../FinalYearProject/requirements.txt')



# Scrape Twitter Fake News Data using Snscrape libraries
# Setting variables to be used below
maxTweets = 10000

# Creating list to append tweet data to
tweets_list2 = []

# Using TwitterSearchScraper to scrape data and append tweets to list
for i,tweet in enumerate(sntwitter.TwitterSearchScraper('#FakeNews false news since:2019-01-01 until:2022-04-10 lang:en').get_items()):
    if i>maxTweets:
        break
    tweets_list2.append([tweet.date, tweet.id, tweet.content,tweet.user.username])
    

# Creating a dataframe from the tweets list above

tweets_df2 = pd.DataFrame(tweets_list2, columns=['Date', 'Tweet Id', 'Comments','User'])

# Display first 5 entries from dataframe
tweets_df2

# Export dataframe into a CSV
tweets_df2.to_csv('../FinalYearProject/Datasets/FakeNewsTweets.csv', sep=',', index=False)
  
#Read original data tweets csv
df=pd.read_csv('../FinalYearProject/Datasets/originalTweets.csv')
  
#Categorize timestamp into different types of time interval
def timeCategory(date):
  if  '2019-01-01'<= date <= '2019-10-30':
    return 'Pre-Covid'
  elif  '2019-11-01'<= date <= '2019-12-31':
    return 'Pre and Post Covid Overlaps'
  elif  '2020-01-01'<= date <= '2022-04-10':
    return 'Post-Covid'

# Assign variable for categorizing different time interval for Twitter fake news dataset
df ['TimeInterval'] = df['Date'].apply(timeCategory )

#Start text preprocessing using Spacy libraries
nlp=spacy.load("en_core_web_sm")

stop_words = [w.lower() for w in stopwords.words()]


#Text-preprocessing
def preprocessing(input_string):
    #Sanitize one string 

    # normalize to lowercase 
    string = input_string.lower()

    # spacy tokenizer 
    string_split = [token.text for token in nlp(string)]

    # in case the string is empty 
    if not string_split:
        return '' 

    #remove # and @
    string = re.sub("@[A-Za-z0-9_]+","", string)
    string = re.sub("#[A-Za-z0-9_]+","", string)

    # remove 'https' links
    string = re.sub(r'https?:\/\/\S*', '', string, flags=re.MULTILINE)

    # removing stop words 
    string = ' '.join([w for w in string.split() if w not in stop_words])

    return string 

 
df["cleanedComments"] = df["Comments"].apply(lambda text: preprocessing(text))

#Assign function for removing emojis

def strip_emoji(text):
    
    new_text = re.sub(emoji.get_emoji_regexp(), r"", text)
    return new_text


with open("../FinalYearProject/Datasets/originalTweets.csv", "r",encoding="utf-8") as file:
    old_text = file.read()

no_emoji_text = strip_emoji(old_text)

df["cleanedComments"] = df["cleanedComments"].apply(lambda text: strip_emoji(text))

#Removing punctuations
df["cleanedComments"] = df["cleanedComments"].str.replace('[^\w\s]', " ")

#saved cleanedtweets to one csv
df.to_csv('../FinalYearProject/Datasets/cleanedTweets.csv')

 

# Read cleanedTweets.csv
df1=pd.read_csv('../FinalYearProject/Datasets/cleanedTweets.csv')

#Convert date to datetime format with year,month, day
df1['Date'] = pd.to_datetime(df1['Date'])
df1['Date']=df1.Date.dt.strftime('%Y-%m')
 
#Creating first quarter of Topic Modelling during pre-COVID19 in Quarter 1 2019
quarter_pre1=df1.loc[df1['Date'].between('2019-01-01','2019-03-31', inclusive=True)]

# Creating a vectorizer
vectorizer = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
data_vectorized = vectorizer.fit_transform(quarter_pre1["cleanedComments"])
NUM_TOPICS = 10

# Latent Dirichlet Allocation Model
lda = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online',verbose=True)
data_lda = lda.fit_transform(data_vectorized)

#Lemmatize clean tweets for topic modelling
stop_words = [w.lower() for w in stopwords.words()]

def clean_text(headline):
      le=WordNetLemmatizer()
      word_tokens=word_tokenize(headline)
      tokens=[le.lemmatize(w) for w in word_tokens if w not in stop_words and len(w)>3]
      cleaned_text=" ".join(tokens)
      return cleaned_text
quarter_pre1['cleanedComments']=quarter_pre1['cleanedComments'].apply(clean_text)

#LDA Topic Modelling for Top 10 Results
lda_model=LatentDirichletAllocation(n_components=10,
learning_method='online',random_state=42,max_iter=1) 

tokenizer = RegexpTokenizer(r'\w+')
vect =TfidfVectorizer(lowercase=True,
                        stop_words='english',
                        ngram_range = (1,1),
                        tokenizer = tokenizer.tokenize,max_features=1000)
vect_text=vect.fit_transform(quarter_pre1['cleanedComments'])
lda_top=lda_model.fit_transform(vect_text)
vocab = vect.get_feature_names_out()
for i, comp in enumerate(lda_model.components_):
    vocab_comp = zip(vocab, comp)
    top_terms_key=sorted(vocab_comp, key = lambda x: x[1], reverse=True)[:15]
    top_terms_list=list(dict(top_terms_key).keys())
    print("Topic "+str(i)+": ",top_terms_list)


for i,topic in enumerate(lda_top[0]):
  print("Topic ",i,": ",topic*100,"%")
 
 
#Creating second quarter of Topic Modelling during pre-COVID19 in Quarter 2 2019
quarter_pre2=df1.loc[df1['Date'].between('2019-04-01','2019-06-30', inclusive=True)]

# Creating a vectorizer
vectorizer = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
data_vectorized = vectorizer.fit_transform(quarter_pre2["cleanedComments"])
NUM_TOPICS = 10

# Latent Dirichlet Allocation Model
lda = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online',verbose=True)
data_lda = lda.fit_transform(data_vectorized)

#Lemmatize clean tweets for topic modelling
stop_words = [w.lower() for w in stopwords.words()]

def clean_text(headline):
      le=WordNetLemmatizer()
      word_tokens=word_tokenize(headline)
      tokens=[le.lemmatize(w) for w in word_tokens if w not in stop_words and len(w)>3]
      cleaned_text=" ".join(tokens)
      return cleaned_text
quarter_pre2['cleanedComments']=quarter_pre2['cleanedComments'].apply(clean_text)

#LDA Topic Modelling for Top 10 Results
lda_model=LatentDirichletAllocation(n_components=10,
learning_method='online',random_state=42,max_iter=1) 

tokenizer = RegexpTokenizer(r'\w+')
vect =TfidfVectorizer(lowercase=True,
                        stop_words='english',
                        ngram_range = (1,1),
                        tokenizer = tokenizer.tokenize,max_features=1000)
vect_text=vect.fit_transform(quarter_pre2['cleanedComments'])
lda_top=lda_model.fit_transform(vect_text)
vocab = vect.get_feature_names_out()
for i, comp in enumerate(lda_model.components_):
    vocab_comp = zip(vocab, comp)
    top_terms_key=sorted(vocab_comp, key = lambda x: x[1], reverse=True)[:15]
    top_terms_list=list(dict(top_terms_key).keys())
    print("Topic "+str(i)+": ",top_terms_list)


for i,topic in enumerate(lda_top[0]):
  print("Topic ",i,": ",topic*100,"%")
 
 
#Creating third quarter of Topic Modelling during pre-COVID19 in Quarter 3 2019
quarter_pre3=df1.loc[df1['Date'].between('2019-07-01','2019-10-31', inclusive=True)]

# Creating a vectorizer
vectorizer = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
data_vectorized = vectorizer.fit_transform(quarter_pre3["cleanedComments"])
NUM_TOPICS = 10

# Latent Dirichlet Allocation Model
lda = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online',verbose=True)
data_lda = lda.fit_transform(data_vectorized)

#Lemmatize clean tweets for topic modelling
stop_words = [w.lower() for w in stopwords.words()]

def clean_text(headline):
      le=WordNetLemmatizer()
      word_tokens=word_tokenize(headline)
      tokens=[le.lemmatize(w) for w in word_tokens if w not in stop_words and len(w)>3]
      cleaned_text=" ".join(tokens)
      return cleaned_text
quarter_pre3['cleanedComments']=quarter_pre3['cleanedComments'].apply(clean_text)

#LDA Topic Modelling for Top 10 Results
lda_model=LatentDirichletAllocation(n_components=10,
learning_method='online',random_state=42,max_iter=1) 

tokenizer = RegexpTokenizer(r'\w+')
vect =TfidfVectorizer(lowercase=True,
                        stop_words='english',
                        ngram_range = (1,1),
                        tokenizer = tokenizer.tokenize,max_features=1000)
vect_text=vect.fit_transform(quarter_pre3['cleanedComments'])
lda_top=lda_model.fit_transform(vect_text)
vocab = vect.get_feature_names_out()
for i, comp in enumerate(lda_model.components_):
    vocab_comp = zip(vocab, comp)
    top_terms_key=sorted(vocab_comp, key = lambda x: x[1], reverse=True)[:15]
    top_terms_list=list(dict(top_terms_key).keys())
    print("Topic "+str(i)+": ",top_terms_list)

for i,topic in enumerate(lda_top[0]):
  print("Topic ",i,": ",topic*100,"%")
 
 
#Creating first quarter of Topic Modelling during transition of pre-COVID19 and post-COVID19 in Quarter 4 2019
quarter_transition1=df1.loc[df1['Date'].between('2019-11-01','2019-12-31', inclusive=True)]

# Creating a vectorizer
vectorizer = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
data_vectorized = vectorizer.fit_transform(quarter_transition1["cleanedComments"])
NUM_TOPICS = 10

# Latent Dirichlet Allocation Model
lda = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online',verbose=True)
data_lda = lda.fit_transform(data_vectorized)

#Lemmatize clean tweets for topic modelling
stop_words = [w.lower() for w in stopwords.words()]

def clean_text(headline):
      le=WordNetLemmatizer()
      word_tokens=word_tokenize(headline)
      tokens=[le.lemmatize(w) for w in word_tokens if w not in stop_words and len(w)>3]
      cleaned_text=" ".join(tokens)
      return cleaned_text
quarter_transition1['cleanedComments']=quarter_transition1['cleanedComments'].apply(clean_text)

#LDA Topic Modelling for Top 10 Results
lda_model=LatentDirichletAllocation(n_components=10,
learning_method='online',random_state=42,max_iter=1) 

tokenizer = RegexpTokenizer(r'\w+')
vect =TfidfVectorizer(lowercase=True,
                        stop_words='english',
                        ngram_range = (1,1),
                        tokenizer = tokenizer.tokenize,max_features=1000)
vect_text=vect.fit_transform(quarter_transition1['cleanedComments'])
lda_top=lda_model.fit_transform(vect_text)
vocab = vect.get_feature_names_out()
for i, comp in enumerate(lda_model.components_):
    vocab_comp = zip(vocab, comp)
    top_terms_key=sorted(vocab_comp, key = lambda x: x[1], reverse=True)[:15]
    top_terms_list=list(dict(top_terms_key).keys())
    print("Topic "+str(i)+": ",top_terms_list)

for i,topic in enumerate(lda_top[0]):
  print("Topic ",i,": ",topic*100,"%")
 
 
#Creating first quarter of Topic Modelling during post-COVID19 in Quarter 1 2020
quarter_post1=df1.loc[df1['Date'].between('2020-01-01','2020-03-31', inclusive=True)]

# Creating a vectorizer
vectorizer = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
data_vectorized = vectorizer.fit_transform(quarter_post1["cleanedComments"])
NUM_TOPICS = 10

# Latent Dirichlet Allocation Model
lda = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online',verbose=True)
data_lda = lda.fit_transform(data_vectorized)

#Lemmatize clean tweets for topic modelling
stop_words = [w.lower() for w in stopwords.words()]

def clean_text(headline):
      le=WordNetLemmatizer()
      word_tokens=word_tokenize(headline)
      tokens=[le.lemmatize(w) for w in word_tokens if w not in stop_words and len(w)>3]
      cleaned_text=" ".join(tokens)
      return cleaned_text
quarter_post1['cleanedComments']=quarter_post1['cleanedComments'].apply(clean_text)

#LDA Topic Modelling for Top 10 Results
lda_model=LatentDirichletAllocation(n_components=10,
learning_method='online',random_state=42,max_iter=1) 

tokenizer = RegexpTokenizer(r'\w+')
vect =TfidfVectorizer(lowercase=True,
                        stop_words='english',
                        ngram_range = (1,1),
                        tokenizer = tokenizer.tokenize,max_features=1000)
vect_text=vect.fit_transform(quarter_post1['cleanedComments'])
lda_top=lda_model.fit_transform(vect_text)
vocab = vect.get_feature_names_out()
for i, comp in enumerate(lda_model.components_):
    vocab_comp = zip(vocab, comp)
    top_terms_key=sorted(vocab_comp, key = lambda x: x[1], reverse=True)[:15]
    top_terms_list=list(dict(top_terms_key).keys())
    print("Topic "+str(i)+": ",top_terms_list)

for i,topic in enumerate(lda_top[0]):
  print("Topic ",i,": ",topic*100,"%")
 
 
#Creating second quarter of Topic Modelling during post-COVID19 in Quarter 2 2020
quarter_post2=df1.loc[df1['Date'].between('2020-04-01','2020-06-30', inclusive=True)]

# Creating a vectorizer
vectorizer = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
data_vectorized = vectorizer.fit_transform(quarter_post2["cleanedComments"])
NUM_TOPICS = 10

# Latent Dirichlet Allocation Model
lda = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online',verbose=True)
data_lda = lda.fit_transform(data_vectorized)

#Lemmatize clean tweets for topic modelling
stop_words = [w.lower() for w in stopwords.words()]

def clean_text(headline):
      le=WordNetLemmatizer()
      word_tokens=word_tokenize(headline)
      tokens=[le.lemmatize(w) for w in word_tokens if w not in stop_words and len(w)>3]
      cleaned_text=" ".join(tokens)
      return cleaned_text
quarter_post2['cleanedComments']=quarter_post2['cleanedComments'].apply(clean_text)

#LDA Topic Modelling for Top 10 Results
lda_model=LatentDirichletAllocation(n_components=10,
learning_method='online',random_state=42,max_iter=1) 

tokenizer = RegexpTokenizer(r'\w+')
vect =TfidfVectorizer(lowercase=True,
                        stop_words='english',
                        ngram_range = (1,1),
                        tokenizer = tokenizer.tokenize,max_features=1000)
vect_text=vect.fit_transform(quarter_post2['cleanedComments'])
lda_top=lda_model.fit_transform(vect_text)
vocab = vect.get_feature_names_out()
for i, comp in enumerate(lda_model.components_):
    vocab_comp = zip(vocab, comp)
    top_terms_key=sorted(vocab_comp, key = lambda x: x[1], reverse=True)[:15]
    top_terms_list=list(dict(top_terms_key).keys())
    print("Topic "+str(i)+": ",top_terms_list)

for i,topic in enumerate(lda_top[0]):
  print("Topic ",i,": ",topic*100,"%")
 
 
#Creating third quarter of Topic Modelling during post-COVID19 in Quarter 3 2020
quarter_post3=df1.loc[df1['Date'].between('2020-07-01','2020-09-30', inclusive=True)]

# Creating a vectorizer
vectorizer = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
data_vectorized = vectorizer.fit_transform(quarter_post3["cleanedComments"])
NUM_TOPICS = 10

# Latent Dirichlet Allocation Model
lda = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online',verbose=True)
data_lda = lda.fit_transform(data_vectorized)

#Lemmatize clean tweets for topic modelling
stop_words = [w.lower() for w in stopwords.words()]

def clean_text(headline):
      le=WordNetLemmatizer()
      word_tokens=word_tokenize(headline)
      tokens=[le.lemmatize(w) for w in word_tokens if w not in stop_words and len(w)>3]
      cleaned_text=" ".join(tokens)
      return cleaned_text
quarter_post3['cleanedComments']=quarter_post3['cleanedComments'].apply(clean_text)

#LDA Topic Modelling for Top 10 Results
lda_model=LatentDirichletAllocation(n_components=10,
learning_method='online',random_state=42,max_iter=1) 

tokenizer = RegexpTokenizer(r'\w+')
vect =TfidfVectorizer(lowercase=True,
                        stop_words='english',
                        ngram_range = (1,1),
                        tokenizer = tokenizer.tokenize,max_features=1000)
vect_text=vect.fit_transform(quarter_post3['cleanedComments'])
lda_top=lda_model.fit_transform(vect_text)
vocab = vect.get_feature_names_out()
for i, comp in enumerate(lda_model.components_):
    vocab_comp = zip(vocab, comp)
    top_terms_key=sorted(vocab_comp, key = lambda x: x[1], reverse=True)[:15]
    top_terms_list=list(dict(top_terms_key).keys())
    print("Topic "+str(i)+": ",top_terms_list)

for i,topic in enumerate(lda_top[0]):
  print("Topic ",i,": ",topic*100,"%")
 
 
#Creating fourth quarter of Topic Modelling during post-COVID19 in Quarter 4 2020
quarter_post4=df1.loc[df1['Date'].between('2020-10-01','2020-12-31', inclusive=True)]

# Creating a vectorizer
vectorizer = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
data_vectorized = vectorizer.fit_transform(quarter_post4["cleanedComments"])
NUM_TOPICS = 10

# Latent Dirichlet Allocation Model
lda = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online',verbose=True)
data_lda = lda.fit_transform(data_vectorized)

#Lemmatize clean tweets for topic modelling
stop_words = [w.lower() for w in stopwords.words()]

def clean_text(headline):
      le=WordNetLemmatizer()
      word_tokens=word_tokenize(headline)
      tokens=[le.lemmatize(w) for w in word_tokens if w not in stop_words and len(w)>3]
      cleaned_text=" ".join(tokens)
      return cleaned_text
quarter_post4['cleanedComments']=quarter_post4['cleanedComments'].apply(clean_text)

#LDA Topic Modelling for Top 10 Results
lda_model=LatentDirichletAllocation(n_components=10,
learning_method='online',random_state=42,max_iter=1) 

tokenizer = RegexpTokenizer(r'\w+')
vect =TfidfVectorizer(lowercase=True,
                        stop_words='english',
                        ngram_range = (1,1),
                        tokenizer = tokenizer.tokenize,max_features=1000)
vect_text=vect.fit_transform(quarter_post4['cleanedComments'])
lda_top=lda_model.fit_transform(vect_text)
vocab = vect.get_feature_names_out()
for i, comp in enumerate(lda_model.components_):
    vocab_comp = zip(vocab, comp)
    top_terms_key=sorted(vocab_comp, key = lambda x: x[1], reverse=True)[:15]
    top_terms_list=list(dict(top_terms_key).keys())
    print("Topic "+str(i)+": ",top_terms_list)

for i,topic in enumerate(lda_top[0]):
  print("Topic ",i,": ",topic*100,"%")
 
 
#Creating first quarter of Topic Modelling during post-COVID19 in Quarter 1 2021
quarter_post5=df1.loc[df1['Date'].between('2021-01-01','2021-03-31', inclusive=True)]

# Creating a vectorizer
vectorizer = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
data_vectorized = vectorizer.fit_transform(quarter_post5["cleanedComments"])
NUM_TOPICS = 10

# Latent Dirichlet Allocation Model
lda = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online',verbose=True)
data_lda = lda.fit_transform(data_vectorized)

#Lemmatize clean tweets for topic modelling
stop_words = [w.lower() for w in stopwords.words()]

def clean_text(headline):
      le=WordNetLemmatizer()
      word_tokens=word_tokenize(headline)
      tokens=[le.lemmatize(w) for w in word_tokens if w not in stop_words and len(w)>3]
      cleaned_text=" ".join(tokens)
      return cleaned_text
quarter_post5['cleanedComments']=quarter_post5['cleanedComments'].apply(clean_text)

#LDA Topic Modelling for Top 10 Results
lda_model=LatentDirichletAllocation(n_components=10,
learning_method='online',random_state=42,max_iter=1) 

tokenizer = RegexpTokenizer(r'\w+')
vect =TfidfVectorizer(lowercase=True,
                        stop_words='english',
                        ngram_range = (1,1),
                        tokenizer = tokenizer.tokenize,max_features=1000)
vect_text=vect.fit_transform(quarter_post5['cleanedComments'])
lda_top=lda_model.fit_transform(vect_text)
vocab = vect.get_feature_names_out()
for i, comp in enumerate(lda_model.components_):
    vocab_comp = zip(vocab, comp)
    top_terms_key=sorted(vocab_comp, key = lambda x: x[1], reverse=True)[:15]
    top_terms_list=list(dict(top_terms_key).keys())
    print("Topic "+str(i)+": ",top_terms_list)

for i,topic in enumerate(lda_top[0]):
  print("Topic ",i,": ",topic*100,"%")
 
 
#Creating second quarter of Topic Modelling during post-COVID19 in Quarter 2 2021
quarter_post6=df1.loc[df1['Date'].between('2021-04-01','2021-06-30', inclusive=True)]

# Creating a vectorizer
vectorizer = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
data_vectorized = vectorizer.fit_transform(quarter_post6["cleanedComments"])
NUM_TOPICS = 10

# Latent Dirichlet Allocation Model
lda = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online',verbose=True)
data_lda = lda.fit_transform(data_vectorized)

#Lemmatize clean tweets for topic modelling
stop_words = [w.lower() for w in stopwords.words()]

def clean_text(headline):
      le=WordNetLemmatizer()
      word_tokens=word_tokenize(headline)
      tokens=[le.lemmatize(w) for w in word_tokens if w not in stop_words and len(w)>3]
      cleaned_text=" ".join(tokens)
      return cleaned_text
quarter_post6['cleanedComments']=quarter_post6['cleanedComments'].apply(clean_text)

#LDA Topic Modelling for Top 10 Results
lda_model=LatentDirichletAllocation(n_components=10,
learning_method='online',random_state=42,max_iter=1) 

tokenizer = RegexpTokenizer(r'\w+')
vect =TfidfVectorizer(lowercase=True,
                        stop_words='english',
                        ngram_range = (1,1),
                        tokenizer = tokenizer.tokenize,max_features=1000)
vect_text=vect.fit_transform(quarter_post6['cleanedComments'])
lda_top=lda_model.fit_transform(vect_text)
vocab = vect.get_feature_names_out()
for i, comp in enumerate(lda_model.components_):
    vocab_comp = zip(vocab, comp)
    top_terms_key=sorted(vocab_comp, key = lambda x: x[1], reverse=True)[:15]
    top_terms_list=list(dict(top_terms_key).keys())
    print("Topic "+str(i)+": ",top_terms_list)

for i,topic in enumerate(lda_top[0]):
  print("Topic ",i,": ",topic*100,"%")
 
 
#Creating third quarter of Topic Modelling during post-COVID19 in Quarter 3 2021
quarter_post7=df1.loc[df1['Date'].between('2021-07-01','2021-9-30', inclusive=True)]

# Creating a vectorizer
vectorizer = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
data_vectorized = vectorizer.fit_transform(quarter_post7["cleanedComments"])
NUM_TOPICS = 10

# Latent Dirichlet Allocation Model
lda = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online',verbose=True)
data_lda = lda.fit_transform(data_vectorized)

#Lemmatize clean tweets for topic modelling
stop_words = [w.lower() for w in stopwords.words()]

def clean_text(headline):
      le=WordNetLemmatizer()
      word_tokens=word_tokenize(headline)
      tokens=[le.lemmatize(w) for w in word_tokens if w not in stop_words and len(w)>3]
      cleaned_text=" ".join(tokens)
      return cleaned_text
quarter_post7['cleanedComments']=quarter_post7['cleanedComments'].apply(clean_text)

#LDA Topic Modelling for Top 10 Results
lda_model=LatentDirichletAllocation(n_components=10,
learning_method='online',random_state=42,max_iter=1) 

tokenizer = RegexpTokenizer(r'\w+')
vect =TfidfVectorizer(lowercase=True,
                        stop_words='english',
                        ngram_range = (1,1),
                        tokenizer = tokenizer.tokenize,max_features=1000)
vect_text=vect.fit_transform(quarter_post7['cleanedComments'])
lda_top=lda_model.fit_transform(vect_text)
vocab = vect.get_feature_names_out()
for i, comp in enumerate(lda_model.components_):
    vocab_comp = zip(vocab, comp)
    top_terms_key=sorted(vocab_comp, key = lambda x: x[1], reverse=True)[:15]
    top_terms_list=list(dict(top_terms_key).keys())
    print("Topic "+str(i)+": ",top_terms_list)

for i,topic in enumerate(lda_top[0]):
  print("Topic ",i,": ",topic*100,"%")
 
 
#Creating fourth quarter of Topic Modelling during post-COVID19 in Quarter 4 2021
quarter_post8=df1.loc[df1['Date'].between('2021-10-01','2021-12-31', inclusive=True)]

# Creating a vectorizer
vectorizer = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
data_vectorized = vectorizer.fit_transform(quarter_post8["cleanedComments"])
NUM_TOPICS = 10

# Latent Dirichlet Allocation Model
lda = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online',verbose=True)
data_lda = lda.fit_transform(data_vectorized)

#Lemmatize clean tweets for topic modelling
stop_words = [w.lower() for w in stopwords.words()]

def clean_text(headline):
      le=WordNetLemmatizer()
      word_tokens=word_tokenize(headline)
      tokens=[le.lemmatize(w) for w in word_tokens if w not in stop_words and len(w)>3]
      cleaned_text=" ".join(tokens)
      return cleaned_text
quarter_post8['cleanedComments']=quarter_post8['cleanedComments'].apply(clean_text)

#LDA Topic Modelling for Top 10 Results
lda_model=LatentDirichletAllocation(n_components=10,
learning_method='online',random_state=42,max_iter=1) 

tokenizer = RegexpTokenizer(r'\w+')
vect =TfidfVectorizer(lowercase=True,
                        stop_words='english',
                        ngram_range = (1,1),
                        tokenizer = tokenizer.tokenize,max_features=1000)
vect_text=vect.fit_transform(quarter_post8['cleanedComments'])
lda_top=lda_model.fit_transform(vect_text)
vocab = vect.get_feature_names_out()
for i, comp in enumerate(lda_model.components_):
    vocab_comp = zip(vocab, comp)
    top_terms_key=sorted(vocab_comp, key = lambda x: x[1], reverse=True)[:15]
    top_terms_list=list(dict(top_terms_key).keys())
    print("Topic "+str(i)+": ",top_terms_list)

for i,topic in enumerate(lda_top[0]):
  print("Topic ",i,": ",topic*100,"%")
 
 
#Creating first quarter of Topic Modelling during post-COVID19 in Quarter 1 2022
quarter_post9=df1.loc[df1['Date'].between('2022-01-01','2022-04-10', inclusive=True)]

# Creating a vectorizer
vectorizer = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
data_vectorized = vectorizer.fit_transform(quarter_post9["cleanedComments"])
NUM_TOPICS = 10

# Latent Dirichlet Allocation Model
lda = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online',verbose=True)
data_lda = lda.fit_transform(data_vectorized)

#Lemmatize clean tweets for topic modelling
stop_words = [w.lower() for w in stopwords.words()]

def clean_text(headline):
      le=WordNetLemmatizer()
      word_tokens=word_tokenize(headline)
      tokens=[le.lemmatize(w) for w in word_tokens if w not in stop_words and len(w)>3]
      cleaned_text=" ".join(tokens)
      return cleaned_text
quarter_post9['cleanedComments']=quarter_post9['cleanedComments'].apply(clean_text)

#LDA Topic Modelling for Top 10 Results
lda_model=LatentDirichletAllocation(n_components=10,
learning_method='online',random_state=42,max_iter=1) 

tokenizer = RegexpTokenizer(r'\w+')
vect =TfidfVectorizer(lowercase=True,
                        stop_words='english',
                        ngram_range = (1,1),
                        tokenizer = tokenizer.tokenize,max_features=1000)
vect_text=vect.fit_transform(quarter_post9['cleanedComments'])
lda_top=lda_model.fit_transform(vect_text)
vocab = vect.get_feature_names_out()
for i, comp in enumerate(lda_model.components_):
    vocab_comp = zip(vocab, comp)
    top_terms_key=sorted(vocab_comp, key = lambda x: x[1], reverse=True)[:15]
    top_terms_list=list(dict(top_terms_key).keys())
    print("Topic "+str(i)+": ",top_terms_list)

for i,topic in enumerate(lda_top[0]):
  print("Topic ",i,": ",topic*100,"%")
 
 
 
#Sentiment Analysis
df1=pd.read_csv('../FinalYearProject/Datasets/cleanedTweets.csv')
#Assign function for sentiment score
def senti(x):
    return TextBlob(x).sentiment  

df1['senti_score'] = df1['cleanedComments'].apply(senti)

sentiment_series = df1['senti_score'].tolist()

columns = ['polarity', 'subjectivity']

df2 = pd.DataFrame(sentiment_series, columns=columns, index=df1.index)

#To classify sentiment scores based on polarity
def getAnalysis(score):
  if score < 0:
    return 'Negative'
  elif score == 0:
    return 'Neutral'
  elif score >0:
    return 'Positive'


df2 ['senti_type'] = df2['polarity'].apply(getAnalysis )

print(df2)

df3 = df1.join(df2)
df3=pd.DataFrame(df3, columns=['Date','TimeInterval','Tweet Id','User','Comments','cleanedComments','senti_score','polarity','subjectivity','senti_type'])

print(df3)

#Output final merged fake news tweets data into csv
df3.to_csv("../FinalYearProject/Datasets/Final_MergedSentimentFakeNewsTweets.csv")
 

#Read original csv and sentiment csv as well as converting date to year and month format
df=pd.read_csv('../FinalYearProject/Datasets/originalTweets.csv')
df3=pd.read_csv("../FinalYearProject/Datasets/Final_MergedSentimentFakeNewsTweets.csv")
df3['Date'] = pd.to_datetime(df3['Date'])
df3['Date']=df3.Date.dt.strftime('%Y-%m')
 
#Retrieve first quarter of pre- COVID-19's data in Quarter 1 2019
quarter_pre1=df3.loc[df['Date'].between('2019-01-01','2019-03-31', inclusive=True)]
quarter_pre1=quarter_pre1.sort_values(by='Date')
#Plotting Positive sentiments polarity for first quarter of pre-COVID-19 in Quarter 1 2019
fig, (ax1) = plt.subplots(ncols=1, figsize=(14, 4))
quarter_pre1_post=quarter_pre1[quarter_pre1['polarity'] > 0]
sns.lineplot(data=quarter_pre1_post, x='Date', y='polarity', hue='senti_type', palette='crest', ax=ax1)
ax1.set_title('Positive sentiments polarity of first quarter of pre- COVID-19 in Quarter 1 2019 (Q1- 2019 JANUARY 1 to 2019 MARCH 31)')
ax1.locator_params(axis='y', integer=True)
fig.savefig('../FinalYearProject/Results/Quarter1_Pre-COVID_Pos.pdf')
plt.show()
 
 
#Retrieve first quarter of pre- COVID-19's data in Quarter 1 2019
quarter_pre1=df3.loc[df['Date'].between('2019-01-01','2019-03-31', inclusive=True)]
quarter_pre1=quarter_pre1.sort_values(by='Date')
#Plotting Negative sentiments for first quarter of pre-COVID-19 in Quarter 1 2019
fig, (ax1) = plt.subplots(ncols=1, figsize=(14, 4))
quarter_pre1_neg=quarter_pre1[quarter_pre1['polarity'] <0]
sns.lineplot(data=quarter_pre1_neg, x='Date', y='polarity', hue='senti_type', palette='rocket', ax=ax1)
ax1.set_title('Negative sentiments polarity of first quarter of pre- COVID-19 in Quarter 1 2019 (Q1- 2019 JANUARY 1 to 2019 MARCH 31)')
ax1.locator_params(axis='y', integer=True)
fig.savefig('../FinalYearProject/Results/Quarter1_Pre-COVID_Neg.pdf')
plt.show()
 
 
#Retrieve second quarter of pre- COVID-19's data in Quarter 2 2019
quarter_pre2=df3.loc[df['Date'].between('2019-04-01','2019-06-30', inclusive=True)]
quarter_pre2=quarter_pre2.sort_values(by='Date')
#Plotting Positive sentiments polarity for second quarter of pre-COVID-19 in Quarter 2 2019
fig, (ax1) = plt.subplots(ncols=1, figsize=(14, 4))
quarter_pre2_post=quarter_pre2[quarter_pre2['polarity'] > 0]
sns.lineplot(data=quarter_pre2_post, x='Date', y='polarity', hue='senti_type', palette='crest', ax=ax1)
ax1.set_title('Positive sentiments polarity of second quarter of pre- COVID-19 in Quarter 2 2019 (Q2- 2019 APRIL 1 to 2019 JUNE 30)')
ax1.locator_params(axis='y', integer=True)
fig.savefig('../FinalYearProject/Results/Quarter2_Pre-COVID_Pos.pdf')
plt.show()
 
 
#Retrieve second quarter of pre- COVID-19's data in Quarter 2 2019
quarter_pre2=df3.loc[df['Date'].between('2019-04-01','2019-06-30', inclusive=True)]
quarter_pre2=quarter_pre2.sort_values(by='Date')
#Plotting Negative sentiments for second quarter of pre-COVID-19 in Quarter 2 2019
fig, (ax1) = plt.subplots(ncols=1, figsize=(14, 4))
quarter_pre2_neg=quarter_pre2[quarter_pre2['polarity'] <0]
sns.lineplot(data=quarter_pre2_neg, x='Date', y='polarity', hue='senti_type', palette='rocket', ax=ax1)
ax1.set_title('Negative sentiments polarity of second quarter of pre- COVID-19 in Quarter 2 2019 (Q2- 2019 APRIL 1 to 2019 JUNE 30)')
ax1.locator_params(axis='y', integer=True)
fig.savefig('../FinalYearProject/Results/Quarter2_Pre-COVID_Neg.pdf')
plt.show()
 
 
#Retrieve third quarter of pre- COVID-19's data in Quarter 3 2019
quarter_pre3=df3.loc[df['Date'].between('2019-07-01','2019-10-31', inclusive=True)]
quarter_pre3=quarter_pre3.sort_values(by='Date')
#Plotting Positive sentiments polarity for third quarter of pre-COVID-19 in Quarter 3 2019
fig, (ax1) = plt.subplots(ncols=1, figsize=(14, 4))
quarter_pre3_post=quarter_pre3[quarter_pre3['polarity'] > 0]
sns.lineplot(data=quarter_pre3_post, x='Date', y='polarity', hue='senti_type', palette='crest', ax=ax1)
ax1.set_title('Positive sentiments polarity of third quarter of pre- COVID-19 in Quarter 3 2019 (Q3- 2019 JULY 1 to 2019 OCTOBER 31)')
ax1.locator_params(axis='y', integer=True)
fig.savefig('../FinalYearProject/Results/Quarter3_Pre-COVID_Pos.pdf')
plt.show()
 
 
#Retrieve third quarter of pre- COVID-19's data in Quarter 3 2019
quarter_pre3=df3.loc[df['Date'].between('2019-07-01','2019-10-31', inclusive=True)]
quarter_pre3=quarter_pre3.sort_values(by='Date')
#Plotting Negative sentiments for third quarter of pre-COVID-19 in Quarter 3 2019
fig, (ax1) = plt.subplots(ncols=1, figsize=(14, 4))
quarter_pre3_neg=quarter_pre3[quarter_pre3['polarity'] <0]
sns.lineplot(data=quarter_pre3_neg, x='Date', y='polarity', hue='senti_type', palette='rocket', ax=ax1)
ax1.set_title('Negative sentiments polarity of third quarter of pre- COVID-19 in Quarter 3 2019 (Q3- 2019 JULY 1 to 2019 OCTOBER 31)')
ax1.locator_params(axis='y', integer=True)
fig.savefig('../FinalYearProject/Results/Quarter3_Pre-COVID_Neg.pdf')
plt.show()
 
 
#Retrieve first quarter of transition periods between pre- and post- of COVID-19's data in Quarter 4 2019
quarter_transition1=df3.loc[df['Date'].between('2019-11-01','2019-12-31', inclusive=True)]
quarter_transition1=quarter_transition1.sort_values(by='Date')
#Plotting Positive sentiments polarity for first quarter of pre-COVID-19 in Quarter 4 2019
fig, (ax1) = plt.subplots(ncols=1, figsize=(14, 4))
quarter_transition1_post=quarter_transition1[quarter_transition1['polarity'] > 0]
sns.lineplot(data=quarter_transition1_post, x='Date', y='polarity', hue='senti_type', palette='crest', ax=ax1)
ax1.set_title('Positive sentiments polarity of first quarter of transition pre- and post- of COVID-19  in Quarter 4 2019 (Q4- 2019 NOVEMBER 1 to 2019 DECEMBER 31)')
ax1.locator_params(axis='y', integer=True)
fig.savefig('../FinalYearProject/Results/Quarter1_Transition_PRE-POST-COVID_Pos.pdf')
plt.show()
 
 
#Retrieve first quarter of transition periods between pre- and post- of COVID-19's data in Quarter 4 2019
quarter_transition1=df3.loc[df['Date'].between('2019-11-01','2019-12-31', inclusive=True)]
quarter_transition1=quarter_transition1.sort_values(by='Date')
#Plotting Negative sentiments polarity first quarter of transition periods between pre- and post- of COVID-19 in Quarter 4 2019
fig, (ax1) = plt.subplots(ncols=1, figsize=(14, 4))
quarter_transition1_neg=quarter_transition1[quarter_transition1['polarity'] < 0]
sns.lineplot(data=quarter_transition1_neg, x='Date', y='polarity', hue='senti_type', palette='rocket', ax=ax1)
ax1.set_title('Positive sentiments polarity of first quarter of transition pre- and post- of COVID-19 in Quarter 4 2019 (Q4- 2019 NOVEMBER 1 to 2019 DECEMBER 31)')
ax1.locator_params(axis='y', integer=True)
fig.savefig('../FinalYearProject/Results/Quarter1_Transition_PRE-POST-COVID_Neg.pdf')
plt.show()
 
 
#Retrieve first quarter of post- COVID-19's data in Quarter 1 2020
quarter_post1=df3.loc[df['Date'].between('2020-01-01','2020-03-31', inclusive=True)]
quarter_post1=quarter_post1.sort_values(by='Date')
#Plotting Positive sentiments polarity for first quarter of post-COVID-19 in Quarter 1 2020
fig, (ax1) = plt.subplots(ncols=1, figsize=(14, 4))
quarter_post1_post=quarter_post1[quarter_post1['polarity'] > 0]
sns.lineplot(data=quarter_post1_post, x='Date', y='polarity', hue='senti_type', palette='crest', ax=ax1)
ax1.set_title('Positive sentiments polarity of first quarter of post- COVID-19 in Quarter 1 2020 (Q1- 2020 JANUARY 1 to 2020 MARCH 31)')
ax1.locator_params(axis='y', integer=True)
fig.savefig('../FinalYearProject/Results/Quarter1_Post-COVID_Pos.pdf')
plt.show()
 
 
#Retrieve first quarter of post- COVID-19's data in Quarter 1 2020
quarter_post1=df3.loc[df['Date'].between('2020-01-01','2020-03-31', inclusive=True)]
quarter_post1=quarter_post1.sort_values(by='Date')
#Plotting Negative sentiments for first quarter of post-COVID-19 in Quarter 1 2020
fig, (ax1) = plt.subplots(ncols=1, figsize=(14, 4))
quarter_post1_neg=quarter_post1[quarter_post1['polarity'] <0]
sns.lineplot(data=quarter_post1_neg, x='Date', y='polarity', hue='senti_type', palette='rocket', ax=ax1)
ax1.set_title('Negative sentiments polarity of first quarter of post- COVID-19 in Quarter 1 2020 (Q1- 2020 JANUARY 1 to 2020 MARCH 31)')
ax1.locator_params(axis='y', integer=True)
fig.savefig('../FinalYearProject/Results/Quarter1_Post-COVID_Neg.pdf')
plt.show()
 
 
#Retrieve second quarter of post- COVID-19's data in Quarter 2 2020
quarter_post2=df3.loc[df['Date'].between('2020-04-01','2020-06-30', inclusive=True)]
quarter_post2=quarter_post2.sort_values(by='Date')
#Plotting Positive sentiments polarity for second quarter of post-COVID-19 in Quarter 2 2020
fig, (ax1) = plt.subplots(ncols=1, figsize=(14, 4))
quarter_post2_post=quarter_post2[quarter_post2['polarity'] > 0]
sns.lineplot(data=quarter_post2_post, x='Date', y='polarity', hue='senti_type', palette='crest', ax=ax1)
ax1.set_title('Positive sentiments polarity of second quarter of post- COVID-19 in Quarter 2 2020 (Q2- 2020 APRIL 1 to 2020 JUNE 30)')
ax1.locator_params(axis='y', integer=True)
fig.savefig('../FinalYearProject/Results/Quarter2_Post-COVID_Pos.pdf')
plt.show()
 
 
#Retrieve second quarter of post- COVID-19's data in Quarter 2 2020
quarter_post2=df3.loc[df['Date'].between('2020-04-01','2020-06-30', inclusive=True)]
quarter_post2=quarter_post2.sort_values(by='Date')
#Plotting Negative sentiments for second quarter of post-COVID-19 in Quarter 2 2020
fig, (ax1) = plt.subplots(ncols=1, figsize=(14, 4))
quarter_post2_neg=quarter_post2[quarter_post2['polarity'] <0]
sns.lineplot(data=quarter_post2_neg, x='Date', y='polarity', hue='senti_type', palette='rocket', ax=ax1)
ax1.set_title('Negative sentiments polarity of second quarter of post- COVID-19 in Quarter 2 2020 (Q2- 2020 APRIL 1 to 2020 JUNE 30)')
ax1.locator_params(axis='y', integer=True)
fig.savefig('../FinalYearProject/Results/Quarter2_Post-COVID_Neg.pdf')
plt.show()
 
 
#Retrieve third quarter of post- COVID-19's data in Quarter 3 2020
quarter_post3=df3.loc[df['Date'].between('2020-07-01','2020-09-30', inclusive=True)]
quarter_post3=quarter_post3.sort_values(by='Date')
#Plotting Positive sentiments polarity for third quarter of post-COVID-19 in Quarter 3 2020
fig, (ax1) = plt.subplots(ncols=1, figsize=(14, 4))
quarter_post3_post=quarter_post3[quarter_post3['polarity'] > 0]
sns.lineplot(data=quarter_post3_post, x='Date', y='polarity', hue='senti_type', palette='crest', ax=ax1)
ax1.set_title('Positive sentiments polarity of third quarter of post- COVID-19 in Quarter 3 2020 (Q3- 2020 JULY 1 to 2020 SEPTEMBER 30)')
ax1.locator_params(axis='y', integer=True)
fig.savefig('../FinalYearProject/Results/Quarter3_Post-COVID_Pos.pdf')
plt.show()
 
 
#Retrieve third quarter of post- COVID-19's data in Quarter 3 2020
quarter_post3=df3.loc[df['Date'].between('2020-07-01','2020-09-30', inclusive=True)]
quarter_post3=quarter_post3.sort_values(by='Date')
#Plotting Negative sentiments for third quarter of post-COVID-19 in Quarter 3 2020
fig, (ax1) = plt.subplots(ncols=1, figsize=(14, 4))
quarter_post3_neg=quarter_post3[quarter_post3['polarity'] <0]
sns.lineplot(data=quarter_post3_neg, x='Date', y='polarity', hue='senti_type', palette='rocket', ax=ax1)
ax1.set_title('Negative sentiments polarity of third quarter of post- COVID-19 in Quarter 3 2020 (Q3- 2020 JULY 1 to 2020 SEPTEMBER 30)')
ax1.locator_params(axis='y', integer=True)
fig.savefig('../FinalYearProject/Results/Quarter3_Post-COVID_Neg.pdf')
plt.show()
 
 
#Retrieve fourth quarter of post- COVID-19's data in Quarter 4 2020
quarter_post4=df3.loc[df['Date'].between('2020-10-01','2020-12-31', inclusive=True)]
quarter_post4=quarter_post4.sort_values(by='Date')
#Plotting Positive sentiments polarity for fourth quarter of post-COVID-19 in Quarter 4 2020
fig, (ax1) = plt.subplots(ncols=1, figsize=(14, 4))
quarter_post4_post=quarter_post4[quarter_post4['polarity'] > 0]
sns.lineplot(data=quarter_post4_post, x='Date', y='polarity', hue='senti_type', palette='crest', ax=ax1)
ax1.set_title('Positive sentiments polarity of fourth quarter of post- COVID-19 in Quarter 4 2020 (Q4- 2020 OCTOBER 1 to 2020 DECEMBER 31)')
ax1.locator_params(axis='y', integer=True)
fig.savefig('../FinalYearProject/Results/Quarter4_Post-COVID_Pos.pdf')
plt.show()
 
 
#Retrieve fourth quarter of post- COVID-19's data in Quarter 4 2020
quarter_post4=df3.loc[df['Date'].between('2020-10-01','2020-12-31', inclusive=True)]
quarter_post4=quarter_post4.sort_values(by='Date')
#Plotting Negative sentiments for fourth quarter of post-COVID-19 in Quarter 4 2020
fig, (ax1) = plt.subplots(ncols=1, figsize=(14, 4))
quarter_post4_neg=quarter_post4[quarter_post4['polarity'] <0]
sns.lineplot(data=quarter_post4_neg, x='Date', y='polarity', hue='senti_type', palette='rocket', ax=ax1)
ax1.set_title('Negative sentiments polarity of fourth quarter of post- COVID-19 in Quarter 4 2020 (Q4- 2020 OCTOBER 1 to 2020 DECEMBER 31)')
ax1.locator_params(axis='y', integer=True)
fig.savefig('../FinalYearProject/Results/Quarter4_Post-COVID_Neg.pdf')
plt.show()
 
 
#Retrieve first quarter of post- COVID-19's data in Quarter 1 2021
quarter_post5=df3.loc[df['Date'].between('2021-01-01','2021-03-31', inclusive=True)]
quarter_post5=quarter_post5.sort_values(by='Date')
#Plotting Positive sentiments polarity for first quarter of post-COVID-19 in Quarter 1 2021
fig, (ax1) = plt.subplots(ncols=1, figsize=(14, 4))
quarter_post5_post=quarter_post5[quarter_post5['polarity'] > 0]
sns.lineplot(data=quarter_post5_post, x='Date', y='polarity', hue='senti_type', palette='crest', ax=ax1)
ax1.set_title('Positive sentiments polarity of first quarter of post- COVID-19 in Quarter 1 2021 (Q1- 2021 JANUARY 1 to 2021 MARCH 31)')
ax1.locator_params(axis='y', integer=True)
fig.savefig('../FinalYearProject/Results/Quarter5_Post-COVID_Pos.pdf')
plt.show()
 
 
#Retrieve first quarter of post- COVID-19's data in Quarter 1 2021
quarter_post5=df3.loc[df['Date'].between('2021-01-01','2021-03-31', inclusive=True)]
quarter_post5=quarter_post5.sort_values(by='Date')
#Plotting Negative sentiments for first quarter of post-COVID-19 in Quarter 1 2021
fig, (ax1) = plt.subplots(ncols=1, figsize=(14, 4))
quarter_post5_neg=quarter_post5[quarter_post5['polarity'] <0]
sns.lineplot(data=quarter_post5_neg, x='Date', y='polarity', hue='senti_type', palette='rocket', ax=ax1)
ax1.set_title('Negative sentiments polarity of first quarter of post- COVID-19 in Quarter 1 2021 (Q1- 2021 JANUARY 1 to 2021 MARCH 31)')
ax1.locator_params(axis='y', integer=True)
fig.savefig('../FinalYearProject/Results/Quarter5_Post-COVID_Neg.pdf')
plt.show()
 
 
#Retrieve second quarter of post- COVID-19's data in Quarter 2 2021 
quarter_post6=df3.loc[df['Date'].between('2021-04-01','2021-06-30', inclusive=True)]
quarter_post6=quarter_post6.sort_values(by='Date')
#Plotting Positive sentiments polarity for second quarter of post-COVID-19 in Quarter 2 2021
fig, (ax1) = plt.subplots(ncols=1, figsize=(14, 4))
quarter_post6_post=quarter_post6[quarter_post6['polarity'] > 0]
sns.lineplot(data=quarter_post6_post, x='Date', y='polarity', hue='senti_type', palette='crest', ax=ax1)
ax1.set_title('Positive sentiments polarity of second quarter of post- COVID-19 in Quarter 2 2021 (Q2- 2021 APRIL 1 to 2021 JUNE 30)')
ax1.locator_params(axis='y', integer=True)
fig.savefig('../FinalYearProject/Results/Quarter6_Post-COVID_Pos.pdf')
plt.show()
 
 
#Retrieve second quarter of post- COVID-19's data in Quarter 2 2021
quarter_post6=df3.loc[df['Date'].between('2021-04-01','2021-06-30', inclusive=True)]
quarter_post6=quarter_post6.sort_values(by='Date')
#Plotting Negative sentiments for second quarter of post-COVID-19 in Quarter 2 2021
fig, (ax1) = plt.subplots(ncols=1, figsize=(14, 4))
quarter_post6_neg=quarter_post6[quarter_post6['polarity'] <0]
sns.lineplot(data=quarter_post6_neg, x='Date', y='polarity', hue='senti_type', palette='rocket', ax=ax1)
ax1.set_title('Negative sentiments polarity of second quarter of post- COVID-19 in Quarter 2 2021 (Q2- 2021 APRIL 1 to 2021 JUNE 30)')
ax1.locator_params(axis='y', integer=True)
fig.savefig('../FinalYearProject/Results/Quarter6_Post-COVID_Neg.pdf')
plt.show()
 
 
#Retrieve third quarter of post- COVID-19's data in Quarter 3 2021
quarter_post7=df3.loc[df['Date'].between('2021-07-01','2021-09-30', inclusive=True)]
quarter_post7=quarter_post7.sort_values(by='Date')
#Plotting Positive sentiments polarity for third quarter of post-COVID-19 in Quarter 3 2021
fig, (ax1) = plt.subplots(ncols=1, figsize=(14, 4))
quarter_post7_post=quarter_post7[quarter_post7['polarity'] > 0]
sns.lineplot(data=quarter_post7_post, x='Date', y='polarity', hue='senti_type', palette='crest', ax=ax1)
ax1.set_title('Positive sentiments polarity of third quarter of post- COVID-19 in Quarter 3 2021 (Q3- 2021 JULY 1 to 2021 SEPTEMBER 30)')
ax1.locator_params(axis='y', integer=True)
fig.savefig('../FinalYearProject/Results/Quarter7_Post-COVID_Pos.pdf')
plt.show()
 
 
#Retrieve third quarter of post- COVID-19's data in Quarter 3 2021
quarter_post7=df3.loc[df['Date'].between('2021-07-01','2021-09-30', inclusive=True)]
quarter_post7=quarter_post7.sort_values(by='Date')
#Plotting Negative sentiments for third quarter of post-COVID-19 in Quarter 3 2021
fig, (ax1) = plt.subplots(ncols=1, figsize=(14, 4))
quarter_post7_neg=quarter_post7[quarter_post7['polarity'] <0]
sns.lineplot(data=quarter_post7_neg, x='Date', y='polarity', hue='senti_type', palette='rocket', ax=ax1)
ax1.set_title('Negative sentiments polarity of third quarter of post- COVID-19 in Quarter 3 2021 (Q3- 2021 JULY 1 to 2021 SEPTEMBER 30)')
ax1.locator_params(axis='y', integer=True)
fig.savefig('../FinalYearProject/Results/Quarter7_Post-COVID_Neg.pdf')
plt.show()
 
 
#Retrieve fourth quarter of post- COVID-19's data in Quarter 4 2021
quarter_post8=df3.loc[df['Date'].between('2021-10-01','2021-12-31', inclusive=True)]
quarter_post8=quarter_post8.sort_values(by='Date')
#Plotting Positive sentiments polarity for fourth quarter of post-COVID-19 in Quarter 4 2021
fig, (ax1) = plt.subplots(ncols=1, figsize=(14, 4))
quarter_post8_post=quarter_post8[quarter_post8['polarity'] > 0]
sns.lineplot(data=quarter_post8_post, x='Date', y='polarity', hue='senti_type', palette='crest', ax=ax1)
ax1.set_title('Positive sentiments polarity of fourth quarter of post- COVID-19 in Quarter 4 2021 (Q4- 2021 OCTOBER 1 to 2021 DECEMBER 31)')
ax1.locator_params(axis='y', integer=True)
fig.savefig('../FinalYearProject/Results/Quarter8_Post-COVID_Pos.pdf')
plt.show()
 
 
#Retrieve fourth quarter of post- COVID-19's data in Quarter 4 2021
quarter_post8=df3.loc[df['Date'].between('2021-10-01','2021-12-31', inclusive=True)]
quarter_post8=quarter_post8.sort_values(by='Date')
#Plotting fourth sentiments for fourth quarter of post-COVID-19 in Quarter 4 2021
fig, (ax1) = plt.subplots(ncols=1, figsize=(14, 4))
quarter_post8_neg=quarter_post8[quarter_post8['polarity'] <0]
sns.lineplot(data=quarter_post8_neg, x='Date', y='polarity', hue='senti_type', palette='rocket', ax=ax1)
ax1.set_title('Negative sentiments polarity of fourth quarter of post- COVID-19 in Quarter 4 2021 (Q4- 2021 OCTOBER 1 to 2021 DECEMBER 31)')
ax1.locator_params(axis='y', integer=True)
fig.savefig('../FinalYearProject/Results/Quarter8_Post-COVID_Neg.pdf')
plt.show()
 
 
#Retrieve first quarter of post- COVID-19's data in Quarter 1 2022
quarter_post9=df3.loc[df['Date'].between('2022-01-01','2022-04-10', inclusive=True)]
quarter_post9=quarter_post9.sort_values(by='Date')
#Plotting Positive sentiments polarity for first quarter of post-COVID-19 in Quarter 1 2022
fig, (ax1) = plt.subplots(ncols=1, figsize=(14, 4))
quarter_post9_post=quarter_post9[quarter_post9['polarity'] > 0]
sns.lineplot(data=quarter_post9_post, x='Date', y='polarity', hue='senti_type', palette='crest', ax=ax1)
ax1.set_title('Positive sentiments polarity of first quarter of post- COVID-19 in Quarter 1 2022 (Q1- 2022 JANUARY 1 to 2022 APRIL 10)')
ax1.locator_params(axis='y', integer=True)
fig.savefig('../FinalYearProject/Results/Quarter9_Post-COVID_Pos.pdf')
plt.show()
 
 
#Retrieve first quarter of post- COVID-19's data in Quarter 1 2022
quarter_post9=df3.loc[df['Date'].between('2022-01-01','2022-04-10', inclusive=True)]
quarter_post9=quarter_post9.sort_values(by='Date')
#Plotting Negative sentiments for first quarter of post-COVID-19 in Quarter 1 2022
fig, (ax1) = plt.subplots(ncols=1, figsize=(14, 4))
quarter_post9_neg=quarter_post9[quarter_post9['polarity'] <0]
sns.lineplot(data=quarter_post9_neg, x='Date', y='polarity', hue='senti_type', palette='rocket', ax=ax1)
ax1.set_title('Negative sentiments polarity of first quarter of post- COVID-19 in Quarter 1 2022 (Q1- 2022 JANUARY 1 to 2022 APRIL 10)')
ax1.locator_params(axis='y', integer=True)
fig.savefig('../FinalYearProject/Results/Quarter9_Post-COVID_Neg.pdf')
plt.show()
 














