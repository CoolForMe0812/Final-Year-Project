Final Year Project's Title: Twitter Social Media Users Perspective towards Fake News during Pre-, Post- COVID-19 and their Transition

This project is aimed to conduct a quarterly topic modelling and sentiment polarity on Twitter social media analysis with the hashtag(#) FakeNews and false news related contents reported on Twitter during pre- COVID-19, post- COVID-19 and transition periods between pre- and post- COVID-19.

Steps for Data Crawling, Data Preprocessing, Data Analysis and Data Visualization:
1. Crawl the Twitter Data that is related #FakeNews and false news during 2019-01-01 to 2022-04-10
2. Save it into originalTweets.csv.
3. Data preprocessing with Natural Language Processing (NLP) by removing punctuation, special characters, emojis, converting to lower case, deleting English stop words, tokenization and lemmatization.
4. Different time interval categories will be specified for pre, post of COVID-19 and transition between them for identifying the occurrence of  fake news tweets that have been scrapped, which are Pre-Covid (2019 JANUARY 1 to 2019 OCTOBER 31), Post-Covid (2020 JANUARY 1 to 2022 APRIL 10) and Pre and Post Covid Overlaps (2019 NOVEMBER 1 to 2019 DECEMBER 31). In the end, this produced a cleaned  dataset with a specified time interval category for each row of data- cleanedTweets.csv.
5. There will be a final dataset- Final_MergedSentimentFakeNewsTweets.csv will be produced as well once done categorizing its’ sentiment types based on its’ polarity on sentiment score. For sentiment polarity >0, it will be classified as positive sentiments in the senti_type’s variable. For sentiment polarity <0, it will be classified as negative sentiments in the senti_type’s variable. For sentiment polarity =0,  it will be classified as neutral sentiment in the senti_type’s variable.
6. A quarterly topic modelling analysis and visualization will be produced for pre-, post- COVID-19 and their transition periods with 3-4 months per quarter to retrieve more insightful insights into changes in the interest of topic discussion.
7. A quarterly sentiment polarity line graphs analysis and visualization will be produced for pre-, post- COVID-19 and their transition periods with 3-4 months per quarter to retrieve more insightful insights into changes in positive and negative sentiments polarity over the time.
