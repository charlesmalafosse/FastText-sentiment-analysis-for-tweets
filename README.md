# FastText sentiment analysis for tweets: A straightforward guide.
In this repository we show how to train a sentiment analysis model using fastText. (Cleaning, upsampling and sentiments for tweets)

## FastText - Shallow neural network architecture
FastText is an open source NLP library developed by facebook AI and initially released in 2016. Its goal is to provide word embedding and text classification in a efficient manner. According to their authors, it is often on par with deep learning classifiers in terms of accuracy, and many orders of magnitude faster for training and evaluation.

## Open dataset for sentiment analysis
Most open datasets for text classification are quite small and we noticed that few, if any, are available for languages other than English. 
For these reasons BetSentiment.com provides files with list of tweets and their respective sentiments in:
* English => 6.3 millions tweets available.
* Spanish => 1.2m tweets.
* French => 250 000 tweets
* Italian => 425 000 tweets
* German => 210 000 tweets
https://betsentiment.com/resources

The sentiment was generated thanks to AWS Comprehend API. For Spanish and French, tweets were first translated to English using Google Translate, and then analysed with AWS Comprehend. Sentiment is classify to either positive, negative, neutral, or mixed.

## For more info please check our article on medium
https://medium.com/@media_73863/

And check https://betsentiment.com/ for Fan Sentiment Analysis and Machine Learning applied to sports betting.

