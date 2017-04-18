import tweepy
from tweepy import OAuthHandler
from utils import get_sentiment_predictor
import numpy as np
from preprocess import W2VTransformer


class TwitterClient(object):


    def __init__(self):

        # keys and tokens from the Twitter Dev Console
        consumer_key = 'yourconsumerkeyher'
        consumer_secret = 'yourconsumersecrete here' # obtain these by registering an app on dev.twitter.com
        self.predictor = get_sentiment_predictor()

        # attempt authentication
        try:
            # create OAuthHandler object
            self.auth = OAuthHandler(consumer_key, consumer_secret)
            # set access token and secret
            # self.auth.set_access_token(access_token, access_token_secret)
            # create tweepy API object to fetch tweets
            self.api = tweepy.API(self.auth)
        except:
            print("Error: Authentication Failed")



    def get_tweet_sentiment(self, tweet):
        '''
         function to classify sentiment of passed tweet
        '''
        prediction = self.predictor([(tweet.encode('utf-8'))])
        if prediction[0][0] < 0.55 and prediction[0][0] > 0.45:
            return 'neutral'
        sentiment = ['negative','positive'][np.argmax(prediction,axis=1)[0]]
        return sentiment

    def get_tweets(self, query, count=10):
        '''
        fetch tweets and parse them.
        '''
        # empty list to store parsed tweets
        tweets = []

        try:
            # call twitter api to fetch tweets
            fetched_tweets = self.api.search(q=query, count=count,lang='en')
            print "fetched {} tweets".format(len(fetched_tweets))
            if len(fetched_tweets) == 0:
                print "No search result, please try another query!"
                return None

            # parsing tweets one by one
            for tweet in fetched_tweets:
                # empty dictionary to store required params of a tweet
                parsed_tweet = {}

                parsed_tweet['text'] = tweet.text

                parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text)

                # appending parsed tweet to tweets list
                if tweet.retweet_count > 0:
                    # if tweet has retweets, ensure that it is appended only once
                    if parsed_tweet not in tweets:
                        tweets.append(parsed_tweet)
                else:
                    tweets.append(parsed_tweet)

            return tweets

        except tweepy.TweepError as e:
            print("Error : " + str(e))
            return None


def main():
    api = TwitterClient()
    while(1):
        q = raw_input("\n\nEnter query: ")
        tweets = api.get_tweets(query=q, count=200)
        if tweets is None:
            continue

        # picking positive tweets from tweets
        ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive']
        # percentage of positive tweets
        print("Positive tweets percentage: {} %".format(100 * len(ptweets) / len(tweets)))
        # picking negative tweets from tweets
        ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']
        # percentage of negative tweets
        print("Negative tweets percentage: {} %".format(100 * len(ntweets) / len(tweets)))
        # percentage of neutral tweets
        print("Neutral tweets percentage: {} % \
            ".format(100 * (len(tweets) - len(ntweets) - len(ptweets)) / len(tweets)))

        # printing first 10 positive tweets
        print("\n\nPositive tweets:")
        for tweet in ptweets[:10]:
            print(tweet['text'])

        # printing first 10 negative tweets
        print("\n\nNegative tweets:")
        for tweet in ntweets[:10]:
            print(tweet['text'])

        c = raw_input("Try another query? (y/n)")
        if c is 'n':
            break


if __name__ == "__main__":
    main()
