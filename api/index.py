import os
from flask import Flask
from flask import request, jsonify, flash, redirect, url_for
import requests
from os import getcwd
import re
import string
import numpy as np
import pandas as pd
import nltk
import json
from sklearn import preprocessing
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './tmp'
ALLOWED_EXTENSIONS = {'csv', 'tsv'}


app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
upload_folder_path = app.config['UPLOAD_FOLDER']

arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
english_punctuations = string.punctuation
punctuations_list = arabic_punctuations + english_punctuations

def remove_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)

def ensure_string_without_bytes(input_string, encoding="utf-8"):
    result_string = ""
    for element in input_string:
        if isinstance(element, bytes):
            # If the element is bytes, decode it to a string using the specified encoding
            element = element.decode(encoding)
        result_string += element
    return result_string

def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text

def remove_repeating_char(text):
    return re.sub(r'(.)\1+', r'\1', text)

def process_tweet(tweet):
    """Process tweet function.
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet

    """
    #stemmer = PorterStemmer()
    stopwords_arabic = stopwords.words('arabic')
    #ensure encoding
    #tweet = ensure_string_without_bytes(tweet, encoding="utf-8")
    #Replace @username with empty string
    tweet = re.sub('@[^\s]+', ' ', tweet)
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)


    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)

    # Replace #word with word method two
    #tweet = re.sub(r'#([^\s]+)', r'\1', tweet)

    # remove punctuations
    tweet= remove_punctuations(tweet)

    # normalize the tweet
    tweet= normalize_arabic(tweet)

    # remove repeated letters
    tweet=remove_repeating_char(tweet)

    # tokenize tweets english
    # tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
    #                           reduce_len=True)

    # Arabic tokenizer easy
    tokenizer = RegexpTokenizer(r'\w+')
    tweet_tokens = tokenizer.tokenize(tweet)

    # Arabic tokenizer using transformers
    #tokenizer = AutoTokenizer.from_pretrained('akhooli/xlm-r-large-arabic-sent')
    #model = AutoModelForSequenceClassification.from_pretrained("akhooli/xlm-r-large-arabic-sent")

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_arabic and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            tweets_clean.append(word)
            # Dont stem until figuring out how to do it in arabic of its good to do it in arabic
            #stem_word = stemmer.stem(word)  # stemming word
            #tweets_clean.append(stem_word)

    return tweets_clean

def build_freqs(tweets, ys):
    """Build frequencies.
    Input:
        tweets: a list of tweets
        ys: an m x 1 array with the sentiment label of each tweet
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its
        frequency
    """
    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element.
    # Also note that this is just a NOP if ys is already a list.
    yslist = np.squeeze(ys).tolist()
    # Start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet.
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs

def build_freqs_json(tweets, ys):
    """Build frequencies.
    Input:
        tweets: a list of tweets
        ys: an m x 1 array with the sentiment label of each tweet
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its
        frequency
    """
    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element.
    # Also note that this is just a NOP if ys is already a list.
    yslist = np.squeeze(ys).tolist()
    # Start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet.
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            if word in freqs:
                freqs[word] += 1
            else:
                freqs[word] = 1

    return freqs

def sigmoid(z):
    '''
    Input:
        z: is the input (can be a scalar or an array)
    Output:
        h: the sigmoid of z
    '''

    ### START CODE HERE ###
    # calculate the sigmoid of z
    h = 1/(1+np.exp(-z))
    ### END CODE HERE ###

    return h

def gradientDescent(x, y, theta, alpha, num_iters):
    '''
    Input:
        x: matrix of features which is (m,n+1)
        y: corresponding labels of the input matrix x, dimensions (m,1)
        theta: weight vector of dimension (n+1,1)
        alpha: learning rate
        num_iters: number of iterations you want to train your model for
    Output:
        J: the final cost
        theta: your final weight vector
    Hint: you might want to print the cost to make sure that it is going down.
    '''
    ### START CODE HERE ###
    # get 'm', the number of rows in matrix x
    m = x.shape[0]
    c = 0
    for i in range(0, num_iters):

        # get z, the dot product of x and theta
        z = np.dot(x, theta)

        # get the sigmoid of z
        h = [[sigmoid(i[0])] for i in z]
        # calculate the cost function
        J = -1.0/m*(np.dot(y.T,np.log(h))+np.dot((np.ones((m,1))-y).T,np.log(np.ones((m,1))-h)))

        # update the weights theta
        theta = theta-1.0*alpha/m*(x.T@(h-y))

    ### END CODE HERE ###
    J = float(J)
    return J, theta

def extract_features(tweet, freqs, process_tweet=process_tweet):
    '''
    Input:
        tweet: a string containing one tweet
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
    Output:
        x: a feature vector of dimension (1,3)
    '''
    # process_tweet tokenizes, stems, and removes stopwords
    word_l = process_tweet(tweet)

    # 3 elements for [bias, positive, negative] counts
    x = np.zeros(3)

    # bias term is set to 1
    x[0] = 1

    ### START CODE HERE ###

    # loop through each word in the list of words
    for word in word_l:
        # increment the word count for the positive label 1
        x[1] += freqs.get((word,1),0)

        # increment the word count for the negative label 0
        x[2] += freqs.get((word,0),0)

    ### END CODE HERE ###

    x = x[None, :]  # adding batch dimension for further processing
    assert(x.shape == (1, 3))
    return x

def predict_tweet(tweet, freqs, theta):
    '''
    Input:
        tweet: a string
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
        theta: (3,1) vector of weights
    Output:
        y_pred: the probability of a tweet being positive or negative
    '''
    ### START CODE HERE ###

    # extract the features of the tweet and store it into x
    x = extract_features(tweet, freqs)

    # make the prediction using x and theta
    y_pred = sigmoid(x@theta)

    ### END CODE HERE ###

    return y_pred

def test_logistic_regression(test_x, test_y, freqs, theta, predict_tweet=predict_tweet):
    """
    Input:
        test_x: a list of tweets
        test_y: (m, 1) vector with the corresponding labels for the list of tweets
        freqs: a dictionary with the frequency of each pair (or tuple)
        theta: weight vector of dimension (3, 1)
    Output:
        accuracy: (# of tweets classified correctly) / (total # of tweets)
    """

    ### START CODE HERE ###

    # the list for storing predictions
    y_hat = []

    for tweet in test_x:
        # get the label prediction for the tweet
        y_pred = predict_tweet(tweet, freqs, theta)

        if y_pred > 0.5:
            # append 1.0 to the list
            y_hat.append(1.0)
        else:
            # append 0 to the list
            y_hat.append(0.0)

    # With the above implementation, y_hat is a list, but test_y is (m,1) array
    # convert both to one-dimensional arrays in order to compare them using the '==' operator
    accuracy = sum(y_hat == test_y.squeeze())/len(y_hat)

    ### END CODE HERE ###

    return accuracy

gql_api_url = app.config.get('GQL_API_URL', 'http://localhost:4001/')

@app.route('/', methods=['GET'])
def home():
    return '''<h1>Hello World</h1>'''

@app.route("/api/python")
def hello_world():
    return "<p>Lets do some sentiment analysis! woowie something may have gone wrong</p>"


@app.route('/api/nlp/logistic', methods=['POST'])
def compute_gradient():
    if request.method == 'POST':
        sentiment_lang = request.args.get('lang', 'ar')
        print(sentiment_lang)

        #Get Data
        nltk.download('stopwords')
        if 'Arabic_tweets_negative' not in request.files or 'Arabic_tweets_positive' not in request.files:
            print(request.files)
            app.logger.error('An error occurred', request.files)
            return redirect('/api/python')
        # if 'Arabic_tweets_negative' not in request.files or 'Arabic_tweets_positive' not in request.files:
        #     print('no files')
        #     app.logger.error('An error occurred', request.files)
        #     return redirect('/api/python')
        arabic_negative_tweets_file = request.files['Arabic_tweets_negative']
        arabic_positive_tweets_file = request.files['Arabic_tweets_positive']
        f1name = secure_filename(arabic_negative_tweets_file.filename)
        f2name = secure_filename(arabic_positive_tweets_file.filename)
        
        arabic_negative_tweets_file.save(os.path.join(app.root_path, upload_folder_path, f1name))
        arabic_positive_tweets_file.save(os.path.join(app.root_path, upload_folder_path, f2name))
        filePath = f"{getcwd()}/../tmp2/"
        nltk.data.path.append(filePath)


        pd.set_option('display.max_colwidth', 280)
        all_negative_tweets = pd.read_csv(os.path.join(app.root_path, upload_folder_path, f1name), sep="\t", header=None, index_col=False)
        all_positive_tweets = pd.read_csv(os.path.join(app.root_path, upload_folder_path, f2name), sep="\t", header=None, index_col=False)

        t_positive_tweets = []
        t_negative_tweets = []

        for index, row in all_positive_tweets.iterrows():
            if(index < 5000):
                t_positive_tweets.append(row.values[0])
        for index, row in all_negative_tweets.iterrows():
            if(index < 5000):
                t_negative_tweets.append(row.values[0])


        train_pos = t_positive_tweets[:4000]
        train_neg = t_negative_tweets[:4000]

        train_x = train_pos + train_neg
        train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
    
        freqs = build_freqs(train_x, train_y)
        

        np.random.seed(1)
        # X input is 10 x 3 with ones for the bias terms
        # tmp_X = np.append(np.ones((10, 1)), np.random.rand(10, 2) * 2000, axis=1)
        # # Y Labels are 10 x 1
        # tmp_Y = (np.random.rand(10, 1) > 0.35).astype(float)
        # # Apply gradient descent
        # tmp_J, tmp_theta = gradientDescent(tmp_X, tmp_Y, np.zeros((3, 1)), 1e-8, 700)

        # Check extract features

        # collect the features 'x' and stack them into a matrix 'X'
        X = np.zeros((len(train_x), 3))
        for i in range(len(train_x)):
            X[i, :]= extract_features(train_x[i], freqs)

        # training labels corresponding to X
        Y = train_y


        # Apply gradient descent
        J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1500)
        # J = 0.6895861535016615
        # theta = [[-5.81385551e-09],[ 7.28341025e-05],[ 4.46689466e-07]]
        # print(J, theta)
        freqs_json_dict = build_freqs_json(train_x, train_y)
        json_data = json.dumps(freqs_json_dict, indent = 4, ensure_ascii=False)
        flattened_theta = [item for sublist in theta for item in sublist]
        graphql_query = '''
            mutation createGradientDescent($input: createGradientDescentJobInput!) {
                createGradientDescentJob(input: $input) {
                    cost
                    frequencies
                    id
                    sources
                    theta
                }
            }
        '''        
        graphql_request_data = {
            'query': graphql_query,
            'variables': {
                "input":{
                    "cost": J,
                    "frequencies": json_data,
                    "theta": flattened_theta,
                    "sources": [f1name, f2name],
                    "version": "NLP_GD_AR_0"
                }
            }
        }
        try:
            response = requests.post(gql_api_url, json=graphql_request_data)

            if response.status_code == 200:
                res_data = response.json()
                jsonify(res_data)
            else:
                return f'Error: {response.status_code}', response.status_code
        except requests.exceptions.RequestException as e:
            return str(e), 500  # Return an error message and 500 status code
        # The cost after training is 0.68958615.
        # The resulting vector of weights is [-1e-08, 7.283e-05, 4.5e-07]

        # print(f"The cost after training is {J:.8f}.")
        # print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(theta)]}")
        response_data = {"message": "Request processed successfully", "data": {"cost": J, "theta": flattened_theta }}
        return jsonify(response_data)
        

@app.route('/api/nlp/predict', methods=['POST'])
def predict_tweet():
    error = None
    if request.method == 'POST' and request.is_json:
        # Prepare 
        sentiment_lang = request.args.get('lang', 'ar')
        print(sentiment_lang)
        # Get data
        request_data = request.get_json()
        print(data)

        # Predict
        graphql_query = '''
            query getGDJob($input: getGradientDescentJob) {
                gradientDescentJob(input: $input) {
                    cost
                    frequencies
                    id
                    sources
                    theta
                }
            }
        '''        
        graphql_request_data = {
            'query': graphql_query,
            'variables': {
                 "input": {
                    "version": "NLP_GD_AR_0"
                }
            }
        }
        try:
            response = requests.post(gql_api_url, json=graphql_request_data)

            if response.status_code == 200:
                data = response.json()
                jsonify(data)
            else:
                return f'Error: {response.status_code}', response.status_code
        except requests.exceptions.RequestException as e:
            return str(e), 500  # Return an error message and 500 status code
        
        tweet_text = request_data.get('text')
        theta = data.get('theta')
        freqs = data.get('frequencies')
        new_theta = np.array(theta).reshape(3,1)
        y_hat = predict_tweet(tweet_text, freqs, new_theta) 
        # Update sentiment in gql API
        tweet_id = request_data.get('text')
        graphql_query = '''
            mutation Mutation($input: updateSentimentInput!) {
                updateSentiment(input: $input) {
                    sentiment
                    id
                    text
                }
            }
        '''        
        graphql_request_data = {
            'query': graphql_query,
            'variables': {
                "input":{
                    "id": tweet_id,
                    "sentiment": y_hat
                }
            }
        }

        try:
            response = requests.post(gql_api_url, json=graphql_request_data)

            if response.status_code == 200:
                data = response.json()
                return jsonify(data)
            else:
                return f'Error: {response.status_code}', response.status_code
        except requests.exceptions.RequestException as e:
            return str(e), 500  # Return an error message and 500 status code

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5328)