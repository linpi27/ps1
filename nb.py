from nltk import FreqDist
import glob
import math
import re
from textblob import TextBlob

########################
### GLOBAL VARIABLES ###
########################

## hand-crafted list of stop words
stops = {"(", ")", "--","*", ":", "-", "may", "though", ";", "thing", "things", "'d", "'ll", "'m", "'ve", "'t", "'s", "'re", "a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "just", "let's", "me", "mightn't", "more", "most", "mustn't", "my", "myself", "needn't", "no", "nor", "not", "now", "o", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "should've", "shouldn't", "so", "some", "such", "t", "than", "that", "that'll", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "will", "with", "won't", "would", "wouldn", "wouldn't", "y'all", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", ",", ".", "!", "?", "'", '"', "I", "i"}

# these store the words used in positive and negative reviews
# these will be populated in read_in_training_data()
poswords = []
negwords = []

# these store the NB probability for each word in positive reviews ("positive word")
# and each word in negative reviews ("negative word")
# these will be populated in calculate_nb_probabilities()
poswordprobs = {}
negwordprobs = {}

# these store the smoothed NB probability for each word 
# these will be populated in calculate_smooth_nb_probabilities()
smooth_poswordprobs = {}
smooth_negwordprobs = {}


#################################
### FUNCTION TO READ IN DATA ###
#################################
# You don't need to modify this.
# This returns the lists of positive words 
# and negative words from the training data.

def read_in_training_data():


    ## Read in all positive reviews
    ## We create a set of unique words for each review. 
    ## We then add that set of words as a list to the master list of positive words.
    positivewords = []
    allpos = glob.glob("pos/*")
    for filename in allpos:
        f = open(filename)
        thesewords = set()
        for line in f:
            words = line.rstrip().split()
            for w in words:
                if w not in stops:
                    thesewords.add(w)
        f.close()
        positivewords.extend(list(thesewords))
    
    print(len(positivewords), "positive tokens found!")
    print(len(set(positivewords)), "positive types found!")
    
    
    ## Read in all negative reviews
    ## We create a set of unique words for each review.
    ## We then add that set of words as a list to the master list of negative words.
    negativewords = []
    allneg = glob.glob("neg/*")
    for filename in allneg:
        f = open(filename)
        thesewords = set()
        for line in f:
            words = line.rstrip().split()
            for w in words:
                if w not in stops:
                    thesewords.add(w)
        f.close()
        negativewords.extend(list(thesewords))
    
    print(len(negativewords), "negative tokens found!")
    print(len(set(negativewords)), "negative types found!")
    return(positivewords, negativewords)



######################################
### FUNCTIONS TO PREDICT SENTIMENT ###
######################################

## FUNCTION USING USER-DEFINED WORDS TO PREDICT SENTIMENT
# You just need to fill in your own keywords below.

def user_defined_keywords(reviewwords):


    #########################################
    ##### YOUR PART A CODE STARTS HERE ######
    #########################################

    # enter your keywords in the lists below
    positive_keywords = ["good", "dream", "cool", "terrific", "better", "enjoyable", "fantastic", "love", "beautiful", "legend"]
    negative_keywords = ["bad", "disaster", "not", "no", "stupid", "least", "revulsed", "worst", "wrong", "sad"]

    #########################################
    ##### YOUR PART A CODE ENDS HERE ########
    #########################################


    # If there are more positive than negative keywords,
    # return "pos". Otherwise, return "neg".

    sentiment = 0
    for w in reviewwords:
        if w in positive_keywords:
            sentiment += 1
        if w in negative_keywords:
            sentiment -=1

    if sentiment > 0:
        return "pos"

    return "neg"


## FUNCTION TO CALCULATE NAIVE BAYES PROBABILITIES 
# You will be writing most of this function.
def calculate_nb_probabilities():

    ## GOAL: Populate these two dicts, where each
    ##      key = word from poswords or negwords (created for you above)
    ##      value = NB probability for that word in that class (calculated by you here)

    poswordprobs = {}
    negwordprobs = {}

    #########################################
    ##### YOUR PART B CODE STARTS HERE ######
    #########################################

    ## Create a FreqDist for poswords below.
    pos_freq_dist = FreqDist(poswords)
    
    ## Create a FreqDist for negwords below.
    neg_freq_dist = FreqDist(negwords)

    ## Loop through your poswords FreqDist, and calculate the
    ## probability of each word in the positive class, like this:
    ## P(word|pos) = count(word) / total number of positive tokens
    ## where count(word) is what you get from the FreqDist for poswords.
    ## Store the results in poswordprobs.
    ## USE LOGS!!!
    poswordprobs = {}
    pos_token_count = sum(pos_freq_dist.values())
    for key, value in pos_freq_dist.items():
        prob_word = pos_freq_dist[key] / pos_token_count
        prob_word = math.log(prob_word)
        poswordprobs[key] = prob_word
        

    ## Now, loop through your negwords FreqDist, and calculate the
    ## probability of each word in the negative class, like this:
    ## P(word|neg) = count(word) / total number of negative tokens
    ## where count(word) is what you get from the FreqDist for negwords.
    ## Store the results in negwordprobs.
    ## USE LOGS!!!
    negwordprobs = {}
    neg_token_count = sum(neg_freq_dist.values())
    for key, value in neg_freq_dist.items():
        prob_word = neg_freq_dist[key] / neg_token_count
        prob_word = math.log(prob_word)
        negwordprobs[key] = prob_word
        

    #########################################
    ##### YOUR PART B CODE ENDS HERE ########
    #########################################

    return (poswordprobs, negwordprobs)


## FUNCTION USING NAIVE BAYES PROBS TO PREDICT SENTIMENT
# You don't need to modify this method, but it relies
# on the code you  wrote above.

def naive_bayes(reviewwords):

    # default probability for unseen words
    defaultprob = math.log(0.0000000000001)
    
    ### POSITIVE SCORE
    posscore = poswordprobs.get(reviewwords[0], defaultprob)
    for i in range(1, len(reviewwords)):
        posscore += poswordprobs.get(reviewwords[i], defaultprob)

    ### CALCULATE NEGATIVE SCORE
    negscore = negwordprobs.get(reviewwords[0], defaultprob)
    for i in range(1, len(reviewwords)):
        negscore += negwordprobs.get(reviewwords[i], defaultprob)

    if (posscore - negscore) >  0:
        return "pos"

    return "neg"



## FUNCTION TO CALCULATE SMOOTHED NAIVE BAYES PROBABILITIES 
# You will write most of this function.
def calculate_smooth_nb_probabilities():

    smooth_poswordprobs = {}
    smooth_negwordprobs = {}

    #########################################
    ##### YOUR PART C CODE STARTS HERE ######
    #########################################

    # Populate the above dictionaries just as you did in the unsmoothed
    # version, but use +1 smoothing so that you can handle unseen words.

    # +1 smoothing: when calculating the probabilities,
    # add 1 to every count found in the FreqDist for each class.
    # Divide the count by the number of types...
    #     *plus* the number of tokens for that class...
    #     *plus* 1 (for the count of the unseen word)
    
    pos_freq_dist = FreqDist(poswords)
    neg_freq_dist = FreqDist(negwords)
    
    pos_token_count = sum(pos_freq_dist.values())
    unique_pos_words = len(set(pos_freq_dist.keys()))
    for key, value in pos_freq_dist.items():
        prob_word = (pos_freq_dist[key] + 1) / (pos_token_count + unique_pos_words + 1)
        prob_word = math.log(prob_word)
        smooth_poswordprobs[key] = prob_word
    
    neg_token_count = sum(neg_freq_dist.values())
    unique_neg_words = len(set(neg_freq_dist.keys()))
    for key, value in neg_freq_dist.items():
        prob_word = (neg_freq_dist[key] + 1) / (neg_token_count + unique_neg_words + 1)
        prob_word = math.log(prob_word)
        smooth_negwordprobs[key] = prob_word
    
    # Don't forget to use logs.

    return (smooth_poswordprobs, smooth_negwordprobs)


## FUNCTION USING SMOOTHED NAIVE BAYES PROBS TO PREDICT SENTIMENT
# You will write most of this function.
def smooth_naive_bayes(reviewwords):

    # These are placeholders that allow the code to run.
    # You will calculate posscore and negscore below.
    posscore = 0
    negscore = 0

    # Adapt the code from naive_bayes() above to work here.
    # Use the smoothed probabilities you created above.

    # Do not forget to create a separate defaultprob for
    # unseen words for the two classes, as follows.

    # The defaultprob for each class should be
    # the log of:
    # 1 (the count of the unseen word) divided by...
    #    the number of types in that class...
    #    *plus* the number of tokens in that class...
    #    *plus* 1
    
    # default probability for unseen words
    pos_freq_dist = FreqDist(poswords)
    pos_token_count = sum(pos_freq_dist.values())
    unique_pos_words = len(set(pos_freq_dist.keys()))
    defaultprob_pos = math.log(1/(pos_token_count + unique_pos_words + 1))
    
    neg_freq_dist = FreqDist(negwords)
    neg_token_count = sum(neg_freq_dist.values())
    unique_neg_words = len(set(neg_freq_dist.keys()))
    defaultprob_neg = math.log(1/(neg_token_count + unique_neg_words + 1))
    
    ### POSITIVE SCORE
    posscore = poswordprobs.get(reviewwords[0], defaultprob_pos)
    for i in range(1, len(reviewwords)):
        posscore += poswordprobs.get(reviewwords[i], defaultprob_pos)

    ### CALCULATE NEGATIVE SCORE
    negscore = negwordprobs.get(reviewwords[0], defaultprob_neg)
    for i in range(1, len(reviewwords)):
        negscore += negwordprobs.get(reviewwords[i], defaultprob_neg)

    #########################################
    ##### YOUR PART C CODE ENDS HERE ########
    #########################################


    if (posscore - negscore) >  0:
        return "pos"

    return "neg"


## FUNCTION FOR APPLYING TEXTBLOB's SENTIMENT TOOL TO A REVIEW
def calculate_textblob(review):

    #########################################
    ##### YOUR PART D CODE STARTS HERE ######
    #########################################

    # First, tead the documentation for the textblob sentiment analysis here:
    # https://textblob.readthedocs.io/en/dev/quickstart.html#sentiment-analysis
    # You will not be able to do this without reading the documentation.

    # Brief instructions:
    # Create a TextBlob object.
    # Populate it to with the review (as a string not as a list of words).
    review_text_blob = TextBlob(str(review))
    
    # Get the first element of its sentiment variable.
    polarity = review_text_blob.sentiment.polarity
    
    # If it's more than the threshold 0, return pos. Otherwise return neg.
    if polarity > 0:
        return "pos"
    #########################################
    ##### YOUR PART D CODE ENDS HERE ########
    #########################################
    
    return "neg"

# Results of Various Threshold = Accuracy
# -1.00 = 
# -0.09 = 0.505
# 0.00 = 0.63
# 0.08 = 0.745
# 0.09 = 0.735
# 0.10 = 0.745
# 0.13 = 0.695
# 0.25 = 0.52
# 1.00 = 0.50

## FUNCTION FOR CALCULATING THE ACCURACY OF YOUR MODELS
# You do not need to modify this code.

def calculate_accuracy():
    keywordscorrect = 0
    nbcorrect = 0
    smnbcorrect = 0
    tbcorrect = 0
    affcorrect = 0

    # read in the test reviews
    testdata = glob.glob("test/*")
    for filename in testdata:
        wholereview = ""
        reviewwords = []
        with open(filename, encoding='utf8') as f:
            wholereview = f.read().rstrip()
        words = set(wholereview.split())
        for w in words:
            if w not in stops:
                reviewwords.append(w)
            
        # read the file name of the file to determine if its pos or neg
        filepolarity = re.sub(r"^.*?(pos|neg)-.*?$", r"\1", filename)
    
        # apply each classifier to that review, and check to see it's correct
        if filepolarity == user_defined_keywords(reviewwords):
            keywordscorrect += 1
    
        if filepolarity == naive_bayes(reviewwords):
            nbcorrect += 1
        """
        else:
            print(f"File is classified as: {filepolarity}")
            print(f"File is classified by NB as: {naive_bayes(reviewwords)}")
            print(reviewwords)
        """
        """
        if (filepolarity != naive_bayes(reviewwords)) and (filepolarity == smooth_naive_bayes(reviewwords)):
            print(f"File's correct classification is {filepolarity}")
            print(f"NB classifies it as {naive_bayes(reviewwords)}")
            print(f"Smooth NB classifies it as {smooth_naive_bayes(reviewwords)}")
            print(reviewwords)
        """
        """
        if (filepolarity != smooth_naive_bayes(reviewwords)):
            print(f"File's correct classification is {filepolarity}")
            print(f"Smooth NB classifies it as {smooth_naive_bayes(reviewwords)}")
            print(reviewwords)
        """
        if filepolarity == smooth_naive_bayes(reviewwords):
            smnbcorrect += 1

        if filepolarity == calculate_textblob(reviewwords):
            tbcorrect += 1

    # report the accuracy of each classifier
    print("User keyword accuracy: ", (keywordscorrect/200))
    print("Naive Bayes accuracy: ", (nbcorrect/200))
    print("Smoothed Naive Bayes accuracy: ", (smnbcorrect/200))
    print("TextBlob accuracy: ", (tbcorrect/200))



#####################
### RUN ALL TESTS ###
#####################

# You do not need to modify this code.

# read in the training data to get all the positive and negative words
poswords, negwords = read_in_training_data()

# calculate the naive bayes probabilities
poswordprobs, negwordprobs = calculate_nb_probabilities()

# calculate smoothed naive bayes probabilities
smooth_poswordprobs, smooth_negwordprobs = calculate_smooth_nb_probabilities()

# calculate the accuracy of all three approaches
calculate_accuracy()

