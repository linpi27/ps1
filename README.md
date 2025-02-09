# Problem Set 1: Implementing naive Bayes for sentiment

In this problem set, you will be experimenting with different classifiers to distinguish positive movie reviews from negative movie reviews. The classifiers are:

* A simple keyword classifier. (Remember: do not use rule-based keyword approaches like this in your project!) 
* An unsmoothed Naive Bayes classifier.
* A smoothed Naive Bayes classifier.
* A pre-trained classifier from the `textblob` library. (Don't forget to install `textblob` with `pip` or Anaconda.)

You will add, commit, and push (1) your own version of `nb.py` and (2) a PDF containing the answers to all questions beginning with Q. **This is due Friday, October 7, at 11:59pm EDT.**

### Getting started
Clone your repo down to your own machine. You will find a python program, `nb.py`, which you will be modifying. You will also see three directories of files:

* `pos`: 900 positive movie reviews, one review per file
* `neg`: 900 negative movie reviews, one review per file
* `test`: 100 negative movie reviews and 100 positive movie reviews, one review per file

Feel free to look at the files in `pos` and `neg`, but *do not look at the files in* `test`.

**Q1: Run the program `nb.py`, and report the accuracy metrics it prints out. No method has actually been implemented, so everything is just a random baseline (which, because this is a balanced dataset, is equal to a majority baseline and a stratified baseline) where everything is marked as negative. What is the random baseline accuracy of this system? (1 point)**

### Part A: User-provided keyword classifier (6 points in total)
Visually inspect the files in the `pos` and `neg` directories (but do not look at `test`!). Identify 10 words that seem strongly associated with positive reviews and 10 words that seem strongly associated with negative reviews. In `nb.py`, go to where it says `YOUR PART A CODE STARTS HERE`, and add your keywords to the two lists, `negative_keywords[]` and `positive_keywords[]`.

**Q2: Run the program `nb.py` again, and adjust your keywords until you get classification accuracy over 0.5. Report the accuracy of this classifier and report how many times you had to adjust the keyword lists to improve upon the random baseline.**

### Part B: Naïve Bayes classifier (12 points in total)
In class, we discussed using a naïve Bayes classifier for spam filtering and for word sense disambiguation. Here, you'll be building your own implementation of a naïve Bayes classifier for a real dataset of movie reviews.

Recall that in naïve Bayes, you need to consider two probabilities: (1) the probability of a word, given a positive or negative review, and (2) the prior probability of each class. The prior probability of each class in our data is the same (0.5) because the two classes have the same number of examples (900) so you can ignore the prior for this problem set.

In `nb.py`, I've provided a lot of code. For Part B, you just need to write the function that calculates the probability of a word in each of the two classes (positive and negative). Find the place in `nb.py` where it says `YOUR PART B CODE STARTS HERE`, and follow the instructions.

When you have implemented your code, run the program `nb.py` again. The Naive Bayes classifier should return an accuracy above 0.7. If it doesn't, you didn't do it correctly, so try again.


**Q3: Which has higher accuracy: your keyword classifier or the naïve Bayes classifier? Why do you think one is better than the other?**


**Q4: Write some temporary code in nb.py so that you can find three test reviews that were misclassified by either of the two classifiers, and report those reviews here.**


**Q5: Why do you think those examples were not correctly classified? What might you do to improve this?**


### Part C: Smoothed Naive Bayes Classifier (18 points)
Find the place in `nb.py` where it says `YOUR PART C CODE STARTS HERE`, and follow the instructions to implement +1 smoothing. Your accuracy should continue to improve. If it doesn't, you did something wrong, so go back and try again.

**Q6: Write some temporary code in nb.py so that you can find three reviews that were misclassified by unsmoothed Naive Bayes but correctly classified by smoothed Naive Bayes, and report those reviews here.**



**Q7: Why do you think smoothing was able to improve classification performance?**

**Q8: Find three reviews that were still classified incorrectly, even with smoothing. Write them out here, and then comment on why you think they might have been challenging for the classifiers.**

### Part D: Using the `textblob` library (6 points)
Find the place in `nb.py` where it says `YOUR PART D CODE STARTS HERE`, and follow the instructions for using the `textblob` library to determine the sentiment of a review. Once you have implemented this function, the results printed out for `textblob` accuracy will increase.

Here is a link to the `textblob` sentiment classification documentation:
http://textblob.readthedocs.io/en/dev/quickstart.html#sentiment-analysis

**Q9: The instructions in the code tell you to use a threshold of 0. Experiment with changing that threshold to other values between -1 and 1. Which values improved accuracy? Which values decreased accuracy? How could you use the training data to select a threshold? (Note that changing the threshold like this is "cheating" since you are tuning to the test set.)**

### Part E: Reporting your results
**Q10: Create a nicely formatted table of the accuracy of all of the classification options you explored in this problem: (1) random baseline, (2) user keywords, (3) naive Bayes, (4) smoothed naive Bayes, (5) `textblob` with 0 threshold, (6) `textblob` with improved threshold. (5 points)**

---

Add, commit, and push (1) your  version of `nb.py` with all of the code you implemented for parts A, B, C, and D; and (2) a PDF containing the answers to all questions beginning with Q. **This is due Friday, October 7, at 11:59pm EDT.**
