# Problem Set 1: Sentiment classification with Naive Bayes and other techniques

In this problem set, you will be experimenting with different classifiers to distinguish positive movie reviews from negative movie reviews. The classifiers are:

* A simple keyword classifier. (Remember: do not use rule-based keyword approaches like this in your project!) 
* An unsmoothed Naive Bayes classifier.
* A smoothed Naive Bayes classifier.
* A pre-trained classifier from the `textblob` library if you are running the code on your own computer. 

**This is due Friday, February 21, at 11:59pm EST.**

# YOU WILL FAIL THIS PROBLEM SET IF YOU DO NOT READ THIS README.


## Getting started: 

### Option 1
Clone your repo down to your own machine. You will find a python program, `nb.py`, which you will be modifying and running on your own computer.

### Option 2
Follow [this link to a Colab version of the `nb.py` program](https://colab.research.google.com/drive/1_gpP0qj0G130_dypwDzfA9oGzzNovGOi?usp=sharing). This is a good option if you aren't used to administering Python and managing libraries on your own computer, but it will involve a lot of "restart and run all". The notebook explains how to get the data into Colab.

**Regardless of which option you choose you are required to submit a PDF with answers to all the questions in this README!**

There are three directories of files in this repo (and in the repo you will clone on Colab, if you chose that option):

* `pos`: 900 positive movie reviews, one review per file
* `neg`: 900 negative movie reviews, one review per file
* `test`: 100 negative movie reviews and 100 positive movie reviews, one review per file

Feel free to look at the files in `pos` and `neg`, but *do not look at the files in* `test`.


**Q1: Run the program `nb.py` (or do `Run all` in Colab), and report the accuracy metrics it prints out. No method has actually been implemented, so everything is just a majority class baseline where everything is marked as negative. What is the majority class baseline accuracy of this system? (1 point)**

### Part A: User-provided keyword classifier (6 points in total)
Visually inspect the files in the `pos` and `neg` directories (but do not look at `test`!). Identify 10 words that seem strongly associated with positive reviews and 10 words that seem strongly associated with negative reviews. In `nb.py` (or in your Colab notebook), go to where it says `YOUR PART A CODE STARTS HERE`, and add your keywords to the two lists, `negative_keywords[]` and `positive_keywords[]`.

**Q2: Run the program `nb.py` again, and adjust your keywords until you get classification accuracy over 0.5. Report the accuracy of this classifier and report how many times you had to adjust the keyword lists to improve upon the random baseline.**

### Part B: Naïve Bayes classifier (12 points in total)
In class, we discussed using a naïve Bayes classifier for spam filtering and for word sense disambiguation. Here, you'll be building your own implementation of a naïve Bayes classifier for a real dataset of movie reviews.

Recall that in naïve Bayes, you need to consider two probabilities: (1) the probability of a word, given a positive or negative review, and (2) the prior probability of each class. The prior probability of each class in our data is the same (0.5) because the two classes have the same number of examples (900) so you can ignore the prior for this problem set.

In `nb.py` (and the Colab notebook), I've provided a lot of code. For Part B, you just need to write the function that calculates the probability of a word in each of the two classes (positive and negative). Find the place in `nb.py` (or the Colab notebook) where it says `YOUR PART B CODE STARTS HERE`, and follow the instructions.

When you have implemented your code, run the program `nb.py` (or in Colab, `Run all`) again. The Naive Bayes classifier should return an accuracy above 0.7. If it doesn't, you didn't do it correctly, so try again.


**Q3: Which has higher accuracy: your keyword classifier or the naïve Bayes classifier? Why do you think one is better than the other?**


**Q4: Write some temporary code in nb.py (or your Colab notebook) so that you can find three test reviews that were misclassified by either of the two classifiers, and report those reviews here.**


**Q5: Why do you think those examples were not correctly classified? What might you do to improve this?**


### Part C: Smoothed Naive Bayes Classifier (18 points)
Find the place in `nb.py` (or your Colab notebook) where it says `YOUR PART C CODE STARTS HERE`, and follow the instructions to implement +1 smoothing. Your accuracy should continue to improve. If it doesn't, you did something wrong, so go back and try again.

**Q6: Write some temporary code in nb.py (or your Colab notebook) so that you can find three reviews that were misclassified by unsmoothed Naive Bayes but correctly classified by smoothed Naive Bayes, and report those reviews here.**



**Q7: Why do you think smoothing was able to improve classification performance?**

**Q8: Find three reviews that were still classified incorrectly, even with smoothing. Write them out here, and then comment on why you think they might have been challenging for the classifiers.**

### Part D: Using the `textblob` library (6 points)
Find the place in `nb.py` (or your Colab notebook) where it says `YOUR PART D CODE STARTS HERE`, and follow the instructions for using the `textblob` library to determine the sentiment of a review. Once you have implemented this function, the results printed out for `textblob` accuracy will increase.

Here is a link to the `textblob` sentiment classification documentation:
http://textblob.readthedocs.io/en/dev/quickstart.html#sentiment-analysis

Please note that textblob is *not* a good sentiment classifier. You can use it as a baseline in your projects if you like, but it's not something you should rely on.

**Q9: The instructions in the code tell you to use a threshold of 0. Experiment with changing that threshold to other values between -1 and 1. Which values improved accuracy? Which values decreased accuracy? How could you use the training data to select a threshold? (Note that changing the threshold like this is "cheating" since you are tuning to the test set.)**

### Part E: Implementing k-Nearest Neighbors (10 points)
Find the place in `nb.py` (or your Colab notebook) where it says `YOUR PART E CODE STARTS HERE`.

### Part F: Reporting your results
**Q10: Create a nicely formatted table of the accuracy of all of the classification options you explored in this problem: (1) random baseline, (2) user keywords, (3) naive Bayes, (4) smoothed naive Bayes, (5) `textblob` with 0 threshold, (6) `textblob` with improved threshold. (5 points)**

---

Add, commit, and push (1) your  version of `nb.py` (or your Colab notebook) with all of the code you implemented for parts A, B, C, and D; and (2) a PDF containing the answers to all questions beginning with Q. If you used Colab, download and push the notebook AND share with me and the TAs. **This is due Friday, February 21 at 11:59pm EST.**
