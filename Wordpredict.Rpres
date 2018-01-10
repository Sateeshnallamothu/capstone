Next Word Prediction Process
========================================================
author: Sateesh Nallamothu
date: Nov 11, 2017
autosize: true

Problem Description and source data
========================================================
Natural Language Processing(NLP) is heavily relied on Machine learning algorithms and Statistical Learning Methods. These algorithms and methods take a large set of 'features' as inputs that are generated from input data block. To predict the 'next word', we'll use the dataset provided by SwiftKey corporation. Following are data characteristics and assumptions.

- The dataset is a zip file including blog posts, new articles and Twitter tweets. Here are some of the statistics about of data/corpus.
 <small> en_US.twitter.txt with 30373583 words, en_US.blogs.txt with 3733413 words and en_US.news.txt with 2643969 words  </small>   
- Since the data is too big to process with my laptop, I'm using 10% of random data from each file.



N-gram model using Markov Assumption
========================================================
Bag-of-words and N-gram model are most commonly used methods to predict the next word in a sentence.
<small>
- N-gram model depends on knowledge of word sequence from (N-1) prior words. This Model is widely used in NLP processing for word prediction.
- The intuition of the N-gram model is the chain rule of the conditional probabilities. Chain rule computing the conditional probability P(a,b,c,d) = P(a)P(b|a)P(c|a,b)P(d|a,b,c) .
- Computing the conditional probabilities and multiplying them is a lot of process. So we simplify this using Markov assumption. 
- We assume the probability of next word only depends on its previous word or previous 2 words (or N-1 words). This is called Markov assumption which helps us predicting the future word without looking too far into the past
- So P(w20|w1,w2.....w18,w19) will become P(w20|w18,w19). 
</small>

Smoothing using Kneser-Ney process
========================================================
<small>
- The N-gram probabilities are estimated by using Maximum Likelihood Estimate (MLE) assumption which uses counts and normalization process. i.e P(word2|word1) = count(Word1,word2)/count(word1).
- N-gram model is a static model and it depends on the training corpus. The model will become better and better as we increase the N.
- But not all words can be accommodated in a model. In order to handle the words that are not in training but in test sentence, we use 'smoothing or discount techniques'.
- There are wide variety smoothing techniques like add-1(laplace), add-k, Stupid Backoff and Kneser-Ney smoothing.
-One of the most commonly used and the best performing is Kneser-Ney smoothing.
</small>
 
 
Model building
========================================================
<small>
- As text characters consume more space, I've converted the words into numeric values (i.e word to index) and counted the N-gram based on the numeric representation of the words.  
- Python is rich with a lot of NLP processing packages to clean, create N-grams, and compute Kneser-Ney smoothing etc,. I pulled a hack job and used Python code to create N-grams and Kneser-Ney data. The Python output data was used in R for further processing in the prediction algorithm.. 
-  Steps to predict the next word: 1) From input text, take two and/or one word/token, search the 3-gram Kneser-Ney table. If one or more matches found, then the algorithm outputs the top predictions for the next word. If no match is found, search the bi-gram table using the last word from the input. If no match is found, the prediction will then use random 4 tokens from one-gram table.
- Shiny app  link <https://sknallamothu.shinyapps.io/NextWordPrediction>
</small>

