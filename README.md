# Cognitive Distortions
Project work by: Yisroel Len
## Problem Statement
As of 2019 [the number of psychological issues among teens is on the rise](https://www.mentalhealthamerica.net/issues/state-mental-health-america) and disturbing, to say the least. The purpose of this project is to help working therapists save time and increase effectiveness by identifying the cognitive distortions that are so common amongst people with depression, anxiety, anger management and many other basic disorders. The [categories of distortions](http://www.pacwrc.pitt.edu/curriculum/313_MngngImpctTrmtcStrssChldWlfrPrfssnl/hndts/HO15_ThnkngAbtThnkng.pdf) that will be used are the ones described by Dr. David Burns in his book, ["Feeling Good, the new mood therapy"](https://feelinggood.com/). This is a well known and well researched categorization of the possible errors in thinking, that is relied upon by psychologists in Cognitive Behavioral Therapy. I will attempt to collect as many possible examples of negative "thoughts" or "thoughts" that indicate negative emotions. These will then be used to attempt to predict irrationality in an actual therapy session. This sort of identification would ideally be used as a method for helping therapists discern areas for future analysis. It would highlight and return the relevant sections to the therapist and client for further investigation. To be clear I'm not going to identify which irrationality it is, rather simply whether or not it is irrational.

## Executive Summary
When I first started this project I was aiming to find examples of irrational thoughts to train my models, however I was unable to find enough to build anything reliable. During this exploration process I did manage to do a couple of things that would be useful later on. First I found a large amount of data for classifying emotion and negativity. Second I did manage to assemble a small collection of irrational thoughts which I thought would be useful to test my final model on. Going back to the first point, I have some background in Cognitive Behavioral Therapy's techniques and methodology. I know that in general negativity results in greater levels of irrationality. So essentially, I gathered two types of data. One: Whether the sentence being spoken was negative at all, regardless of the speaker's actual emotions. Two: Whether or not there was negative emotion in the speaker's sentence, which is more specific. All the data I gathered is in the table below with their relevant links:

**Training Data**

|Basic Description|Source|
|:---|:---|
|Positive and Negative sentences. |[Kaggle Positive and Negative Sentences](https://www.kaggle.com/chaitanyarahalkar/positive-and-negative-sentences)|
|1,600,000 tweets examined based on emoticons on a negative-positive scale of 0-4. |[Tweet Emoticon Emotional Classifier](https://www.kaggle.com/kazanova/sentiment140)|
|An emotional analysis dataset that was gathered manually.|[Sentiment Analysis: Emotion in Text from DataWorld](https://data.world/crowdflower/sentiment-analysis-in-text)|
|A second emotional analysis dataset. |[Emotion Classification](https://www.kaggle.com/eray1yildiz/emotion-classification)|
|An emotional analysis based on words. |[Word Based Emotion Classification](https://www.kaggle.com/iwilldoit/emotions-sensor-data-set)|

**Testing Data**

|Basic Description|Source|
|:---|:---|
|A short back and forth between a therapist and client|[Interpretations of a Counselour in Training](http://counselingexaminer.org/counseling-transcription-interpretations-of-a-counselor-in-training/#more-2487)|
|A full therapy session with the therapist's commentary|[Cognitive Behavioral Therapy Transcript]|(https://www.psychotherapy.net/data/uploads/5113d623c0a74.pdf)|
|Examples of irrational and rational thoughts|[Put Together by Yisroel Len](https://docs.google.com/spreadsheets/d/1nbTu0bUTqk0kv-lAE-Yvt4aI8XSbaolIbv9mNpgkCII/edit?usp=sharing)|

Since irrationality is something that requires context I wanted to train my data with larger n_grams and attempt to capture that "context". However I knew there was going to be a big issue with this whole idea in the first place, which I will describe and then address. I'm using negativity data to predict irrationality. That means my model will predict negativity and potentially shoot many completely off shots into the water. Meaning it will falsely predict that something is irrational very frequently when in fact it's just negative (High rate of False Negatives). Solution: Build several models that will make predictions, give them a weight based on how accurate they are at forecasting, and create a final prediction based off those weights. The formula that I will use for this is similar to the one used in the AdaBoosting models. Thanks to my instructor, Boom Devahastin Na Ayudhya, for helping me nail the mathematical part of this down. The models that I'm going to use will be Logistic Regression, Naive Bayes, and a Recurrent Neural Network. Recurrent Neural Networks are good at getting 'context' in sequential data while Logistic Regression and Naive Bayes are more simple.
