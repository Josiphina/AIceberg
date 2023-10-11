# German Hatespeech Recognition

## Description
In this project we build a pipeline of existing well trained Neuronal Networks (NN) in order to detect hatespeech in german language. We will show both our methods how to combine these NN and the evaluation of the model that emerges from that.

For hatespeech detection we use a pipeline of :
* [googletrans](https://pypi.org/project/googletrans/) to translate german comments into English 
* [detoxify](https://github.com/unitaryai/detoxify) for English hatespeech detection, that predicts the probability (value between 0 and 1) to which the comment belongs to each of the seven categories: : toxicity, severe toxicity, obscene, identity attack, insult, threat and sexual explicit.

Furthermore, we handled the problem of language specific hatspeech by adding a function that checks if the text sequence contains an element of a list of German swear words [ Schimpfwortliste ](https://www.woltlab.com/community/thread/5044-schimpfwortliste-f%C3%BCr-die-option-zensur/).

## How to use

Use Is_hatespeech.is_hatespeech(query) from the folder "HateSeech_-Erkennung" to apply hatespeech detection model on text sequence (in German). Use Is_Hatespeech.is_hatespeech.is_HardRules(query) to check if the query is contained in the list of German swear words . There is an example provided in the Is_Hatespeech file to test run the model.

## Evaluation
We evaluated our model on a test data set consisting of 30% of the entire data. To be able to compare our results with the original binary labels hatespeech (yes/no) we trained a classifier to translate the vector of seven probabilities into a binary label. We trained the classifier on [dataset](https://github.com/valerieholtz/Hate-Speech-Detector) sampled from existing German hatespeech datasets.Details of how we have chosen the classifier model Linear Discriminant Analysis (LDA) can be found in the evaluation folder. Additionally, we provide a diagram of the sensitivity and specificity in dependency of the threshold.
