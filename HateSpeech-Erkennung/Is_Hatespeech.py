import pandas as pd
import numpy as np
from googletrans import Translator
import joblib
from timeit import default_timer as timer
import sys
sys.path.append(r"C:\Users\thiel\OneDrive\Desktop\AIceberg\detoxify")
from detoxify.detoxify import (


    Detoxify,
    multilingual_toxic_xlm_r,
    toxic_albert,
    toxic_bert,
    unbiased_albert,
    unbiased_toxic_roberta,
)
from transformers import (
    AlbertForSequenceClassification,
    BertForSequenceClassification,
    RobertaForSequenceClassification,
    XLMRobertaForSequenceClassification,
)


CLASSES = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit",
] 

#hardrule Liste importieren

hardrules=pd.read_csv('/../Data/hardrules.csv', sep=";")
hardrules = list(hardrules[hardrules.iloc[:, 1] == 1].iloc[: , 0])
hardrules = [ item.lower() for item in hardrules]
thres = 0.1024

classifier = joblib.load('20230123162746_trained_model.pkl')

def is_hatespeech(query):
    translator = Translator()
    translated= translator.translate(query, dest='en', scr='de')
    model = Detoxify("unbiased")
    batched_result = model.predict([translated.text])
    #results = {'text_de': query, 'text_en': translated.text, **batched_result}
    resultsdf = pd.DataFrame.from_dict(batched_result)
    resultsarray= np.array(resultsdf.iloc[0, [ 2, 3, 4]])
    HSresults = classifier.predict_proba([resultsarray])
    if (HSresults[0 , 1] >= thres) :
         return True
    else:
        return False
        

def is_HardRules(query):
   if (query in hardrules):
       return True
   else:
       return False



# example
#print(is_HardRules('doppeldepp'))
# start = timer()
# print(is_hatespeech('Du bist ein Arschloch!'))
# end = timer()
# print(end-start)