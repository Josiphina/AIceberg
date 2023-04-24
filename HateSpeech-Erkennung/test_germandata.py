import sys
import numpy as np
sys.path.append(r"C:\Users\thiel\OneDrive\Desktop\AIceberg\detoxify")
import pandas as pd

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



# path: übersetzte daten ; path2: hardrules liste mit übersetzung
path = r'C:\Users\thiel\OneDrive\Desktop\AIceberg\Data\translated_comments_cleaned.csv'
df = pd.read_csv(path)
path2 = r'C:\Users\thiel\OneDrive\Desktop\AIceberg_Dateien\translated_Schimpfwortliste.csv'
Schimpfwortliste = pd.read_csv(path2)

#comments = list(df['message_english'])
#resultcomments = df.drop('message_english' , axis =1)
comments = list(Schimpfwortliste['message_english'])
Schimpfwortliste = Schimpfwortliste.drop('message_english', axis =1)

def batch(your_list, bs=1):
    l = len(your_list)
    for i in range(0, l, bs):
        yield your_list[i:min(i + bs, l)]


model = Detoxify('unbiased')

results=pd.DataFrame({'text':[], **{cl: [] for cl in model.class_names}})

cnt = 0 
for batched_text in batch(comments, 1):
    batched_result = model.predict(batched_text)
    mydf= pd.DataFrame(batched_result)
    dict_to_append = {'text': batched_text, **batched_result}
    # append result dict for each batch to csv file
    results = pd.concat([results, pd.DataFrame(dict_to_append)], ignore_index = True )

    cnt += 1

Schimpfwortliste = pd.concat([Schimpfwortliste, results], axis=1)
#resultcomments = pd.concat([resultcomments, results], axis =1 ]
#results= results[['text', 'HateSpeech', 'toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack']]

        
Schimpfwortliste.to_csv('hardrules_unbiased.csv')
#resultcomments.to_csv('results_unbiased.csv')