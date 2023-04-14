from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import joblib
import datetime
import os


# # test data
# from sklearn.datasets import fetch_20newsgroups

# data = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))
# docs = data["data"]
# categories = data["target"]
# category_names = data["target_names"]
# print(docs[1], type(docs[1]))

#prepared data upload and concat annotations
path1 = r'C:\Users\thiel\OneDrive\Desktop\AIceberg\indkat-14-03.csv'
path2 = r'C:\Users\thiel\OneDrive\Desktop\AIceberg\indkat-07-03.csv'
df1= pd.read_csv(path1, sep=';')
df2 = pd.read_csv(path2, sep=';')
#print(df1.iloc[1, 2] , type(df1.iloc[1, 2]))
data = pd.concat([df1, df2])

# #ungelabelte comments aussortieren
# for i in range(len(data)):
#     print( data.iloc[i][2] , type(data.iloc[i][1]))
#     if data.iloc[i][1] == []:
#         data.drop(i)
# print(data.info())

# docs ist data , y ist datalabel
docs = list(df1['text']) + list(df2['text'])
y= list(df1['indikatoren'] ) + list(df2['indikatoren'])
#indicators = data['indikatoren']

empty_dimensionality_model = BaseDimensionalityReduction()
clf = LogisticRegression()
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
topic_model = BERTopic( embedding_model='all-MiniLM-L6-v2')
# topic_model = BERTopic(umap_model=empty_dimensionality_model,
#         hdbscan_model=clf,
#         ctfidf_model=ctfidf_model)
topics, probs = topic_model.fit_transform(docs)
print(topic_model.get_topic_info(), topic_model.get_topic(0) , topic_model.get_representative_docs(0))

# #save model
# now = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M')
# filename = os.path.join( f'{now}_trained_topicmodel.pkl')
# joblib.dump(topic_model, filename)