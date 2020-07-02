from gensim import corpora
import csv
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from time import time
t0 = time()
import pandas


dateiname_TDM = "./TDM-GM-3-1-1.csv"
dateiname_corpus = "./Topic_Modeling/Input_Data/corpus.mm" 
dateiname_dictionary = "./Topic_Modeling/Input_Data/dictionary.dict"

"""Fixer Code"""
ids = {}
corpus = []

with open(dateiname_TDM, newline='', encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile, delimiter=';', quotechar='|') 
    documente = next(reader, None)[1:]
    for rownumber, row in enumerate(reader): 
        for index, field in enumerate(row):
            if index == 0:
                if rownumber >= 0:
                    ids[rownumber] = field 
            else:
                if rownumber == 0:
                    corpus.append([])
                else:
                    try:
                        if int(field) > 0:
                            corpus[index-1].append((rownumber, int(field)))
                    except ValueError:
                        corpus[index-1].append((rownumber, 0))
                          
a=pandas.DataFrame(data=documente)
pandas.DataFrame(a).to_csv("./Topic_Modeling/Results/dokumentenliste.csv", sep=';', header=False, index=False)                                     

corpora.MmCorpus.serialize(dateiname_corpus, corpus)
corpus1 = corpora.MmCorpus(dateiname_corpus)
dictionary =  corpora.Dictionary.from_corpus(corpus = corpus1, id2word = ids)
dictionary.save_as_text(dateiname_dictionary)
del corpus
del ids
del a
#del corpus1
#del dictionary
#del documente


print("\n\nTime needed: %i seconds.\n\n" % (time() - t0))