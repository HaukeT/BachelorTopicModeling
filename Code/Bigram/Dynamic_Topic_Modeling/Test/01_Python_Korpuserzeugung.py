from gensim import corpora
import csv
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


dateiname_TDM = "Finale_TDM_Unigramm_2000_2013.csv" #Name der TDM

dateiname_corpus = "Unigramm_Korpus_2000_bis_2013.mm" #Name der Corpus-Datei
dateiname_dictionary = "Unigramm_Dictionary_2000_bis_2013.dict" #Name der Dictionary-Datei

"""Fixer Code"""
ids = {} 
corpus = [] 

with open(dateiname_TDM, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=';', quotechar='|') 
    for rownumber, row in enumerate(reader): 
        for index, field in enumerate(row):
            if index == 0:
                if rownumber > 0:
                    ids[rownumber-1] = field 
            else:
                if rownumber == 0:
                    corpus.append([])
                else:
                    corpus[index-1].append((rownumber-1, int(field))) 
                                      
corpora.MmCorpus.serialize(dateiname_corpus, corpus)
corpus1 = corpora.MmCorpus(dateiname_corpus)

dictionary =  corpora.Dictionary.from_corpus(corpus = corpus1, id2word = ids)
dictionary.save_as_text(dateiname_dictionary)
del corpus
del ids