from gensim import models, corpora
import logging
from time import time
import numpy
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
t0 = time()

hyperparameter_alpha = "auto"                                    #Hat Einfluss auf die Document to Topic Probability Matrix (je höher desto extremer ist die Verteilung)
hyperparameter_beta = "auto"                                       #Hat Einfuss auf die Topic to Word Probablility Matrix (je höher desto extremer ist die Verteilung)
Anzahl_Iterationen = 20000                                               #Je höher der Wert, desto genauer werden die Ergebnisse des Modells / Ein allgemeingültiger Wert existiert nicht
Minimale_Wahrscheinlichkeit = 0                                         #Bestimmt die Wahrscheinlichkeitsgrenze für die Document to Topic Zuordnung. Der Wert Null liefert das gesamte Spektrum an Zuordnungen. Default is 0.01 laut gensim.models.ldamodel.py
corpus = corpora.MmCorpus("./Topic_Modeling/Input_Data/corpus.mm")                                     #Speicherpfad der Corpus-Datei im .mm-Format
dictionary = corpora.dictionary.Dictionary.load_from_text("./Topic_Modeling/Input_Data/dictionary.dict")     #Speicherpfad der Dictionary-Datei im .dict-Format
min=201
max=300
step=1
   
for topics in range(min,max+1,step):
    lda = models.ldamodel.LdaModel(random_state = 1, alpha= hyperparameter_alpha, eta= hyperparameter_beta, corpus = corpus, id2word = dictionary, num_topics =topics, passes = 20, iterations = Anzahl_Iterationen, minimum_probability = Minimale_Wahrscheinlichkeit, eval_every=1)
    lda.save("./Topic_Modeling/Models/Topic_Model_%i" %topics)
    del lda

print("\n\nTime needed: %i seconds.\n\n" % (time() - t0))