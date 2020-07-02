from gensim import models, corpora
import logging
from time import time
import pandas
import csv

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
t0 = time()

pat_no = 10  # Anzahl der Patente, die in der TDM enthalten sind
dateiname_TDM = "Bigram_Small.csv"  # Name der TDM der neuen Dokumente
lda = models.LdaModel.load("Topic_Model_51")  # Laden des trainierten LDA-Models
common_dictionary = corpora.dictionary.Dictionary.load_from_text(
    "./dictionary_bigram.dict")  # Ladend es Dic des trainierten LDA
output_topic_over_doc_speicherpfad = "./Results/doc_topic_probability_%i.csv" % lda.num_topics  # Speicherpfad für Ergebnisse festlegen

"""FIXER CODE"""
# 1. Schritt: TDM in ein NxN-Array umwandeln
ifile = open(dateiname_TDM, "r")
reader = csv.reader(ifile, delimiter=";")
rownum = 0
a = []
for row in reader:
    a.append(row)
    rownum += 1
ifile.close()

# 2. Schritt: Erstelle eine Liste mit Listen. Wobei jede Liste für ein Dokument steht. In der Liste stehen alle Uni-Gramme des Doc.
other_texts = []
i = 1  # Iterator
while i <= pat_no:
    doc = []  # Liste der Liste
    k = 1
    while k < len(a):  # len(a): Anzahl der Uni-Gramme in der TDM
        variable = a[k][i]
        if variable != '0':
            doc.append(a[k][0])
        k = k + 1
    other_texts.append(doc)
    i = i + 1

# 3. Schritt: Erstelle aus den neuen Dokumenten einen Textkörper, basierend auf dem Dictionary des LDA-Modells, welches verwendet wird.
other_corpus = [common_dictionary.doc2bow(text) for text in other_texts]

# 4. Schritt: Ermittlung der Ähnlichkeiten zwischen dem neuen Dokumenten und den Topics des bestehenden Models!
topic_over_docs = []
for i in range(len(other_corpus)):
    unseen_doc = other_corpus[i]
    vector = lda[unseen_doc]
    liste = []
    for j in range(len(vector)):
        liste.append(vector[j][1])
    topic_over_docs.append(liste)

# 5. Schritt: Extraktion der Daten
pandas.DataFrame(topic_over_docs).to_csv(output_topic_over_doc_speicherpfad, sep=";")
del topic_over_docs
del other_texts

sekunden = time() - t0
print("Benötigte Zeit: %0.1f Sekunden" % (sekunden))