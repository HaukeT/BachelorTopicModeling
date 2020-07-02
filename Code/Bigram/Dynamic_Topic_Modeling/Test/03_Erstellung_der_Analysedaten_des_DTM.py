from gensim import corpora, utils, matutils
import pandas
import numpy as np
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from time import time

t0 = time()

Speicherplatz_DTM_Modelle = "\\"
Speicherplatz_Auswertungen = "\\"
dateiname_model1 = "DTM_20_Topics.model"
model1 = utils.SaveLoad.load(Speicherplatz_DTM_Modelle + dateiname_model1)
dateiname_corpus1 = "Unigramm_Korpus_2000_bis_2013.mm"
corpus1 = corpora.MmCorpus(dateiname_corpus1)


def Terms_Over_Time(solely_terms=True, term_prob=True):
    # Bag-of-words of topics ohne und mit Topic-Term-Probability
    # solely_terms = True bedeutet, dass eine ExcelDatei nur mit den Wörtern erstellt wird
    # für term_prob = True wird unter den Wörtern zusätzlich die Term-Topic-Wahrscheinlichkeit eingefügt
    if solely_terms is True:
        writer1 = pandas.ExcelWriter(Speicherplatz_Auswertungen + "Terms_Over_Time_%i.xlsx" % (model1.num_topics))
    if term_prob is True:
        writer2 = pandas.ExcelWriter(
            Speicherplatz_Auswertungen + "Terms_Over_Time_With_Probability_%i.xlsx" % (model1.num_topics))
    topic_num = np.arange(model1.num_topics)
    for topic1 in topic_num:
        topics = []
        topics_formatted1 = []
        for time1 in range(0, model1.num_time_slices):
            topic = model1.topic_chains[topic1].e_log_prob
            topic = np.transpose(topic)
            topic = np.exp(topic[time1])
            topic = topic / topic.sum()
            bestn = matutils.argsort(topic, 10, reverse=True)
            beststr1 = [model1.id2word[id_] for id_ in bestn]
            beststr2 = [topic[id_] for id_ in bestn]
            topics.append(beststr1)
            topics_formatted1.append(beststr2)
        if solely_terms is True:
            pandas.DataFrame(topics).to_excel(writer1, sheet_name="Topic " + str(topic1 + 1))
        if term_prob is True:
            formatted1 = pandas.DataFrame(topics)
            blank_row = formatted1.append(pandas.DataFrame([np.nan]), ignore_index=True)
            formatted2 = blank_row.append(pandas.DataFrame(topics_formatted1))
            pandas.DataFrame(formatted2).to_excel(writer2, sheet_name="Topic " + str(topic1 + 1))
    if solely_terms is True:
        writer1.save()
    if term_prob is True:
        writer2.save()


def Doc_Topic():
    # Doc_Topic_Distribution
    Speicherort_Doc_Topic = Speicherplatz_Auswertungen + "Doc_Topic_Distribution_%i.xlsx" % (model1.num_topics)
    docnum_list = []
    writer3 = pandas.ExcelWriter(Speicherort_Doc_Topic)
    for i in range(len(corpus1)):
        doctop = model1.doc_topics(doc_number=i)
        docnum_list.append(doctop)
    pandas.DataFrame(docnum_list).to_excel(writer3)
    writer3.save()
    del docnum_list




def Topic_Term(complete=False, mean_values=True):
    # Topic_Term_Distribution in Zeitscheiben eingeteilt
    topic_term = []
    if complete is True:
        # Wenn die Tabelle so groß ist muss man zip64 benutzen
        writer4 = pandas.ExcelWriter(
            Speicherplatz_Auswertungen + 'Topic_Term_Distribution_%i.xlsx' % (model1.num_topics), engine='xlsxwriter')
        writer4.book.use_zip64()
    if mean_values is True:
        writer5 = pandas.ExcelWriter(
            Speicherplatz_Auswertungen + 'Mean_Topic_Term_Distribution_%i.xlsx' % (model1.num_topics), engine='xlsxwriter')
        writer5.book.use_zip64()
    mean_topic_term = []
    for time2 in range(len(model1.time_slice)):
        top_term = [np.exp(np.transpose(chain.e_log_prob)[time2]) / np.exp(np.transpose(chain.e_log_prob)[time2]).sum()
                    for k, chain in enumerate(model1.topic_chains)]
        topic_term.extend(top_term)
        top_mean = np.mean(a=top_term, axis=0)
        mean_topic_term.append(top_mean)
    vocab = [model1.id2word[i] for i in range(0, len(model1.id2word))]
    if complete is True:
        # Unverrechnete Tabelle (SEHR GROß!!)
        x0 = pandas.DataFrame(data=topic_term, columns=vocab)
        x1 = x0.transpose()
        pandas.DataFrame(data=x1).to_excel(writer4)
        writer4.save()
    if mean_values is True:
        # Mittelwert
        mean_x0 = pandas.DataFrame(data=mean_topic_term, columns=vocab)
        mean_x1 = mean_x0.transpose()
        pandas.DataFrame(data=mean_x1).to_excel(writer5)
        writer5.save()


Topic_Term(complete=True, mean_values=True)
Doc_Topic()
Terms_Over_Time(solely_terms=True, term_prob=True)

sekunden = time() - t0
print("Benötigte Zeit: %0.3f Sekunden" % (sekunden))
