from gensim import models, corpora
import pandas
import logging
from time import time

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

t0 = time()
Zeitabschnitte = [1, 60, 69, 173, 165, 136, 151, 157, 106, 138, 85, 100, 56, 13]  # variable, erster wert entspricht anzahl dokumente in erster zeiteinheit
dateiname_corpus = "./Topic_Modeling/Input_Data/corpus.mm"  # variable
dateiname_dictionary = "./Topic_Modeling/Input_Data/dictionary.dict"  # variable

Speicherplatz_DTM_Modelle = "./Topic_Modeling/Models/"  # variable
Speicherplatz_Evaluation_Datei = "./Topic_Modeling/Evaluation/"  # variable

LDA_trained = models.LdaModel.load("Topic_Model_51")

anfangsjahr = 2004  # variable
start = 51  # variable, anzahl topics anfangswert
limit = 51  # variable, anzahl topics endwert
step = 1  # variable, schritte zwischen den anzahlen an topics
chain_variance = 0.005  # variable; 0.005 als Standardwert im Package
dictionary1 = corpora.dictionary.Dictionary.load_from_text(dateiname_dictionary)
corpus1 = corpora.MmCorpus(dateiname_corpus)

for num_topics in range(start, limit + 1, step):
    t0 = time()
    dtmmodels = models.ldaseqmodel.LdaSeqModel(initialize='ldamodel', lda_model=LDA_trained, corpus=corpus1,
                                               time_slice=Zeitabschnitte, id2word=dictionary1,
                                               num_topics=num_topics, passes=1, chain_variance=chain_variance,
                                               lda_inference_max_iter=1, em_min_iter=1, em_max_iter=1)
    dtmmodels.save(Speicherplatz_DTM_Modelle + "DTM_%i_Topics.model" % num_topics)
    coherence = []
    time_span = []
    for time1 in range(len(Zeitabschnitte)):
        time_span.append(time1 + anfangsjahr)
        topics_dtm = dtmmodels.dtm_coherence(time1)
        cm = models.CoherenceModel(topics=topics_dtm, dictionary=dictionary1, corpus=corpus1,
                                   coherence='u_mass').get_coherence()
        coherence.append(cm)
    x = pandas.DataFrame(data=coherence, columns=["%i Topics" % num_topics], index=time_span)
    pandas.DataFrame(data=x).to_csv(
        Speicherplatz_Evaluation_Datei + "DTM_Evaluation_%i_Topics_chain_variance_0.5.csv" % (num_topics), sep=';')

    sekunden = time() - t0
    with open(Speicherplatz_Evaluation_Datei + "DTM_Simulationsdauer_%i_Topics_chain_variance_0.5.txt" % (num_topics),
              "w") as f:
        f.write("Benötigte Zeit für %i Topics: %0.3f Sekunden" % (num_topics, sekunden))
