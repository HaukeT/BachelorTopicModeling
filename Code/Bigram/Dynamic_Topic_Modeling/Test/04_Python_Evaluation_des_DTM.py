from gensim import corpora, models, utils
import pandas
from time import time
t0 = time()
coherence_model = []
topicanzahl_liste = []

Anfangsjahr = 2000 #variable
Endjahr = 2013 #variable
start= 1 #variable
end= 100 #variable
step=1 #variable
 
for topicanzahl in range(start, end+1, step):
    topicanzahl_liste.append(topicanzahl)
    dateiname_model1 = "DTM_%i_Topics.model" % (topicanzahl)
    model1 = utils.SaveLoad.load(dateiname_model1)
    dateiname_corpus1 = "Korpus_2000_bis_2013.mm"
    corpus1 = corpora.MmCorpus(dateiname_corpus1)

    dateiname_dictionary1 = "Dictionary_2000_bis_2013.dict"
    dictionary1 = corpora.dictionary.Dictionary.load_from_text(dateiname_dictionary1)

    coherence = []
    coherence_model.append(coherence)    
    for time1 in range(0, Endjahr-Anfangsjahr+1, 1):
        topics_dtm = model1.dtm_coherence(time1)
        cm = models.CoherenceModel(topics = topics_dtm, dictionary = dictionary1, corpus = corpus1, coherence='u_mass').get_coherence()
        coherence.append(cm) 
        
jahressequenz = []
for i in range(Anfangsjahr, Endjahr+1, 1):
        jahressequenz.append(i)

x = pandas.DataFrame(data=coherence_model, columns=jahressequenz, index=topicanzahl_liste)
a = pandas.DataFrame.transpose(x)
dateiname_evaluation = "DTM_Bigramm_Evaluation_%i_%i.csv" %(start,end)
pandas.DataFrame(data = a).to_csv(dateiname_evaluation, sep=';')
print("Benötigte Zeit: %0.3fs." % (time() - t0))