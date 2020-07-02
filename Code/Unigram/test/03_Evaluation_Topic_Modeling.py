from gensim import corpora, models
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import pandas
from time import time
t0 = time()
min=1
max=100
step=1
corpus = corpora.MmCorpus("./Topic_Modeling/Input_Data/corpus.mm")
dictionary = corpora.dictionary.Dictionary.load_from_text("./Topic_Modeling/Input_Data/dictionary.dict")
topicanzahl=[]
coherence10 = []
coherence20 = []
for i in range(min,max+1,step):
    topicanzahl.append(i)
    model = models.LdaModel.load("./Topic_Modeling/Models/Topic_Model_%i" %i)
    u_mass_10 = models.CoherenceModel(corpus = corpus, model = model, dictionary = dictionary, coherence='u_mass', topn=10).get_coherence()
    u_mass_20 = models.CoherenceModel(corpus = corpus, model = model, dictionary = dictionary, coherence='u_mass', topn=20).get_coherence()
    coherence10.append(u_mass_10)    
    coherence20.append(u_mass_20)

top10a = pandas.DataFrame(data=coherence10, index = topicanzahl)
top20a = pandas.DataFrame(data=coherence20, index = topicanzahl)

pandas.DataFrame(top10a).to_csv("./Topic_Modeling/Evaluation/UMass_Score_10_words.csv", sep=';', decimal=',')
pandas.DataFrame(top20a).to_csv("./Topic_Modeling/Evaluation/UMass_Score_20_words.csv", sep=';', decimal=',')


def coherence_umass():
    topicanzahl=[]
    coherence10 = []
    coherence20 = []
    for i in range(min,max+1,step):
        topicanzahl.append(i)
        model = models.LdaModel.load("./Topic_Modeling/Models/Topic_Model_%i" %i)
        u_mass_10 = models.CoherenceModel(corpus = corpus, model = model, dictionary = dictionary, coherence='u_mass', topn=10).get_coherence()
        u_mass_20 = models.CoherenceModel(corpus = corpus, model = model, dictionary = dictionary, coherence='u_mass', topn=20).get_coherence()
        coherence10.append(u_mass_10)    
        coherence20.append(u_mass_20)
    
    top10a = pandas.DataFrame(data=coherence10, index = topicanzahl)
    top20a = pandas.DataFrame(data=coherence20, index = topicanzahl)
    
    pandas.DataFrame(top10a).to_csv("./Topic_Modeling/Evaluation/UMass_Score_10_words.csv", sep=';', decimal=',')
    pandas.DataFrame(top20a).to_csv("./Topic_Modeling/Evaluation/UMass_Score_20_words.csv", sep=';', decimal=',')
    
def perplexity():
    topicanzahl=[]
    perplexity =[]
    logperplexity=[]
    for i in range(min,max+1,step):
        topicanzahl.append(i)
        model = models.LdaModel.load("./Topic_Modeling/Models/Topic_Model_%i" %i)    
        logperp = model.log_perplexity(corpus)
        logperplexity.append(logperp)
        perp = 2**(-logperp)
        perplexity.append(perp)
    
    a = pandas.DataFrame(data=perplexity, index = topicanzahl)
    b = pandas.DataFrame(data=logperplexity, index = topicanzahl)
    
    pandas.DataFrame(a).to_csv("./Topic_Modeling/Evaluation/Perplexity.csv", sep=';', decimal=',')
    pandas.DataFrame(b).to_csv("./Topic_Modeling/Evaluation/Log_Perplexity.csv", sep=';', decimal=',')

def coherence_cv(texts_file):
    #texts sind tokenized texts
    import csv
    texts=[]
    with open(texts_file, newline='', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='|')
        for i in reader:
             texts.append(i)  
    topicanzahl=[]
    coherence10 = []
    coherence20 = []
    for i in range(min,max+1,step):
        topicanzahl.append(i)
        model = models.LdaModel.load("./Topic_Modeling/Models/Topic_Model_%i" %i)
        c_v_10 = models.CoherenceModel(texts = texts, model = model, dictionary = dictionary, coherence='c_v', topn=10, processes=1)
        c_v_10 = c_v_10.get_coherence()
        c_v_20 = models.CoherenceModel(texts = texts, model = model, dictionary = dictionary, coherence='c_v', topn=20, processes=1)
        c_v_20 = c_v_20.get_coherence()
        coherence10.append(c_v_10)    
        coherence20.append(c_v_20)
    
    top10b = pandas.DataFrame(data=coherence10, index = topicanzahl)
    top20b = pandas.DataFrame(data=coherence20, index = topicanzahl)
    
    pandas.DataFrame(top10b).to_csv("./Topic_Modeling/Evaluation/Cv_Score_10_words.csv", sep=';', decimal=',')
    pandas.DataFrame(top20b).to_csv("./Topic_Modeling/Evaluation/Cv_Score_20_words.csv", sep=';', decimal=',')

perplexity()
coherence_umass()
#coherence_cv(texts_file='./Topic_Modeling/Input_Data/texts_for_cv.csv')

print("\n\nTime needed: %i seconds.\n\n" % (time() - t0))