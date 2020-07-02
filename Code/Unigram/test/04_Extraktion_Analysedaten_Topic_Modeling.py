"""Offene ToDos: https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/distance_metrics.ipynb

"""

from gensim import models, corpora
import pandas
import logging
import numpy
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import time
t0 = time.time()
import csv

#time.sleep(10000)

Topicanzahl = 83
model1 =  models.LdaModel.load("./Topic_Modeling/Models/Topic_Model_%i" %Topicanzahl) 
corpus1 = corpora.MmCorpus("./Topic_Modeling/Input_Data/corpus.mm")
dictionary1 = corpora.dictionary.Dictionary.load_from_text("./Topic_Modeling/Input_Data/dictionary.dict")

def topics(num_terms=10):
    anzahl_terme = num_terms
    topics1 = [[word for rank, (word, prob) in enumerate(words)]
                          for topic_id, words in model1.show_topics(formatted=False, num_words=anzahl_terme, num_topics=model1.num_topics)]
    topics1_speicherpfad = "./Topic_Modeling/Results/topics_%i.csv" % model1.num_topics
    topicnumber=[]
    for i in range(1, model1.num_topics+1,1):
        topicnumber.append("Topic " + str(i))
    wordnumber=[]
    for i in range(1,anzahl_terme+1,1):
        wordnumber.append("Term " + str(i))
    
    pandas.DataFrame(topics1, index=topicnumber, columns= wordnumber).to_csv(topics1_speicherpfad, sep=';', decimal=',')
    print("\n\nOverview of topics processed!\n\n")

def doc_topic():
    topic_over_doc = [dict(model1[x]) for x in corpus1]
    output_topic_over_doc_speicherpfad = "./Topic_Modeling/Results/doc_topic_probability_%i.csv" % model1.num_topics
    topicnumber=[]
    for i in range(1, model1.num_topics+1,1):
        topicnumber.append("Topic " + str(i))
    documents=[]
    with open("./Topic_Modeling/Results/dokumentenliste.csv", 'r')as infile:
        reader = csv.reader(infile)
        for row in reader:
            documents.append(row[0])

    data=pandas.DataFrame(data=topic_over_doc, index = documents)
    pandas.DataFrame(data=data).to_csv(output_topic_over_doc_speicherpfad, sep=";", decimal=',', header = topicnumber)
    print("\n\nDocument topic matrix processed!\n\n")
    
def topic_terms():
    topic_terms = model1.state.get_lambda() 
    topic_terms_proba = numpy.apply_along_axis(lambda x: x/x.sum(),1,topic_terms)
    term_topic_proba = numpy.matrix.transpose(topic_terms_proba)
    words = [model1.id2word[i] for i in range(topic_terms_proba.shape[1])]
    matrix_similarity_word_over_topic = "./Topic_Modeling/Results/topic_word_probability_%i.csv" % model1.num_topics
    matrix_similarity_topic_over_word = "./Topic_Modeling/Results/word_topic_probability_%i.csv" % model1.num_topics
    topicnumber=[]
    for i in range(1, model1.num_topics+1,1):
        topicnumber.append("Topic " + str(i))
        
    pandas.DataFrame(topic_terms_proba, index= topicnumber, columns=words).to_csv(matrix_similarity_word_over_topic, sep = ';', decimal=',')
    print("\n\nTopic term matrix processed!\n\n")
    
    pandas.DataFrame(term_topic_proba, index= words, columns=topicnumber).to_csv(matrix_similarity_topic_over_word, sep = ';', decimal=',')
    print("\n\nTerm topic matrix processed!\n\n")

def pyldavis():
    """Visualisierung ähnlich wie MDS
    
    To read about the methodology behind pyLDAvis, see `the original
    paper <http://nlp.stanford.edu/events/illvi2014/papers/sievert-illvi2014.pdf>`__,
    which was presented at the `2014 ACL Workshop on Interactive Language
    Learning, Visualization, and
    Interfaces <http://nlp.stanford.edu/events/illvi2014/>`__ in Baltimore
    on June 27, 2014."""
    
    import pyLDAvis.gensim
    pyLDAvis_data =  pyLDAvis.gensim.prepare(model1,corpus1, dictionary1, sort_topics=False)
    pyLDAvis.save_html(pyLDAvis_data, './Topic_Modeling/Results/pyldavis_%i.html' % model1.num_topics)
    print("\n\npyLDAvis processed!\n\n")
    
def topic_distance():
    """Topicunähnlichkeit/Topicdistanz"""
    import pprint
    words=1000
    distance='hellinger'
    mdiff, annotation = model1.diff(model1, distance=distance, num_words=words)  
    anzahl_terme = 5
    topics1 = [[word for rank, (word, prob) in enumerate(words)]
                          for topic_id, words in model1.show_topics(formatted=False, num_words=anzahl_terme, num_topics=model1.num_topics)]
    topics1 = [['|'.join(str(e) for e in list1)] for list1 in topics1]
    data=pandas.DataFrame(data=mdiff)
    pandas.DataFrame(data=data).to_csv('./Topic_Modeling/Results/Topic_Distance_%s_%i.csv' %(distance,words), header=topics1,  sep=";", decimal=',')
    print("\n\nTopic distance matrix with distance measure: %s processed!\n\n" %distance)
    
    
def fit(Input_files= "../02_Patent_data/PatFT/AN Miele/txt_preprocessed/", semantic_structure="bigram", windowsize=4, Output_Matrix = './new_topic_prob.csv'):
        
    import os
    import nltk
    nltk.download('punkt')
    from nltk.util import skipgrams
    Output_Matrix = Output_Matrix
    semantic_structure = semantic_structure
    
    #Fenstergroesse auswaehlen, mindestens semantic_structure + 2
    windowsize = windowsize
    
    if semantic_structure == "bigram": 
        ngram=2
    elif semantic_structure == "trigram":
        ngram=3
    elif semantic_structure == "unigram":
        ngram=1
    
    new_texts = []
    new_documents = []
    Input_files = "../02_Patent_data/PatFT/AN Miele/txt_preprocessed/"
    for files in os.listdir(Input_files):
        new_documents.append(files)
        with open(Input_files + files) as f:
             new_texts.append(f.read())
    
    a=pandas.DataFrame(data=new_documents)
    pandas.DataFrame(a).to_csv("./new_dokumentenliste.csv", sep=';', header=False, index=False) 
    del a
    
    
    with open("stop_words_conducted.txt" ,"r") as stopwordlist:
        list1 = stopwordlist.read().split('\n')
        stoplist = [i for i in list1 if i != ' ']
    
    stoplist = set(stoplist)
    
    new_texts = [[word for word in document.lower().split() if word not in stoplist or len(word) < 3] for document in new_texts]
    
    if not semantic_structure == "unigram": 
        new_texts = [list(skipgrams(line,ngram,windowsize-2)) for line in new_texts]
    """Bigramm Funktion"""
    if semantic_structure == "bigram":  
        texts_new=[]
    
        for doc in new_texts:
            text_new  = [("{0} {1}".format(item[0], item[1])) for item in doc]       
            texts_new.append(text_new)
        
        new_texts = [*texts_new]
        del text_new
        del texts_new
        
    elif semantic_structure == "trigram":
        texts_new=[]
        
        for doc in new_texts:
            text_new  = [("{0} {1} {2}".format(item[0], item[1], item[2])) for item in doc]      
            texts_new.append(text_new)
            
        new_texts = [*texts_new]
        del text_new
        del texts_new
        
    new_corpus = [dictionary1.doc2bow(item) for item in new_texts]
    
    
    new_doc_topic = [model1.get_document_topics(item) for item in new_corpus]
    
    new_doc_topic2=[]
    for item2 in new_doc_topic:
        new_text_topic = []
        for item in item2:
            new_text_topic.append(item[1])
        new_doc_topic2.append(new_text_topic)
    
    topicnumber=["Topic " + str(i) for i in range(1, model1.num_topics+1,1)]
    new_data = pandas.DataFrame(data=new_doc_topic2, index = new_documents)
    pandas.DataFrame(data=new_data).to_csv(Output_Matrix, sep=";", decimal = ',', header = topicnumber)

        
topics(num_terms=20)
doc_topic()
topic_terms()
#topic_distance()
pyldavis()



print("\n\nTime needed: %i seconds.\n\n" % (time.time() - t0))