from gensim import models, utils, corpora
import glob
import pandas as pd
import nltk
import numpy
import re
from gensim.models import CoherenceModel
import pickle
from gensim.corpora import Dictionary
import csv

def custom_tokenize(text):
    text.str.strip()
    return nltk.word_tokenize(text)

def Tokenize_CSVs():
    # glob.glob('data*.csv') - returns List[str]
    # pd.read_csv(f) - returns pd.DataFrame()
    # for f in glob.glob() - returns a List[DataFrames]
    # pd.concat() - returns one pd.DataFrame()

    df = pd.concat([pd.read_csv(f) for f in glob.glob('*.csv')], ignore_index=True)
    df.to_csv('testConcat.csv')
    #df = df.dropna()
    #df = pd.concat([pd.read_csv(f) for f in glob.glob('./Patent_texts_filtered_test/*.csv')], ignore_index = True)
    df = df.replace({'\t': '', '\n': '', '  ': ' '}, regex=True)
    df.to_csv('testCleaned.csv')

    #df['tokenized_rows'] = df.apply(lambda row: nltk.word_tokenize(row['Title\tAbstract\tClaims']), axis=1)
    df['tokenized'] = df.apply(lambda x: nltk.word_tokenize(x['Title\tAbstract\tClaims']), axis=1)
    #df['tokenized_rows'] = df.column.apply(custom_tokenize())
    #print(df.isna().values.any())
    list = []
    #list = df['tokenized_rows'].values.tolist()

    df['tokenized'].to_csv('testTokenized.csv')

    print(df['tokenized'])

    list = df['tokenized'].to_numpy().tolist()


    with open('outfile', 'wb') as fp:
        pickle.dump(list, fp)

    #df = df.to_string(index=False)
    #df = df.replace('\n', ' ')
    #df = df.replace('  ', '')
    #df = df.apply(nltk.word_tokenize(text='str'))

    print('[%s]' % '\n'.join(map(str, list)))

    #if re.search("\t", df):
    #    print("found")
    #df.to_csv('test.csv')
    #texts_tokenized = df.to_list

def Search_Texts():
    with open('outfile', 'rb') as fp:
        list = pickle.load(fp)
    for text in list:
        for word in text:
            if word == "VSRomegaprimkpkVSRTCR":
                print("VSRomegaprimkpkVSRTCR found")
                print(text)
                return
    print("vsromegaprimkpkvsrtcr not found")


def Calc_Coherence():
    with open('outfile', 'rb') as fp:
        list = pickle.load(fp)

    print('[%s]' % '\n'.join(map(str, list)))
    print(len(list))


    numpy.seterr(divide='warn', invalid='warn')
    model = models.LdaModel.load('./model/Topic_Model_83')
    dictionary = corpora.dictionary.Dictionary.load_from_text("./dict/dictionary.dict")
    corpus = [dictionary.doc2bow(text) for text in list]
    #corpus = corpora.MmCorpus("./corpus/corpus.mm")
    cm = CoherenceModel(model=model, texts=list, coherence='c_v', dictionary=dictionary, topn=10, processes=1)
    coherence = cm.get_coherence()
    print(coherence)






#model = models.LdaModel.load("Topic_Model_51")
#cm = CoherenceModel(model=model, texts=texts_tokenized, coherence='c_v')
#print(cm.get_coherence())





