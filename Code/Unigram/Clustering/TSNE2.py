from collections import OrderedDict
import pandas as pd
import pickle as pk
from scipy import sparse as sp

def get_doc_topic_dist(model, corpus, kwords=False):
    '''
    LDA transformation, for each doc only returns topics with non-zero weight
    This function makes a matrix transformation of docs in the topic space.
    '''
    top_dist = []
    keys = []

    for d in corpus:
        tmp = {i: 0 for i in range(num_topics)}
        tmp.update(dict(model[d]))
        vals = list(OrderedDict(tmp).values())
        top_dist += [array(vals)]
        if kwords:
            keys += [array(vals).argmax()]

    return array(top_dist), keys