import tomotopy as tp
import pickle
from gensim import models, corpora
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
from networkx.drawing.nx_agraph import graphviz_layout

print(tp.isa)  # prints 'avx2', 'avx', 'sse2' or 'none'

corpus = corpora.MmCorpus("./Topic_Modeling/Input_Data/corpus.mm")
dictionary = corpora.dictionary.Dictionary.load_from_text("./Topic_Modeling/Input_Data/dictionary.dict")

with open('bigram_2d_matrix', 'rb') as fp:
    text_list = pickle.load(fp)


iterations = 100
seed = 100
tw = tp.TermWeight.IDF
alpha = 0.3
eta = 0.6
gamma = 0.15
depth = 3
rm_top = 1


mdl = tp.HLDAModel(seed=seed, tw=tw, alpha=alpha, eta=eta, gamma=gamma, depth=depth, rm_top=rm_top)
for i in text_list:
    mdl.add_doc(words=i)
#start_time = time.time()

mdl.burn_in = 10000
mdl.train(0)
print('Num docs:', len(mdl.docs), ', Vocab size:', mdl.num_vocabs,
      ', Num words:', mdl.num_words)
print('Removed top words:', mdl.removed_top_words)

file_name = "tree " + "burn_in" + str(mdl.burn_in) + " iterations " + str(iterations) + " seed " + str(
    seed) + " tw " + str(tw) + " alpha " + str(alpha) + " eta " + str(eta) + " gamma " + str(gamma) + " depth " + str(
    depth) + " rm_top " + str(rm_top)


def make_graph(sub_iteration):
    tree = nx.DiGraph()
    name = mdl.get_topic_words(0, top_n=3)
    concat_name_root = name[0][0] + ", " + name[1][0] + ", " + name[2][0]
    tree.add_node(concat_name_root)

    children = mdl.children_topics(0)
    for topic in children:
        name_of_topic = mdl.get_topic_words(topic, top_n=3)
        concat_name_topic = name_of_topic[0][0] + ", " + name_of_topic[1][0] + ", " + name_of_topic[2][0]
        tree.add_edge(concat_name_root, concat_name_topic)
        for sub_topic in mdl.children_topics(topic):
            name_of_sub_topic = mdl.get_topic_words(sub_topic, top_n=3)
            concat_name_sub_topic = name_of_sub_topic[0][0] + ", " + name_of_sub_topic[1][0] + ", " + \
                                    name_of_sub_topic[2][
                                        0]
            tree.add_edge(concat_name_topic, concat_name_sub_topic)


    nx.write_graphml(tree, "./Bigram_Trees/" + "iteration " + str(sub_iteration) + file_name + ".graphml")


for i in range(0, 1000, 100):
    mdl.train(iter=iterations, workers=4)
    print('Iteration: {}\tLog-likelihood: {}'.format(i, mdl.ll_per_word))

    make_graph(i)
    mdl.summary()
    mdl.save("./Bigram_Trees/" + file_name + ".bin")

#elapsed_time = time.time() - start_time








