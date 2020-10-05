import sys

import tomotopy as tp
import pickle
from gensim import models, corpora
from gensim.models import CoherenceModel

print(tp.isa)  # prints 'avx2', 'avx', 'sse2' or 'none'


with open('outfile', 'rb') as fp:
    text_list = pickle.load(fp)

hdp = tp.HDPModel(seed=1000, tw=tp.TermWeight.IDF, initial_k=10, )
for i in text_list:
    hdp.add_doc(words=i)


# Initiate sampling burn-in  (i.e. discard N first iterations)
hdp.burn_in = 10000
hdp.train(0)
print('Num docs:', len(hdp.docs), ', Vocab size:', hdp.num_vocabs,
      ', Num words:', hdp.num_words)
print('Removed top words:', hdp.removed_top_words)


# Train model
for i in range(0, 1000, 100):
    hdp.train(100)  # 100 iterations at a time
    print('Iteration: {}\tLog-likelihood: {}\tNum. of topics: {}'.format(i, hdp.ll_per_word, hdp.live_k))
    hdp.save("unigram_hdp_model.bin")

hdp.summary()


