import tomotopy as tp
import pickle
from gensim import models, corpora
from gensim.models import CoherenceModel


def eval_coherence(topics_dict, word_list, coherence_type='u_mass'):
    '''Wrapper function that uses gensim Coherence Model to compute topic coherence scores

    ** Inputs **
    topic_dict: dict -> topic dictionary from train_HDPmodel function
    word_list: list -> lemmatized word list of lists
    coherence_typ: str -> type of coherence value to comput (see gensim for opts)

    ** Returns **
    score: float -> coherence value
    '''

    # Build gensim objects
    vocab = corpora.Dictionary(word_list)
    corpus = [vocab.doc2bow(words) for words in word_list]

    # Build topic list from dictionary
    topic_list = []
    for k, tups in topics_dict.items():
        topic_tokens = []
        for w, p in tups:
            topic_tokens.append(w)

        topic_list.append(topic_tokens)

    # Build Coherence model
    print("Evaluating topic coherence...")
    cm = CoherenceModel(topics=topic_list, corpus=corpus, dictionary=vocab, texts=word_list,
                        coherence=coherence_type)

    score = cm.get_coherence()
    print("Done\n")
    return score


def get_hdp_topics(hdp, top_n):
    '''Wrapper function to extract topics from trained tomotopy HDP model

    ** Inputs **
    hdp:obj -> HDPModel trained model
    top_n: int -> top n words in topic based on frequencies

    ** Returns **
    topics: dict -> per topic, an arrays with top words and associated frequencies
    '''

    # Get most important topics by # of times they were assigned (i.e. counts)
    sorted_topics = [k for k, v in sorted(enumerate(hdp.get_count_by_topics()), key=lambda x: x[1], reverse=True)]

    topics = dict()

    # For topics found, extract only those that are still assigned
    for k in sorted_topics:
        if not hdp.is_live_topic(k): continue  # remove un-assigned topics at the end (i.e. not alive)
        topic_wp = []
        for word, prob in hdp.get_topic_words(k, top_n=top_n):
            topic_wp.append((word, prob))

        topics[k] = topic_wp  # store topic word/frequency array

    return topics


hlda = tp.HLDAModel.load('./Bigram_Trees/tree burn_in10000 iterations 100 seed 100 tw TermWeight.IDF alpha 0.3 eta 0.6 gamma 0.15 depth 3 rm_top 1.bin')

with open('bigram_2d_matrix', 'rb') as fp:
    text_list = pickle.load(fp)

model_topics = get_hdp_topics(hlda, 10)

print("u_mass score: " + str(eval_coherence(model_topics, text_list)))
