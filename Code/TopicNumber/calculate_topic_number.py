from IPython import get_ipython
from gensim import models, corpora
import pandas
import numpy as np
import matplotlib.pyplot as plt
from decimal import *

Topicanzahl = 83
model1 = models.LdaModel.load("./Topic_Modeling_Unigram/Models/Topic_Model_%i" % Topicanzahl)
model2 = models.LdaModel.load("./Topic_Modeling_Bigram/Models/Topic_Model_%i" % 84)


def coherence_umass(model):
    corpus1 = corpora.MmCorpus("./Topic_Modeling_Bigram/Input_Data/corpus.mm")
    dictionary1 = corpora.dictionary.Dictionary.load_from_text("./Topic_Modeling_Bigram/Input_Data/dictionary.dict")
    u_mass = models.CoherenceModel(corpus=corpus1, model=model, dictionary=dictionary1, coherence='u_mass',
                                   processes=-1).get_coherence()
    return u_mass


def optimize_number_of_topics():
    step = 1
    min = 1
    max = 300
    model_coherences = []
    model_distances = []
    for i in range(min, max + 1, step):
        print(i, " von ", max)
        model = models.LdaModel.load("./Topic_Modeling_Unigram/Models/Topic_Model_%i" % i)
        u_mass = abs(coherence_umass(model))
        model_coherences.append(u_mass)
        model_average_distance = calc_average_distance(calc_model_distance(model, model))
        model_distances.append(model_average_distance)
    print(model_coherences)
    print(model_distances)
    model_combined_data = [model_coherences, model_distances]
    excel_out = pandas.DataFrame(data=model_combined_data)

    pandas.DataFrame(excel_out).to_csv("./Topic_Modeling_Unigram/Evaluation/Coherence_vs_Difference_Jaccard_50_avg_300_topics.csv", sep=';', decimal=',')


# def calc_average_distance(mdiff):
#     upper_right_triangle_matrix_size = 0.0
#     diff_sum = 0.0
#     if len(mdiff) == 1:
#         return mdiff[0, 0]
#     for row in range(0, len(mdiff) - 1):
#         for column in range(row + 1, len(mdiff)):
#             diff_sum += mdiff[row, column]
#             upper_right_triangle_matrix_size += 1.0
#     print("size", upper_right_triangle_matrix_size)
#     print("diff_sum", diff_sum)
#     print(diff_sum / upper_right_triangle_matrix_size)
#     return diff_sum/upper_right_triangle_matrix_size


def calc_average_distance(mdiff):
    upper_right_triangle_differences = []
    if len(mdiff) == 1:
        return mdiff[0, 0]
    for row in range(0, len(mdiff) - 1):
        for column in range(row + 1, len(mdiff)):
            upper_right_triangle_differences.append(mdiff[row, column])
    print(np.average(upper_right_triangle_differences))
    return np.average(upper_right_triangle_differences)


# def calc_median_distance(mdiff):
#     upper_right_triangle_matrix_size = 0.0
#     upper_right_triangle_differences = []
#     if len(mdiff) == 1:
#         return mdiff[0, 0]
#     for row in range(0, len(mdiff) - 1):
#         for column in range(row + 1, len(mdiff)):
#             element = Decimal(mdiff[row, column])
#             print(element)
#             upper_right_triangle_differences.append(element)
#             upper_right_triangle_matrix_size += 1.0
#     if upper_right_triangle_differences.__len__() > 0:
#         print(upper_right_triangle_matrix_size)
#         print(np.median(upper_right_triangle_differences))
#         return np.median(upper_right_triangle_differences)


def median(lst):
    sortedLst = sorted(lst)
    lstLen = len(lst)
    index = (lstLen - 1) // 2

    if (lstLen % 2):
        return sortedLst[index]
    else:
        return (sortedLst[index] + sortedLst[index + 1])/2.0


def calc_median_distance(mdiff):
    upper_right_triangle_differences = []
    if len(mdiff) == 1:
        return mdiff[0, 0]
    for row in range(0, len(mdiff) - 1):
        for column in range(row + 1, len(mdiff)):
            upper_right_triangle_differences.append(mdiff[row, column])
    if upper_right_triangle_differences.__len__() > 0:
        #print(mdiff)
        print(median(upper_right_triangle_differences))
        return median(upper_right_triangle_differences)


def calc_model_distance(model_1, model_2):
    mdiff, annotation = model_1.diff(model_2, distance='jaccard', num_words=50)
    print(mdiff)
    return mdiff


def plot_model(mdiff):
    plot_difference_matplotlib(mdiff, title="Topic difference (one model)[Jaccard distance]")


def plot_difference_plotly(mdiff, title="", annotation=None):
    """Plot the difference between models.

    Uses plotly as the backend."""
    import plotly.graph_objs as go
    import plotly.offline as plot

    annotation_html = None
    if annotation is not None:
        annotation_html = [
            [
                "+++ {}<br>--- {}".format(", ".join(int_tokens), ", ".join(diff_tokens))
                for (int_tokens, diff_tokens) in row
            ]
            for row in annotation
        ]

    data = go.Heatmap(z=mdiff, colorscale='RdBu', text=annotation_html)
    layout = go.Layout(width=950, height=950, title=title, xaxis=dict(title="topic"), yaxis=dict(title="topic"))
    plot.iplot(dict(data=[data], layout=layout))


def plot_difference_matplotlib(mdiff, title="", annotation=None):
    """Helper function to plot difference between models.

    Uses matplotlib as the backend."""

    font = {'family': 'sans',
            'weight': 'normal',
            'size': 22}

    plt.rc('font', **font)

    plt.figure(figsize=(18, 14))


    #plt.subplot(1, 2, 2)
    plt.imshow(mdiff, cmap='Greens', origin='lower')
    plt.colorbar()
    plt.clim(0.7, 1)
    plt.savefig("unigram_jaccard_50_green_07")
    plt.show()



try:
    get_ipython()
    import plotly.offline as py
except Exception:
    #
    # Fall back to matplotlib if we're not in a notebook, or if plotly is
    # unavailable for whatever reason.
    #
    plot_difference = plot_difference_matplotlib
else:
    py.init_notebook_mode()
    plot_difference = plot_difference_plotly




def topic_distance():
    """Topicunähnlichkeit/Topicdistanz"""
    import pprint
    words = 1000
    distance = 'hellinger'
    mdiff, annotation = model1.diff(model1, distance=distance, num_words=words)
    anzahl_terme = 5
    topics1 = [[word for rank, (word, prob) in enumerate(words)]
               for topic_id, words in
               model1.show_topics(formatted=False, num_words=anzahl_terme, num_topics=model1.num_topics)]
    topics1 = [['|'.join(str(e) for e in list1)] for list1 in topics1]
    data = pandas.DataFrame(data=mdiff)
    pandas.DataFrame(data=data).to_csv('./Topic_Modeling/Results/Topic_Distance_%s_%i.csv' % (distance, words),
                                       header=topics1, sep=";", decimal=',')
    print("\n\nTopic distance matrix with distance measure: %s processed!\n\n" % distance)
