from gensim import corpora, models, matutils
from sklearn.cluster import KMeans
import pickle
import numpy as np
from sklearn import manifold
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer, InterclusterDistance

lda_model = models.LdaModel.load("./model/Topic_Model_83")
corpus = corpora.MmCorpus("./data/corpus.mm")
dictionary = corpora.dictionary.Dictionary.load_from_text("./data/dictionary.dict")
with open('outfile', 'rb') as fp:
    texts = pickle.load(fp)


tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

print("TFIDF:")
corpus_tfidf = matutils.corpus2csc(corpus_tfidf).transpose()
print(corpus_tfidf)
print("__________________________________________")

kmeans = KMeans(n_clusters=10, init ='k-means++', max_iter=300, n_init=10,random_state=0 )
print(corpus_tfidf)
print(kmeans.fit(corpus_tfidf))

#Elbow Method

#model = KMeans()
#visualizer = KElbowVisualizer(model, k=(4,20))

#visualizer.fit(corpus_tfidf)        # Fit the data to the visualizer
#visualizer.show()        # Finalize and render the figure


# Instantiate the clustering model and visualizer
#model = KMeans(n_clusters=10)
#visualizer = InterclusterDistance(model)

#visualizer.fit(corpus_tfidf)        # Fit the data to the visualizer
#visualizer.show()        # Finalize and render the figure


#ClusterPlot

# y_kmeans = kmeans.fit_predict(corpus_tfidf)
#
# X = corpus_tfidf
#
# plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], s=100, c='red', label ='Cluster 1')
# plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], s=100, c='blue', label ='Cluster 2')
# plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], s=100, c='green', label ='Cluster 3')
# plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1], s=100, c='cyan', label ='Cluster 4')
# plt.scatter(X[y_kmeans==4, 0], X[y_kmeans==4, 1], s=100, c='#9C33FF', label ='Cluster 5')
# plt.scatter(X[y_kmeans==5, 0], X[y_kmeans==5, 1], s=100, c='#FF5733', label ='Cluster 6')
# plt.scatter(X[y_kmeans==6, 0], X[y_kmeans==6, 1], s=100, c='#33FFC4', label ='Cluster 7')
# plt.scatter(X[y_kmeans==7, 0], X[y_kmeans==7, 1], s=100, c='#FF33AC', label ='Cluster 8')
# plt.scatter(X[y_kmeans==8, 0], X[y_kmeans==8, 1], s=100, c='#CEFF33', label ='Cluster 9')
# plt.scatter(X[y_kmeans==9, 0], X[y_kmeans==9, 1], s=100, c='#7D3F3F', label ='Cluster 10')

# def plot_embedding(X, title=None):
#     x_min, x_max = np.min(X, 0), np.max(X, 0)
#     X = (X - x_min) / (x_max - x_min)
#
#     plt.figure()
#     ax = plt.subplot(111)
#     for i in range(X.shape[0]):
#         plt.text(X[i, 0], X[i, 1], str(y[i]),
#                  color=plt.cm.Set1(y[i] / 10.),
#                  fontdict={'weight': 'bold', 'size': 9})
#
#     plt.xticks([]), plt.yticks([])
#     if title is not None:
#         plt.title(title)
#
# print("Computing t-SNE embedding")
# tsne = manifold.TSNE(n_components=10, init='pca', random_state=0)
# X_tsne = tsne.fit_transform(corpus_tfidf)
#
# plot_embedding(X_tsne, "t-SNE")

# y_km = kmeans.fit_predict(corpus_tfidf)
#
# # plot the 3 clusters
# plt.scatter(
#     corpus_tfidf[y_km == 0, 0], corpus_tfidf[y_km == 0, 1],
#     s=50, c='lightgreen',
#     marker='s', edgecolor='black',
#     label='cluster 1'
# )
#
# plt.scatter(
#     corpus_tfidf[y_km == 1, 0], corpus_tfidf[y_km == 1, 1],
#     s=50, c='orange',
#     marker='o', edgecolor='black',
#     label='cluster 2'
# )
#
# plt.scatter(
#     corpus_tfidf[y_km == 2, 0], corpus_tfidf[y_km == 2, 1],
#     s=50, c='lightblue',
#     marker='v', edgecolor='black',
#     label='cluster 3'
# )
#
# plt.scatter(
#     corpus_tfidf[y_km == 3, 0], corpus_tfidf[y_km == 3, 1],
#     s=50, c='red',
#     marker='v', edgecolor='black',
#     label='cluster 4'
# )
#
# plt.scatter(
#     corpus_tfidf[y_km == 4, 0], corpus_tfidf[y_km == 4, 1],
#     s=50, c='pink',
#     marker='v', edgecolor='black',
#     label='cluster 5'
# )
#
# plt.scatter(
#     corpus_tfidf[y_km == 5, 0], corpus_tfidf[y_km == 5, 1],
#     s=50, c='brown',
#     marker='v', edgecolor='black',
#     label='cluster 6'
# )
#
# plt.scatter(
#     corpus_tfidf[y_km == 6, 0], corpus_tfidf[y_km == 6, 1],
#     s=50, c='yellow',
#     marker='v', edgecolor='black',
#     label='cluster 7'
# )
#
# plt.scatter(
#     corpus_tfidf[y_km == 7, 0], corpus_tfidf[y_km == 7, 1],
#     s=50, c='darkblue',
#     marker='v', edgecolor='black',
#     label='cluster 8'
# )
#
# plt.scatter(
#     corpus_tfidf[y_km == 8, 0], corpus_tfidf[y_km == 8, 1],
#     s=50, c='darkgreen',
#     marker='v', edgecolor='black',
#     label='cluster 9'
# )
#
# plt.scatter(
#     corpus_tfidf[y_km == 9, 0], corpus_tfidf[y_km == 9, 1],
#     s=50, c='violet',
#     marker='v', edgecolor='black',
#     label='cluster 10'
# )
# # plot the centroids
# plt.scatter(
#     kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
#     s=250, marker='*',
#     c='red', edgecolor='black',
#     label='centroids'
# )
# plt.legend(scatterpoints=1)
# plt.grid()
# plt.show()
