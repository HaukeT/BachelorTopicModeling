import nltk
from gensim import models, utils, corpora
import glob
import pandas as pd
from collections import deque
import numpy as np


from csv import reader

d = []
with open('topics_83.csv') as f:
    csv_reader = reader(f, delimiter=';')
    for i, line in enumerate(csv_reader, start=1):
        d.append(list(map(str, line)))

print(d)




