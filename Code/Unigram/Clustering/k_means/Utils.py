from gensim import models, utils, corpora
import glob
import pandas as pd
from collections import deque

df = pd.read_csv('topics_83.csv')


print(df)

topics_list = df.to_numpy().tolist()
print(topics_list)

for topics in topics_list:
    del topics[0]

print(topics_list)
