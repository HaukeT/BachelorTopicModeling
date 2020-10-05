import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('TDM-GM-3-2-5-korrigiert.csv', sep=';')
df = df.fillna(0.0)
columns = list(df)

list_of_list = df.to_numpy().tolist()
bigrams = []
for part in list_of_list:
    bigrams.append(part[0])
df = df.drop(['Unnamed: 0'], axis=1)
list_of_list_numbers_only = df.to_numpy().tolist()

document_text = []
list_of_list_ints_only = []
for i in list_of_list_numbers_only:
    list_of_list_ints_only.append([int(j) for j in i])


list_of_list_ints_only_inverted = np.transpose(list_of_list_ints_only)


# for bigram in bigrams:
#     for numbers in list_of_list_ints_only_inverted:
#         document_text = []
#         for number in numbers:
#             document_text.append(bigram * number)
#     bigram_text.append(document_text)

for numbers in list_of_list_ints_only_inverted:
    document = []
    for counter, number in enumerate(numbers):
        if number != 0:
            document.extend([bigrams[counter]] * number)
    document_shallow_copy = list(document)
    document_text = document_text + [document_shallow_copy]
    document.clear()

# print(len(bigram_text[0]))
with open('bigram_2d_matrix', 'wb') as fp:
    pickle.dump(document_text, fp)
