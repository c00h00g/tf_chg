# -*- coding:utf-8 -*-

import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

all_embeddings = []
with open('emb_res') as f:
    for line in f.readlines():
        line = line.rstrip()
        line_split = line.split(' ')
        line_split = [ float(x) for x in line_split ]
        all_embeddings.append(line_split)

all_embeddings = np.array(all_embeddings)

all_words = []
id2word = dict()
with open('id2term') as f:
    for line in f.readlines():
        line = line.rstrip()
        line_split = line.split('\t')
        id = int(line_split[0])
        term = line_split[1]
        id2word[id] = term
        all_words.append(term)


for idx, one_emb in enumerate(all_embeddings):
    one_emb = [one_emb]
    d = cosine_similarity(one_emb, all_embeddings)[0]
    d = zip(all_words, d)
    d = sorted(d, key=lambda x:x[1], reverse=True)

    print('term is : ', id2word[idx])
    for w in d[:10]:
        if len(w[0]) < 2:
            continue
        print(w)
