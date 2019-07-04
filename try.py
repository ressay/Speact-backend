import json

import gensim
import nltk

nltk.data.path.append('/home/ressay/workspace/PFE_M2/app-backend/resources/')
tagger = nltk.tag.PerceptronTagger()
print(tagger.tag(['hello','world']))

w2v_model = gensim.models.KeyedVectors.load_word2vec_format('models/GoogleNews-vectors-negative300.bin.gz', binary=True)
print(w2v_model['hello'])