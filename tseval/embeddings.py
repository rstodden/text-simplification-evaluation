# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

import numpy as np
import torch
from pathlib import Path

from tseval.utils.paths import FASTTEXT_EMBEDDINGS_PATH
from tseval.utils.prepare import prepare_resource
#from align_sentences import LANG


def load_fasttext_embeddings(vocab_size=None, lang="en"):
    FASTTEXT_EMBEDDINGS_PATH_LANG = Path(str(FASTTEXT_EMBEDDINGS_PATH)+"/cc." + lang + ".300.vec")
    # print(FASTTEXT_EMBEDDINGS_PATH_LANG)
    # os.path.join(VARIOUS_DIR, FASTTEXT_EMBEDDINGS_PATH, "cc." + lang + ".300.vec")
    if not os.path.exists(FASTTEXT_EMBEDDINGS_PATH_LANG):
        prepare_resource('fasttext_embeddings', lang, FASTTEXT_EMBEDDINGS_PATH)
    with open(FASTTEXT_EMBEDDINGS_PATH_LANG, 'r') as f:
        # First line contains number and size of vectors
        total_embeddings, embedding_dim = [int(val) for val in f.readline().split()]
        if vocab_size is None:
            vocab_size = total_embeddings
        word_embeddings = torch.zeros(vocab_size, embedding_dim)
        # TODO: Is having a vector of zeros the best embedding for unknown words?
        embedded_words = ['<unk>']
        for i, line in enumerate(f):
            i = i + 1  # Shift i to take unk into account
            if i >= vocab_size:
                break
            word, *embedding = line.strip(' \n').split(' ')
            embedded_words.append(word)
            word_embeddings[i, :] = torch.FloatTensor(np.array(embedding, dtype=float))
    # For fast embedding retrieval
    word2index = {word: i for i, word in enumerate(embedded_words)}
    return word_embeddings, word2index


def to_embeddings(sentence, lang):
    if 'EMBEDDINGS' not in globals():
        global EMBEDDINGS, WORD2INDEX
        print('Loading '+lang+' FastText embeddings...' )
        EMBEDDINGS, WORD2INDEX = load_fasttext_embeddings(vocab_size=100000, lang=lang)
        print('Done.')
    indexes = [WORD2INDEX.get(word.text, WORD2INDEX['<unk>']) for word in sentence]
    return EMBEDDINGS[indexes]
