# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from functools import lru_cache
import re
from string import punctuation
import pyphen
import spacy
#from tseval.utils.paths import LANG
# from align_sentences import LANG

@lru_cache(maxsize=1)
def get_stopwords():
    # TODO: #language_specific
    # Inline lazy import because importing nltk is slow
    import nltk
    try:
        return set(nltk.corpus.stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        return set(nltk.corpus.stopwords.words('english'))


def to_words(sentence, lang):
    if len(list(sentence.sents)) > 1:
        return [token.text for sent in sentence.sents for token in sent]
    return [token.text for token in sentence]


def to_lemma(sentence, lang):
    if len(list(sentence.sents)) > 1:
        return [token.lemma_ for sent in sentence.sents for token in sent]
    return [token.lemma_ for token in sentence]

@lru_cache(maxsize=1000)
def is_punctuation(word):
    return word in punctuation


@lru_cache(maxsize=100)
def remove_punctuation_tokens(text, lang):
    return [token for token in text if not token.is_punct]


def remove_stopwords(text, lang):
    return [token for token in text if not token.is_stop]


def count_words(sentence, lang):
    return len(sentence)  # /count_sentences(sentence)


def count_sentences(text, lang):
    return len(list(text.sents))


def count_syllables_in_sentence(sentence, lang):
    dictionary = pyphen.Pyphen(lang=lang)
    return sum([len(dictionary.inserted(token.text).split("-")) for token in sentence])


def to_lower(sentence, lang):
    if len(list(sentence.sents)) > 1:
        return [token.text.lower() for sent in sentence.sents for token in sent]
    return [token.text.lower() for token in sentence]