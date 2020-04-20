# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from collections import Counter
import itertools
from functools import lru_cache
import os
import spacy, spacy_udpipe
import stanza
from spacy_stanza import StanzaLanguage


import Levenshtein
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import skew
from scipy.special import boxcox1p
from sklearn.base import TransformerMixin
import torch
import torch.nn.functional as F

from tseval.embeddings import to_embeddings
from tseval.evaluation.readability import sentence_fre, sentence_fkgl
# from tseval.evaluation.quest import get_quest_vectorizers
from tseval.utils.paths import VARIOUS_DIR, FASTTEXT_EMBEDDINGS_PATH
from tseval.text import (count_words, count_sentences, count_syllables_in_sentence, remove_stopwords,
                         remove_punctuation_tokens, to_words, to_lemma, to_lower)
from tseval.utils.helpers import yield_lines


@lru_cache(maxsize=1)
def get_word2concreteness():
    concrete_words_path = os.path.join(VARIOUS_DIR, 'concrete_words.tsv')
    df = pd.read_csv(concrete_words_path, sep='\t')
    df = df[df['Bigram'] == 0]  # Remove bigrams
    return {row['Word']: row['Conc.M'] for _, row in df.iterrows()}


@lru_cache(maxsize=1)
def get_word2frequency(lang):
    frequency_table_path = os.path.join(VARIOUS_DIR, FASTTEXT_EMBEDDINGS_PATH, "cc."+lang+".300.vec")
    word2frequency = {}
    for line in yield_lines(frequency_table_path):
        word, frequency = line.split('\t')
        word2frequency[word] = int(frequency)
    return word2frequency


@lru_cache(maxsize=1)
def get_word2rank(lang, vocab_size=50000):
    frequency_table_path = os.path.join(VARIOUS_DIR, FASTTEXT_EMBEDDINGS_PATH, "cc."+lang+".300.vec")
    word2rank = {}
    for rank, line in enumerate(yield_lines(frequency_table_path)):
        if (rank+1) > vocab_size:
            break
        word = line.lstrip().split(' ')[0]
        word2rank[word] = rank
    return word2rank


# def get_concreteness(word):
#     # TODO: Default value is arbitrary
#     return get_word2concreteness().get(word, 5)
#
#
# def get_frequency(word):
#     return get_word2frequency().get(word, None)
#
#
# def get_negative_frequency(word):
#     return -get_frequency(word)


def get_rank(word, lang):
    """if no rank is found in dictionary, the last rank is used"""
    return get_word2rank(lang).get(word, len(get_word2rank(lang)))


# def get_negative_log_frequency(word):
#     return -np.log(1 + get_frequency(word))


def get_log_rank(word, lang):
    return np.log(1 + get_rank(word, lang))


# Single sentence feature extractors with signature method(sentence) -> float
#def get_concreteness_scores(sentence):
#    return np.log(1 + np.array([get_concreteness(word) for word in sentence]))


def get_frequency_table_ranks(sentence, lang):
    return np.log(1 + np.array([get_rank(word.text, lang) for word in sentence]))


def get_wordrank_score(sentence, lang):
    # Computed as the third quartile of log ranks
    words = remove_stopwords(remove_punctuation_tokens(sentence, lang=lang), lang)
    words = [word.text for word in words if word.text in get_word2rank(lang)]
    if len(words) == 0:
        return np.log(1 + len(get_word2rank(lang)))  # TODO: This is completely arbitrary
    return np.quantile([get_log_rank(word, lang) for word in words], 0.75)


def count_characters(sentence, lang):
    return len(sentence.text)


def safe_division(a, b):
    if b == 0:
        return b
    return a / b


def count_words_per_sentence(sentence, lang):
    return safe_division(count_words(sentence, lang=lang), count_sentences(sentence, lang=lang))


def count_characters_per_sentence(sentence, lang):
    return safe_division(count_characters(sentence, lang=lang), count_sentences(sentence, lang=lang))


def count_syllables_per_sentence(sentence, lang):
    return safe_division(count_syllables_in_sentence(sentence, lang=lang), count_sentences(sentence, lang=lang))


def count_characters_per_word(sentence, lang):
    return safe_division(count_characters(sentence, lang=lang), count_words(sentence, lang=lang))


def count_syllables_per_word(sentence, lang):
    return safe_division(count_syllables_in_sentence(sentence, lang=lang), count_words(sentence, lang=lang))


def max_pos_in_freq_table(sentence, lang):
    #to_embeddings("", lang)
    return max(get_frequency_table_ranks(sentence, lang=lang))


def average_pos_in_freq_table(sentence, lang):
    return np.mean(get_frequency_table_ranks(sentence, lang=lang))


#def min_concreteness(sentence):
#    return min(get_concreteness_scores(sentence))


#def average_concreteness(sentence):
#    return np.mean(get_concreteness_scores(sentence))


# OPTIMIZE: Optimize feature extractors? A lot of computation is duplicated (e.g. to_words)
def get_sentence_feature_extractors():
    return [
        count_words,
        count_characters,
        count_sentences,
        count_syllables_in_sentence,
        count_words_per_sentence,
        count_characters_per_sentence,
        count_syllables_per_sentence,
        count_characters_per_word,
        count_syllables_per_word,
        max_pos_in_freq_table,
        average_pos_in_freq_table,
        #min_concreteness,
        #average_concreteness,
        sentence_fre,
        sentence_fkgl,
        # average_sentence_lm_prob,
        # min_sentence_lm_prob,
        get_lexical_complexity_score
    ]


# Sentence pair feature extractors with signature method(complex_sentence, simple_sentence) -> float
def count_sentence_splits(complex_sentence, simple_sentence, lang):
    return safe_division(count_sentences(simple_sentence, lang=lang), count_sentences(complex_sentence, lang=lang))


def get_compression_ratio(complex_sentence, simple_sentence, lang):
    return safe_division(count_characters(simple_sentence, lang), count_characters(complex_sentence, lang))


def word_intersection(complex_sentence, simple_sentence, lang):
    complex_words = [token.lemma_ for token in complex_sentence]
    simple_words = [token.lemma_ for token in simple_sentence]
    nb_common_words = len(set(complex_words).intersection(set(simple_words)))
    nb_max_words = max(len(set(complex_words)), len(set(simple_words)))
    return safe_division(nb_common_words, nb_max_words)


@lru_cache(maxsize=10000)
def average_dot(complex_sentence, simple_sentence, lang):
    complex_embeddings = to_embeddings(complex_sentence, lang)
    simple_embeddings = to_embeddings(simple_sentence, lang)
    return float(torch.dot(complex_embeddings.mean(dim=0), simple_embeddings.mean(dim=0)))


@lru_cache(maxsize=10000)
def average_cosine(complex_sentence, simple_sentence, lang):
    complex_embeddings = to_embeddings(complex_sentence, lang)
    simple_embeddings = to_embeddings(simple_sentence, lang)
    return float(F.cosine_similarity(complex_embeddings.mean(dim=0),
                                     simple_embeddings.mean(dim=0),
                                     dim=0))


@lru_cache(maxsize=10000)
def hungarian_dot(complex_sentence, simple_sentence, lang):
    complex_embeddings = to_embeddings(complex_sentence, lang)
    simple_embeddings = to_embeddings(simple_sentence, lang)
    similarity_matrix = torch.mm(complex_embeddings, simple_embeddings.t())
    row_indexes, col_indexes = linear_sum_assignment(-similarity_matrix)
    # TODO: Penalize less deletion of unimportant words
    return float(similarity_matrix[row_indexes, col_indexes].sum() / max(len(complex_sentence), len(simple_sentence)))


@lru_cache(maxsize=10000)
def hungarian_cosine(complex_sentence, simple_sentence, lang):
    complex_embeddings = to_embeddings(complex_sentence, lang)
    simple_embeddings = to_embeddings(simple_sentence, lang)
    similarity_matrix = torch.zeros(len(complex_embeddings), len(simple_embeddings))
    for (i, complex_embedding), (j, simple_embedding) in itertools.product(enumerate(complex_embeddings),
                                                                           enumerate(simple_embeddings)):
        similarity_matrix[i, j] = F.cosine_similarity(complex_embedding, simple_embedding, dim=0)
    row_indexes, col_indexes = linear_sum_assignment(-similarity_matrix)
    # TODO: Penalize less deletion of unimportant words
    return float(similarity_matrix[row_indexes, col_indexes].sum() / max(len(complex_sentence), len(simple_sentence)))


def characters_per_sentence_difference(complex_sentence, simple_sentence, lang):
    return count_characters(complex_sentence, lang) - count_characters(simple_sentence, lang)


def is_exact_match(complex_sentence, simple_sentence, lang):
    return int(complex_sentence == simple_sentence)


def get_levenshtein_similarity(complex_sentence, simple_sentence, lang):
    return Levenshtein.ratio(complex_sentence.text, simple_sentence.text)


def get_levenshtein_distance(complex_sentence, simple_sentence, lang):
    return 1 - get_levenshtein_similarity(complex_sentence, simple_sentence, lang)


def flatten_counter(counter):
    return [k for key, count in counter.items() for k in [key] * count]

def num_changes(counter):
    return sum(counter.values())

def get_added_words(c, s, lang):
    """new words which are in the simplified but not in the complex sentence, disregarding of word form"""
    return flatten_counter(Counter(to_lemma(s, lang)) - Counter(to_lemma(c, lang)))


def get_rewritten_words(c, s, lang):
    """rewritten word, e.g., lowercased or inflected differently. No exact match of word form anymore but another representation of the lemma."""
    if 'SPACY_MODEL' not in globals():
        print("load spacy model")
        global SPACY_MODEL
        """if lang == "en":
            SPACY_MODEL = spacy.load("en_core_web_sm")
        elif lang == "cs":
            SPACY_MODEL = spacy_udpipe.load(lang)
        else:
            SPACY_MODEL = spacy.load(lang + "_core_news_sm")"""
        snlp = stanza.Pipeline(lang=lang)
        SPACY_MODEL = StanzaLanguage(snlp)
    added_words = get_added_words(c, s, lang)
    rewritten_or_added = [SPACY_MODEL(token)[0].lemma_ for token in flatten_counter(Counter(to_words(s, lang)) - Counter(to_words(c, lang)))]
    return list(set(rewritten_or_added).difference(set(added_words)))


def get_deleted_words(c, s, lang):
    """the token does not occur in an inflected or capitalized oder lowercased version in the simplified sentence"""
    return flatten_counter(Counter(to_lemma(c, lang)) - Counter(to_lemma(s, lang)))


def get_kept_words(c, s, lang):
    """words which occur in simplified and complex sentence but inflected, lowercased or capitalized"""
    return flatten_counter(Counter(to_lemma(c, lang)) & Counter(to_lemma(s, lang)))


def get_unchanged_words(c, s, lang):
    """word which occur exactly the same in simplified and complex sentence."""
    return flatten_counter(Counter(to_words(c, lang)) & Counter(to_words(s, lang)))



def get_lcs(seq1, seq2):
    '''Returns the longest common subsequence using memoization (only in local scope)'''
    @lru_cache(maxsize=None)
    def recursive_lcs(seq1, seq2):
        if len(seq1) == 0 or len(seq2) == 0:
            return []
        if seq1[-1] == seq2[-1]:
            return recursive_lcs(seq1[:-1], seq2[:-1]) + [seq1[-1]]
        else:
            return max(recursive_lcs(seq1[:-1], seq2), recursive_lcs(seq1, seq2[:-1]), key=lambda seq: len(seq))

    try:
        return recursive_lcs(tuple(seq1), tuple(seq2))
    except RecursionError as e:
        print(e)
        # TODO: Handle this case
        return []


def get_reordered_words(c, s, lang):
    # A reordered word is a word that is contained in the source and simplification
    # but not in the longuest common subsequence
    c = c.lower()
    s = s.lower()
    lcs = get_lcs(to_words(c, lang), to_words(s, lang))
    return flatten_counter(Counter(get_kept_words(c, s, lang)) - Counter(lcs))


def get_n_added_words(c, s, lang):
    return num_changes(Counter(to_lemma(s, lang)) - Counter(to_lemma(c, lang)))
    #return len(get_added_words(c, s, lang))


def get_n_deleted_words(c, s, lang):
    return num_changes(Counter(to_lemma(c, lang)) - Counter(to_lemma(s, lang)))
    # return len(get_deleted_words(c, s, lang))


def get_n_kept_words(c, s, lang):
    return num_changes(Counter(to_lemma(c, lang)) & Counter(to_lemma(s, lang)))
    #     # return len(get_kept_words(c, s, lang))


def get_n_reordered_words(c, s, lang):
    # todo
    # A reordered word is a word that is contained in the source and simplification
    # but not in the longuest common subsequence

    lcs = get_lcs(to_lower(c, lang), to_lower(s, lang))
    return flatten_counter(Counter(get_kept_words(c, s, lang)) - Counter(lcs))
    # return len(get_reordered_words(c, s, lang))


def get_n_rewritten_words(c, s, lang):
    """rewritten word, e.g., lowercased or inflected differently. No exact match of word form anymore but another representation of the lemma."""
    if 'SPACY_MODEL' not in globals():
        print("load spacy model")
        global SPACY_MODEL
        """if lang == "en":
            SPACY_MODEL = spacy.load("en_core_web_sm")
        elif lang == "cs":
            SPACY_MODEL = spacy_udpipe.load(lang)
        else:
            SPACY_MODEL = spacy.load(lang + "_core_news_sm")"""
        snlp = stanza.Pipeline(lang=lang)
        SPACY_MODEL = StanzaLanguage(snlp)
    added_words = get_added_words(c, s, lang)
    rewritten_or_added = [SPACY_MODEL(token)[0].lemma_ for token in
                          (Counter(to_words(s, lang)) - Counter(to_words(c, lang))).keys()]
    rewritten_or_added_words = set(rewritten_or_added).difference(set(added_words))
    return sum([count for word, count in Counter(s).items() if word.lemma_ in rewritten_or_added_words])  #rewritten_or_added])
    # return list(set(rewritten_or_added).difference(set(added_words)))
    # return len(get_rewritten_words(c, s, lang))


def get_n_unchanged_words(c, s, lang):
    return num_changes(Counter(to_words(c, lang)) & Counter(to_words(s, lang)))
    # return len(get_unchanged_words(c, s, lang))


def get_added_words_proportion(c, s, lang):
    # Relative to simple sentence
    return safe_division(get_n_added_words(c, s, lang), count_words(s, lang))


def get_deleted_words_proportion(c, s, lang):
    # Relative to complex sentence
    return safe_division(get_n_deleted_words(c, s, lang), count_words(c, lang))


def get_rewritten_words_proportion(c, s, lang):
    # Relative to simple sentence
    return safe_division(get_n_rewritten_words(c, s, lang), count_words(s, lang))


def get_unchanged_words_proportion(c, s, lang):
    # Relative to simple sentence
    return safe_division(get_n_unchanged_words(c, s, lang), count_words(s, lang))


def get_kept_words_proportion(c, s, lang):
    # Relative to simple sentence
    return safe_division(get_n_kept_words(c, s, lang), count_words(s, lang))


def get_sum_of_simple_proportion(c, s, lang):
    return get_added_words_proportion(c, s, lang) + get_kept_words_proportion(c, s, lang)


def get_reordered_words_proportion(c, s, lang):
    # @todo
    # Relative to complex sentence
    return safe_division(get_n_deleted_words(c, s, lang), count_words(s, lang))


def only_deleted_words(c, s, lang):
    # Only counting deleted words does not work because sometimes there is reordering
    return not is_exact_match(c, s, lang) and get_lcs(to_words(c, lang), to_words(s, lang)) == to_words(s, lang)


@lru_cache(maxsize=1)
def get_nlgeval():
    try:
        from nlgeval import NLGEval
    except ModuleNotFoundError:
        print('nlg-eval module not installed. Please install with ',
              'pip install nlg-eval@git+https://github.com/Maluuba/nlg-eval.git')
    print('Loading NLGEval models...')
    return NLGEval(no_skipthoughts=True, no_glove=True)


# Making one call to nlgeval returns all metrics, we therefore cache the results in order to limit the number of calls
@lru_cache(maxsize=10000)
def get_all_nlgeval_metrics(complex_sentence, simple_sentence, lang):
    return get_nlgeval().compute_individual_metrics([complex_sentence.text], simple_sentence.text)


def get_nlgeval_methods():
    """Returns all scoring methods from nlgeval package.

    Signature: method(complex_sentence, simple_setence)
    """
    def get_scoring_method(metric_name):
        """Necessary to wrap the scoring_method() in get_scoring_method(), in order to set the external variable to
        its current value."""
        def scoring_method(complex_sentence, simple_sentence, lang):
            return get_all_nlgeval_metrics(complex_sentence, simple_sentence, lang)[metric_name]
        return scoring_method

    nlgeval_metrics = [
        # Fast metrics
        'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr',
        # Slow metrics
        # 'SkipThoughtCS', 'EmbeddingAverageCosineSimilairty', 'VectorExtremaCosineSimilarity', 'GreedyMatchingScore',
    ]
    methods = []
    for metric_name in nlgeval_metrics:
        scoring_method = get_scoring_method(metric_name)
        scoring_method.__name__ = f'nlgeval_{metric_name}'
        methods.append(scoring_method)
    return methods


def get_nltk_bleu_methods():
    """Returns bleu methods with different smoothings from NLTK.
Signature: scoring_method(complex_sentence, simple_setence)
    """
    # Inline lazy import because importing nltk is slow
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    def get_scoring_method(smoothing_function):
        """Necessary to wrap the scoring_method() in get_scoring_method(), in order to set the external variable to
        its current value."""
        def scoring_method(complex_sentence, simple_sentence, lang):
            try:
                return sentence_bleu([[token.text for token in complex_sentence]], [token.text for token in simple_sentence],
                                     smoothing_function=smoothing_function)
            except (ZeroDivisionError, AssertionError) as e:
                return 0
        return scoring_method

    methods = []
    for i in range(8):
        smoothing_function = getattr(SmoothingFunction(), f'method{i}')
        scoring_method = get_scoring_method(smoothing_function)
        scoring_method.__name__ = f'nltkBLEU_method{i}'
        methods.append(scoring_method)
    return methods


def get_sentence_pair_feature_extractors():
    return [
        word_intersection,
        get_compression_ratio,
        get_levenshtein_distance,
        get_levenshtein_similarity,
        characters_per_sentence_difference,
        average_dot,
        average_cosine,
        hungarian_dot,
        hungarian_cosine,
    ]  + get_nltk_bleu_methods()  + get_simplification_transformations() + get_nlgeval_methods()
    # + get_quest_vectorizers()  # + get_terp_vectorizers()


def get_simplification_transformations():
    return [
        get_added_words_proportion,
        get_deleted_words_proportion,
        get_rewritten_words_proportion,
        get_kept_words_proportion,
        get_unchanged_words_proportion,
    ]


def get_lexical_complexity_score(input_sentence, lang):
    return get_wordrank_score(input_sentence, lang)

# Various
def wrap_single_sentence_vectorizer(vectorizer):
    '''Transform a single sentence vectorizer to a sentence pair vectorizer

    Change the signature of the input vectorizer
    Initial signature: method(simple_sentence)
    New signature: method(complex_sentence, simple_sentence)
    '''
    def wrapped(complex_sentence, simple_sentence):
        return vectorizer(simple_sentence)

    wrapped.__name__ = vectorizer.__name__
    return wrapped


def reverse_vectorizer(vectorizer):
    '''Reverse the arguments of a vectorizer'''
    def reversed_vectorizer(complex_sentence, simple_sentence):
        return vectorizer(simple_sentence, complex_sentence)

    reversed_vectorizer.__name__ = vectorizer.__name__ + '_reversed'
    return reversed_vectorizer


def get_all_vectorizers(reversed=False):
    vectorizers = [wrap_single_sentence_vectorizer(vectorizer)
                   for vectorizer in get_sentence_feature_extractors()] + get_sentence_pair_feature_extractors()
    if reversed:
        vectorizers += [reverse_vectorizer(vectorizer) for vectorizer in vectorizers]
    return vectorizers


def concatenate_corpus_vectorizers(vectorizers):
    '''Given a list of corpus vectorizers, create a new single concatenated corpus vectorizer.

    Corpus vectorizer:
    Given a numpy array of shape (n_samples, 2), it will extract features for each sentence pair
    and output a (n_samples, n_features) array.
    '''
    def concatenated(sentence_pairs):
        return np.column_stack([vectorizer(sentence_pairs) for vectorizer in vectorizers])
    return concatenated


class FeatureSkewer(TransformerMixin):
    '''Normalize features that have a skewed distribution'''
    def fit(self, X, y):
        self.skewed_indexes = [i for i in range(X.shape[1]) if skew(X[:, i]) > 0.75]
        return self

    def transform(self, X):
        for i in self.skewed_indexes:
            X[:, i] = boxcox1p(X[:, i], 0)
        return np.nan_to_num(X)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)



########################################
# additional functin compared to tseval of facebookresearch

def get_average(vectorizer, orig_sentences, sys_sentences):
    """ code originally fromo https://github.com/feralvam/easse GNU GENERAL PUBLIC LICENSE"""
    cumsum = 0
    count = 0
    for orig_sentence, sys_sentence in zip(orig_sentences, sys_sentences):
        cumsum += vectorizer(orig_sentence, sys_sentence)
        count += 1
    return cumsum / count


def get_token_with_deprel(sentence, tags: list):
    return [token for token in sentence if token.dep_.lower() in tags]


def get_token_id_with_deprel(sentence, tags: list):
    return [token.i for token in sentence if token.dep_.lower() in tags]


def get_ratio_of_deprel_per_sent(sentence, tags: list):
    """
    Calculates the ratio of tokens per sentence, which are labeled with a dependency label of <tags>
    :param sentence: spacy.tokens.doc.Doc object, can be either a sentence or a multiple sentences.
    :param tags: list containing universal dependency relation labels, see here for a list https://spacy.io/api/annotation#dependency-parsing-universal
    :return: average of tokens with the specified dependency relation per sentence
    """
    # count all tokens with deprel overall sentences and divide by length of all sentences
    sent_len = 0
    n_tok = 0
    for sent in list(sentence.sents):
        sent_len += len(sent)
        n_tok += len(get_token_with_deprel(sent, tags))
    return safe_division(n_tok, sent_len)


def get_token_with_part_of_speech_tag(sentence, tags: list):
    return [token for token in sentence if token.pos_.lower() in tags or token.tag_.lower() in tags]


def get_token_id_with_part_of_speech_tag(sentence, tags: list):
    return [token.i for token in sentence if token.pos_.lower() in tags]


def get_ratio_of_part_of_speech_per_sent(sentence, tags: list):
    """
    Calculates the ratio of tokens per sentence, which are labeled with a part of speech tag of <tags>
    :param sentence: spacy.tokens.doc.Doc object, can be either a sentence or a multiple sentences.
    :param tags: list containing universal part of speech tags, see here for a list https://spacy.io/api/annotation#pos-universal
    :return: average of tokens with the specified dependency relation per sentence
    """
    sent_len = 0
    n_tok = 0
    for sent in list(sentence.sents):
        sent_len += len(sent)
        n_tok += len(get_token_with_part_of_speech_tag(sent, tags))
    return safe_division(n_tok, sent_len)


def get_parse_tree_height(sentence, lang):
    """
    :param sentence: spacy.tokens.doc.Doc object, can be either a sentence or a multiple sentences.
    :return: dependency parse tree height per sentence
    """
    return np.mean([tree_height(sent.root) for sent in list(sentence.sents)])


def tree_height(root):
    """
    Code originally from https://gist.github.com/drussellmrichie/47deb429350e2e99ffb3272ab6ab216a#file-average_parse_tree_height-py
    Find the maximum depth (height) of the dependency parse of a spacy sentence by starting with its root
    Code adapted from https://stackoverflow.com/questions/35920826/how-to-find-height-for-non-binary-tree
    :param root: spacy.tokens.token.Token
    :return: int, maximum height of sentence's dependency parse tree
    """
    if not list(root.children):
        return 1
    else:
        return 1 + max(tree_height(x) for x in root.children)


def get_ratio_of_function_words(sentence, lang):
    """Following Universal Dependencies, the following dependency relations mark function words:
     aux, cop, mark and case. See: https://universaldependencies.org/v2/function.html [Jan 2020] and for the
     German labels of the Tiger Treebank https://files.ifi.uzh.ch/cl/siclemat/lehre/papers/tiger-annot.pdf"""

    return min(1, get_ratio_of_deprel_per_sent(sentence, get_dependency_label(lang, "functionwords")))


def get_ratio_of_nouns(sentence, lang):
    """
    spacy normally follows the UD tags but for English and German it uses other tags, see here
    https://spacy.io/api/annotation#pos-tagging
    :param sentence: spacy.tokens.doc.Doc object, can be either a sentence or multiple sentences.
    :return: ratio of universal part of speech tag "noun", "propn" (proper noun), "pron" (pronoun) per sentence
    """
    return min(1, get_ratio_of_part_of_speech_per_sent(sentence, get_pos_tags(lang, "noun")))


def get_ratio_of_pronouns(sentence, lang):
    """
    :param sentence: spacy.tokens.doc.Doc object, can be either a sentence or a multiple sentences.
    :return: ratio of universal part of speech tag "pron" (pronouns) per sentence
    """
    return min(1, get_ratio_of_part_of_speech_per_sent(sentence, get_pos_tags(lang, "pronoun")))


def get_ratio_of_verbs(sentence, lang):
    """
    :param sentence: spacy.tokens.doc.Doc object, can be either a sentence or multiple sentences.
    :return: ratio of universal part of speech tag "verb" per sentence
    """
    return min(1, get_ratio_of_part_of_speech_per_sent(sentence, get_pos_tags(lang, "verb")))


def get_ratio_of_punctuation(sentence, lang):
    return min(1, get_ratio_of_part_of_speech_per_sent(sentence, get_pos_tags(lang, "punct")))


def get_ratio_of_adjectives(sentence, lang):
    return min(1, get_ratio_of_part_of_speech_per_sent(sentence, get_pos_tags(lang, "adj")))


def get_ratio_of_adpositions(sentence, lang):
    return min(1, get_ratio_of_part_of_speech_per_sent(sentence, get_pos_tags(lang, "adp")))


def get_ratio_of_adverbs(sentence, lang):
    return min(1, get_ratio_of_part_of_speech_per_sent(sentence, get_pos_tags(lang, "adv")))


def get_ratio_of_auxiliary_verbs(sentence, lang):
    return min(1, get_ratio_of_part_of_speech_per_sent(sentence, get_pos_tags(lang, "aux")))


def get_ratio_of_determiners(sentence, lang):
    return min(1, get_ratio_of_part_of_speech_per_sent(sentence, get_pos_tags(lang, "det")))


def get_ratio_of_interjections(sentence, lang):
    return min(1, get_ratio_of_part_of_speech_per_sent(sentence, get_pos_tags(lang, "intj")))


def get_ratio_of_numerals(sentence, lang):
    return min(1, get_ratio_of_part_of_speech_per_sent(sentence, get_pos_tags(lang, "num")))


def get_ratio_of_particles(sentence, lang):
    return min(1, get_ratio_of_part_of_speech_per_sent(sentence, get_pos_tags(lang, "part")))


def get_ratio_of_symbols(sentence, lang):
    return min(1, get_ratio_of_part_of_speech_per_sent(sentence, get_pos_tags(lang, "sym")))


def get_ratio_of_space(sentence, lang):
    return min(1, get_ratio_of_part_of_speech_per_sent(sentence, get_pos_tags(lang, "space")))


def get_pos_tags(lang, type):
    tags = []
    if type == "verb":
        if lang == "de":
            tags = ["vmfin", "vminf", "vmpp", "vvfin", "vvimp", "vvinf", "vvizu", "vvpp", "verb"]
        elif lang == "en":
            tags = ["vb", "vbd", "vbg", "vbn", "vbp", "vbz", "verb"]
        else:
            tags = ["verb"]
    elif type == "pronoun":
        tags = ["pron"]
        if lang == "de":
            tags.extend(["pds", "pis", "pper", "pposs", "prels", "prf", "pws"])
        elif lang == "en":
            tags.extend(["ex", "prp", "wp"])
    elif type == "noun":
        if lang == "de":
            tags = ["ne", "nn", "nne", "pds", "pis", "pper", "pposs", "prels", "prf", "pws", "noun", "propn",
                         "pron"]
        elif lang == "en":
            tags = ["ex", "nn", "nnp", "nnps", "nns", "prp", "wp", "noun", "propn", "pron"]
        else:
            tags = ["noun", "propn", "pron"]
    elif type == "adj":
        tags = ["adj"]
        if lang == "en":
            tags.extend(["afx", "jj", "jjr", "jjs"])
        elif lang == "de":
            tags.extend(["adja", "adjd"])
    elif type == "adp":
        tags = ["adp"]
        if lang == "en":
            tags.extend(["in", "rp"])
        elif lang == "de":
            tags.extend(["appo", "appr", "apprart", "apzr", "ptkzu"])
    elif type == "adv":
        tags = ["adv"]
        if lang == "en":
            tags.extend(["rb", "rbr", "rbs", "wrb"])
        elif lang == "de":
            tags.extend(["proav", "pwav"])
    elif type == "aux":
        tags = ["aux"]
        if lang == "en":
            tags.extend(["md"])
        elif lang == "de":
            tags.extend(["vafin", "vaimp", "vainf", "vapp"])
    elif type == "det":
        tags = ["det"]
        if lang == "en":
            tags.extend(["dt", "pdt", "prp$", "wdt", "wp$"])
        elif lang == "de":
            tags.extend(["art", "pdat", "piat", "pposat", "prelat", "pwat"])
    elif type == "conj":
        tags = ["conj", "cconj", "sconj"]
        if lang == "en":
            tags.extend(["cc", "in"])
        elif lang == "de":
            tags.extend(["kokom", "kon", "koui", "kous"])
    elif type == "sconj":
        tags = ["sconj"]
        if lang == "de":
            tags.extend(["koui", "kous"])
    elif type == "intj":
        tags = ["intj"]
        if lang == "en":
            tags.extend(["uh"])
        elif lang == "de":
            tags.extend(["itj"])
    elif type == "num":
        tags = ["num"]
        if lang == "en":
            tags.extend(["cd"])
        elif lang == "de":
            tags.extend(["card"])
    elif type == "part":
        tags = ["part"]
        if lang == "en":
            tags.extend(["pos", "to"])
        elif lang == "de":
            tags.extend(["ptka", "ptkant", "ptkneg", "ptkzu", ])
    elif type == "sym":
        tags = ["sym"]
        if lang == "en":
            tags.extend(["$"])
    elif type == "oov":
        tags = ["x"]
        if lang == "en":
            tags.extend(["add", "xx", "nil", "ls", "fw", "gw"])
        elif lang == "de":
            tags.extend(["trunc", "fm"])
    elif type == "space":
        tags = ["space"]
        if lang == "en":
            tags.extend(["sp", "_sp"])
        elif lang == "de":
            tags.extend(["_sp"])
    elif type == "referential":
        tags = ["ex"]
        if lang == "en":
            tags.extend(["ex"])
    elif type == "punct":
        tags = ["punct"]
        if lang == "en":
            tags.extend(["``", "''", ",", "-lrb-", "-rrb-", ".", ":", "hyph", "nfp"])
        elif lang == "de":
            tags.extend(["$(", "$,", "$."])
    return tags


def get_dependency_label(lang, type):
    """ mapping from here https://spacy.io/api/annotation#dependency-parsing-universal"""
    if type == "subclause":
        tags = ["csubj", "xcomp", "ccomp", "advcl", "acl"]
        if lang == "de":
            tags.extend(["cc", "cp", "ccp", "mo", "rc"])
    elif type == "conj":
        tags = ["cc", "conj"]
        if lang == "de":
            tags.extend(["cm", "cp",  "cd"])
    elif type == "coordclause":
        tags = ["conj"]
        if lang == "de":
            tags.extend(["cm", "cp", "cd"])
    elif type == "subject":
        tags = ["nsubj", "csubj"]
        if lang == "de":
            tags.extend(["sb", "sbp", "sp"])
    elif type == "subordclause":
        tags = ["csubj", "xcomp", "ccomp", "advcl", "acl"]
        if lang == "de":
            tags.extend(["cp", "ccp"])
    elif type == "functionwords":
        tags = ["aux", "cop", "mark", "case"]
        if lang == "de":
            tags.extend(["oc", "pd", "ac", "dm"])
    elif type == "case":
        tags = ["case"]
        if lang == "de":
            tags.extend(["ac", "mnr"])
    elif type == "relative":
        tags = ["acl", "acl:relcl"]
        if lang == "en":
            tags.extend(["relcl"])
        if lang == "de":
            tags.extend(["rc"])
    elif type == "passive":
        tags = ["auxpass"]
        if lang == "de":
            tags.extend(["sbp"])
        elif tags == "en":
            tags.extend(["auxpass", "csubjpass", "nsubjpass"])  # en
    elif type == "orphan":
        tags = ["orphan"]
    elif type == "parataxis":
        tags = ["parataxis"]
        if lang == "de":
            tags.extend(["dl"])
    elif type == "mwe":
        tags = ["flat", "fixed", "compound"]
        if lang == "en":
            tags.extend(["nn"])
        elif lang == "de":
            tags.extend(["svp", "pnc", "cvc"])
    elif type == "repeat":
        tags = []
        if lang == "de":
            tags.extend(["re"])
    elif type == "referential":
        tags = ["expl"]
    else:
        tags = []
    return tags


def get_ratio_spelling_errors(sentence, lang):
    # todo
    return min(1, get_ratio_of_deprel_per_sent(sentence, ["goeswith"]))


def get_ratio_mwes(sentence, lang):
    return min(1, get_ratio_of_deprel_per_sent(sentence, get_dependency_label(lang, "mwe")))


def get_ratio_named_entities(sentence, lang):
    sent_len = 0
    n_tok = 0
    for sent in list(sentence.sents):
        sent_len += len(sent)
        n_tok += len([token for token in sent if token.ent_type])
    return safe_division(n_tok, sent_len)


#def get_ratio_ellipsis(sentence, lang):
#	# todo: https://universaldependencies.org/u/overview/specific-syntax.html#ellipsis
#	return get_ratio_of_deprel_per_sent(sentence, get_dependency_label(lang, "orphan"))


def get_ratio_of_subordinate_clauses(sentence, lang):
    """
    calculates ratio of subordinate clauses per sentence, as marker for them we use the universal dependency relations
    "csubj" (clausal subject), "xcomp" (open clausal component), "ccomp" (clausal complement), "advcl" (adverbial clause modifier),
    "acl" (clausal modifier of noun (adjectival clause)), Source: https://universaldependencies.org/u/overview/complex-syntax.html#subordination
    [Jan 2020]

    :param sentence: spacy.tokens.doc.Doc object, can be either a sentence or multiple sentences.
    :return: ratio of universal dependency relations describing subordinate phrases per sentence
    """
    sub_tags = get_dependency_label(lang, "subordclause")
    return min(1, get_ratio_of_deprel_per_sent(sentence, sub_tags))


def get_ratio_of_coordinating_clauses(sentence, lang):
    return min(1, get_ratio_of_part_of_speech_per_sent(sentence, get_pos_tags(lang, "sconj")))

#
# def get_verb_token_ratio(sentence, lang):
# 	"""
# 	:param sentence: spacy.tokens.doc.Doc object, can be either a sentence or multiple sentences.
# 	:return: ratio of all nouns and verbs per token per sentence
# 	"""
# 	verb_tags = get_pos_tags(lang, "verb")
# 	# noun_tags = get_pos_tags(lang, "noun")
# 	return min(1, get_ratio_of_part_of_speech_per_sent(sentence, verb_tags))


#def get_ratio_repeated_element(sentence, lang):
#	return get_ratio_of_deprel_per_sent(sentence, get_dependency_label(lang, "repeat"))


def get_ratio_clauses(sentence, lang):
    """
    orign of tags see here https://universaldependencies.org/u/overview/complex-syntax.html
    :param sentence: sentence: spacy.tokens.doc.Doc object, can be either a sentence or multiple sentences.
    :lang: language of the current used sample
    :return: ratio of clauses per sentence
    """
    sub_clause_tags = get_dependency_label(lang, "subclause")
    coord_clause_tags = get_dependency_label(lang, "coordclause")
    return min(1, get_ratio_of_deprel_per_sent(sentence, sub_clause_tags+coord_clause_tags))


def complex_sentence_is_split(complex_sent, simple_sent, lang):
    """
    Test if he complex sentence is split into at least two simple sentences. Test if 1-N simplification
    :param complex_sent: spacy.tokens.doc.Doc object, can be either a sentence or multiple sentences.
    :param simple_sent: spacy.tokens.doc.Doc object, can be either a sentence or multiple sentences.
    :return: boolean
    """
    return int(count_sentences(complex_sent, lang) < count_sentences(simple_sent, lang))


def complex_sentences_are_joined(complex_sent, simple_sent, lang):
    """
    Test if at least two complex sentences are jointed into one simple sentence. Test if N-1 simplification
    :param complex_sent: spacy.tokens.doc.Doc object, can be either a sentence or multiple sentences.
    :param simple_sent: spacy.tokens.doc.Doc object, can be either a sentence or multiple sentences.
    :return: boolean
    """
    return int(count_sentences(complex_sent, lang) > count_sentences(simple_sent, lang))


def check_pos_of_head_of_sent(sentence, pos):
    """
    check if the syntactic head of the sentence has part of speech <pos>
    :param sentence: spacy.tokens.doc.Doc object, can be either a sentence or multiple sentences.
    :return: boolean
    """
    return sentence.root.pos_.lower() in pos


def check_if_head_is_verb(sentence, lang):
    """
    check if the syntactic head of the sentence is verb
    :param sentence: spacy.tokens.doc.Doc object, can be either a sentence or multiple sentences.
    :return:  number of sentences with head as verb.
    """
    verb_tag = get_pos_tags(lang, "verb")
    for sent in list(sentence.sents):
        if check_pos_of_head_of_sent(sent, verb_tag):
            return True
    return False


def check_if_head_is_noun(sentence, lang):
    """
    check if the syntactic head of the sentence is noun
    :param sentence: spacy.tokens.doc.Doc object, can be either a sentence or multiple sentences.
    :return: number of sentences with head as noun.
    """
    noun_tags = get_pos_tags(lang, "noun")
    for sent in list(sentence.sents):
        if check_pos_of_head_of_sent(sent, noun_tags):
            return True
    return False


def check_if_one_child_of_root_is_subject(sentence, lang):
    subject_tags = get_dependency_label(lang, "subject")
    # number_subject_root = 0
    all_sents_subj = False
    for sent in list(sentence.sents):
        all_sents_subj = False
        for child in sent.root.children:
            if child in get_token_with_deprel(sentence, subject_tags):
                # number_subject_root += 1
                all_sents_subj = True
                break
    return all_sents_subj


def no_change(complex_sent, simple_sent, lang):
    return int([token.text for token in complex_sent] == [token.text for token in simple_sent])


def passive_to_active(complex_sent, simple_sent, lang):
    if check_passive_voice(complex_sent, lang) and not check_passive_voice(simple_sent, lang):
        return 1
    else:
        return 0


def non_projective_to_projective(complex_sent, simple_sent, lang):
    if is_non_projective(complex_sent, lang) and not is_non_projective(simple_sent, lang):
        return 1
    else:
        return 0


def reduction_of_parse_tree_height(complex_sent, simple_sent, lang):
    if get_parse_tree_height(complex_sent, lang) > get_parse_tree_height(simple_sent, lang):
        return 1
    else:
        return 0


def word_length_reduced(complex_sent, simple_sent, lang):
    if count_characters_per_word(complex_sent, lang) > count_characters_per_word(simple_sent, lang):
        return 1
    else:
        return 0


def syntactic_simplification(complex_sent, simple_sent, lang):
    if complex_sentence_is_split(complex_sent, simple_sent, lang=lang):
        return 1
    elif complex_sentences_are_joined(complex_sent, simple_sent, lang=lang):
        return 1
    elif non_projective_to_projective(complex_sent, simple_sent, lang):
        return 1
    elif reduction_of_parse_tree_height(complex_sent, simple_sent, lang):
        return 1
    elif passive_to_active(complex_sent, simple_sent, lang):
        return 1
    return 0


def lexical_simplification(complex_sent, simple_sent, lang):
    if get_rewritten_words_proportion(complex_sent, simple_sent, lang) != 0:
        return 1
    elif get_added_words_proportion(complex_sent, simple_sent, lang) != 0:
        return 1
    else:
        return 0


def get_ratio_referential(sentence, lang):
    # todo, values higher than 1?
    avg = list()
    for sent in list(sentence.sents):
        tokens = set()
        tokens = tokens.union(set(get_token_id_with_part_of_speech_tag(sent, get_pos_tags(lang, "referential"))))
        tokens = tokens.union(set(get_token_id_with_deprel(sent, get_dependency_label(lang, "referential"))))
        avg.append(len(tokens)/len(sent))
    return min(1, np.mean(avg))


def get_type_token_ratio(sentence, lang):
    return min(1, safe_division(len(set(to_lemma(sentence, lang))),len(sentence)))


def ratio_out_of_vocab(sentence, lang):
# 	if len(list(sentence.sents)) > 1:
# 		return [token for sent in sentence.sents for token in sent if token.is_oov]
    avg = list()
    for sent in list(sentence.sents):
        tokens = set()
        tokens = tokens.union(set(get_token_id_with_part_of_speech_tag(sent, get_pos_tags(lang, "oov"))))
        tokens = tokens.union(set([token for token in sent if token.like_email or token.like_url]))
        avg.append(len(tokens)/len(sent))
    return min(1, np.mean(avg))


def is_non_projective(sentence, lang):
    """if at least one of the sentence in the complex or simple sentences is non-projective True will be returned"""
    heads = [token.head.i for token in sentence]
    #print(heads, list(sentence.sents))
    is_non_projective_value = spacy.syntax.nonproj.is_nonproj_tree(heads)
    return int(is_non_projective_value)


def ratio_non_projective_arcs(sentence, lang):
    n = 0
    heads = [token.head.i for token in sentence]
    for head in heads:
        if spacy.syntax.nonproj.is_nonproj_arc(head, heads):
            n += 1
    return min(1, n/len(sentence))


def ratio_non_projective_arcs_per_sent(sentence, lang):
    ratios = list()
    for sent in list(sentence.sents):
        ratios.append(ratio_non_projective_arcs(sent, lang))
    return min(1, np.mean(ratios))



def get_ratio_prepositional_phrases(sentence, lang):
    return min(1, get_ratio_of_deprel_per_sent(sentence, get_dependency_label(lang, "case")))


def get_ratio_of_conjunctions(sentence, lang):
    avg = list()
    for sent in list(sentence.sents):
        tokens = set()
        tokens = tokens.union(set(get_token_id_with_part_of_speech_tag(sent, get_pos_tags(lang, "conj"))))
        tokens = tokens.union(set(get_token_id_with_deprel(sent, get_dependency_label(lang, "conj"))))
        avg.append(len(tokens)/len(sent))
    return min(1, np.mean(avg))


def get_ratio_relative_phrases(sentence, lang):
    return min(1, get_ratio_of_deprel_per_sent(sentence, get_dependency_label(lang, "relative")))


def check_passive_voice(sentence, lang):
    """works only for DE and EN so far """
    if get_ratio_of_deprel_per_sent(sentence, get_dependency_label(lang, "passive")) > 0:
        return 1
    else:
        return 0


def get_average_length_NP(sentence, lang):
    return get_avg_length_phrase(sentence, lang, "noun")


def get_average_length_VP(sentence, lang):
    return get_avg_length_phrase(sentence, lang, "verb")


def get_avg_length_phrase(sentence, lang, type):
    phrase_length = list()
    for token in sentence:
        if len(list(token.children)) > 0 and token.pos_.lower() in get_pos_tags(lang, type):
            phrase_length.append(len([t for t in token.subtree]))
    return np.mean(phrase_length)


def get_avg_length_PP(sentence, lang):
    phrase_length = list()
    for token in sentence:
        if len(list(token.children)) > 0 and token.pos_.lower() in get_dependency_label(lang, "case"):
            phrase_length.append(len([t for t in token.subtree]))
    return np.mean(phrase_length)



def FKBLEU(sentence, lang):
    # todo, need reference, input and output sentence
    return 1


def get_sentence_simplification_feature_extractors():
    return [
        get_type_token_ratio,
        get_ratio_of_function_words,
        get_ratio_of_coordinating_clauses,
        get_ratio_of_subordinate_clauses,
        get_ratio_prepositional_phrases,
        get_ratio_relative_phrases,
        get_ratio_clauses,
        get_ratio_referential,
        get_ratio_named_entities,
        get_ratio_mwes,
        # get_verb_token_ratio, same as ratio verb
        get_parse_tree_height,
        check_if_head_is_noun,
        check_if_head_is_verb,
        check_if_one_child_of_root_is_subject,
        check_passive_voice,
        is_non_projective,
        get_average_length_NP,
        get_average_length_VP,
        get_avg_length_PP
    ] + get_pos_proportion_feature_extractors()


def get_pos_proportion_feature_extractors():
    return [
        get_ratio_of_nouns,  # incl proper nouns
        get_ratio_of_verbs,
        get_ratio_of_adjectives,
        get_ratio_of_adpositions,
        get_ratio_of_adverbs,
        get_ratio_of_auxiliary_verbs,
        get_ratio_of_conjunctions,  # incl coordinating conj, subordinating conjunction
        get_ratio_of_determiners,
        get_ratio_of_interjections,
        get_ratio_of_numerals,
        get_ratio_of_particles,
        get_ratio_of_pronouns,
        get_ratio_of_punctuation,
        get_ratio_of_symbols
    ]


def get_sentence_pair_simplification_feature_extractors():
    return [
        complex_sentences_are_joined,
        complex_sentence_is_split,
        syntactic_simplification,
        lexical_simplification,
        no_change,
        passive_to_active,
        non_projective_to_projective,
        reduction_of_parse_tree_height,
        word_length_reduced
    ]