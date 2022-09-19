from collections import defaultdict
import string
import numpy as np

punct = set(string.punctuation)

noun_suffix = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness", "or", "ry", "scape", "ship", "ty"]
verb_suffix = ["ate", "ify", "ise", "ize"]
adj_suffix = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ly", "ous"]
adv_suffix = ["ward", "wards", "wise"]

def build_vocab(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    tokens = [line.split('\t')[0] for line in lines]
    frequency = defaultdict(int)
    for token in tokens:
        frequency[token] += 1
    vocab = [k for k, v in frequency.items() if (v > 1 and k != '\n')]
    unk_toks = ["--unk--", "--unk_adj--", "--unk_adv--", "--unk_digit--", "--unk_noun--", "--unk_punct--", "--unk_upper--", "--unk_verb--"]
    vocab.extend(unk_toks)
    vocab.append("--n--")
    vocab.append(" ")
    vocab = sorted(set(vocab))
    return vocab

def build_vocab2idx(filepath):
    vocab = build_vocab(filepath)
    vocab2idx = {}
    for i, token in enumerate(sorted(vocab)):
        vocab2idx[token] = i
    return vocab2idx

def training_data(filepath):
    with open(filepath, 'r') as f:
        training_set = f.readlines()
    return training_set

def assign_unk(token):
    if any(char.isdigit() for char in token):
        return "--unk_digit--"
    elif any(char in punct for char in token):
        return "--unk_punct--"
    elif any(char.isupper() for char in token):
        return "--unk_upper--"
    elif any(token.endswith(suffix) for suffix in noun_suffix):
        return "--unk_noun--"
    elif any(token.endswith(suffix) for suffix in verb_suffix):
        return "--unk_verb--"
    elif any(token.endswith(suffix) for suffix in adj_suffix):
        return "--unk_adj--"
    elif any(token.endswith(suffix) for suffix in adv_suffix):
        return "--unk_adv--"
    return "--unk--"

def get_word_tag(line, vocab):
    if not line.split():
        word = "--n--"
        tag = "--s--"
        return word, tag
    else:
        word, tag = line.split()
        if word not in vocab:
            word = assign_unk(word)
        return word, tag

def create_dictionaries(training_set, vocab2idx):
    emission_dict = defaultdict(int)
    transition_dict = defaultdict(int)
    tag_counts = defaultdict(int)
    prev_tag = '--s--'
    for token_tag in training_set:
        token, tag = get_word_tag(token_tag, vocab2idx)
        transition_dict[(prev_tag, tag)] += 1
        emission_dict[(tag, token)] += 1
        tag_counts[tag] += 1
        prev_tag = tag
    return emission_dict, transition_dict, tag_counts

def create_transition_matrix(transition_dict, tag_counts, alpha):
    all_tags = sorted(tag_counts.keys())
    num_tags = len(all_tags)
    A = np.zeros((num_tags, num_tags))
    for i in range(num_tags):
        for j in range(num_tags):
            count = 0
            key = (all_tags[i], all_tags[j])
            if key in transition_dict:
                count = transition_dict[key]
            num_prev_tag = tag_counts[all_tags[i]]
            A[i, j] = (count + alpha) / (num_prev_tag + alpha * num_tags)
    return A

def create_emission_matrix(emission_dict, tag_counts, vocab_dict, alpha):
    num_tags = len(tag_counts)
    all_tags = sorted(tag_counts.keys())
    num_words = len(vocab_dict)
    B = np.zeros((num_tags, num_words))
    for i in range(num_tags):
        for j in range(num_words):
            count = 0
            key =  (all_tags[i], vocab_dict[j])
            if key in emission_dict:
                count = emission_dict[key]
            count_tag = tag_counts[all_tags[i]]
            B[i, j] = (count + alpha) / (count_tag + alpha * num_words)
    return B

def initialize(A, B, tag_counts, vocab_dict, states, prep_tokens):
    num_tags = len(tag_counts)
    probs = np.zeros((num_tags, num_tags))
    paths = np.zeros((num_tags, len(prep_tokens)), dtype=int)
    s_idx = states.index('--s--')
    for i in range(num_tags):
        if A[s_idx, i] == 0:
            probs[i, 0] = float('-inf')
        else:
            probs[i,0] = np.log(A[s_idx, i]) + np.log(B[i, vocab_dict[prep_tokens[0]]])
    return probs, paths

def viterbi_forward(A, B, prep_tokens, probs, paths, vocab_dict):
    num_tags = probs.shape[0]
    for i in range(1, len(prep_tokens)):
        for j in range(num_tags):
            prob_i = float('-inf')
            path_i = None
            for k in range(num_tags):
                prob = probs[k,i-1]+np.log(A[k,j]) +np.log(B[j,vocab_dict[prep_tokens[i]]])
                if prob > prob_i:
                    prob_i = prob
                    path_i = k
            probs[j, i] = prob_i
            paths[j, i] = path_i
    return probs, paths

def viterbi_backward(probs, paths, tags):
    m = paths.shape[1]
    z = [None] * m
    num_tags = probs.shape[0]
    prob_last = float('-inf')
    pred = [None] * m
    for k in range(num_tags):
        if probs[k, m - 1] > prob_last:
            prob_last = probs[k, m - 1]
            z[m - 1] = k
    pred[m - 1] = tags[z[m - 1]]
    for i in range(m-1, -1, -1):
        pos_tag = z[i]
        z[i - 1] = paths[pos_tag,i]
        pred[i - 1] = tags[z[i - 1]]
    return pred

def processing(vocab, text):
    processed_sentence = []
    for word in text:
        if not word.split():
            word = "--n--"
            processed_sentence.append(word)
            continue
        elif word.strip() not in vocab:
            word = assign_unk(word)
            processed_sentence.append(word)
            continue
        else:
            processed_sentence.append(word.strip())
    assert(len(processed_sentence) == len(text))
    return processed_sentence