import os
import numpy as np
import torch

## Creating a vocab and tokenizing the text. 
class Dictionary(object):
    def __init__(self):
        self.word2idx = {} # word: index
        self.idx2word = [] # position(index): word

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    '''
        Usage: corpus = Corpus(path). Your path must contain the keyword ptb or wiki. 
    '''
    def __init__(self, path):
        self.dictionary = Dictionary()
        if 'ptb' in path:
        # three tensors of word index
            self.train = self.tokenize(os.path.join(path, 'ptb.train.txt'))
            self.valid = self.tokenize(os.path.join(path, 'ptb.valid.txt'))
            self.test = self.tokenize(os.path.join(path, 'ptb.test.txt'))
            
        else: 
            
            self.train = self.tokenize(os.path.join(path, 'wiki.train.tokens'))
            self.valid = self.tokenize(os.path.join(path, 'wiki.valid.tokens'))
            self.test = self.tokenize(os.path.join(path, 'wiki.test.tokens'))
            
    
    def tokenize(self, path):
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                # line to list of token + eos
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

    
      
### A low put way to create a quick lazy dataloader. Split the data into batches
def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data

 ## Deatch to keep gradients flowing
def repackage_hidden(h):
    # detach
    return tuple(v.clone().detach() for v in h)

#seq length = 64. And create the minibatches with inputs and targets.
def get_batch(source, i):
    # source: size(total_len//bsz, bsz)
    seq_len = min(64, len(source) - 1 - i)
    #data = torch.tensor(source[i:i+seq_len]) # size(bptt, bsz)
    data = source[i:i+seq_len].clone().detach()
    target = source[i+1:i+1+seq_len].clone().detach().view(-1)
    #target = torch.tensor(source[i+1:i+1+seq_len].view(-1)) # size(bptt * bsz)
    return data, target

def word_count(corpus):
    counter = [0] * len(corpus.dictionary.idx2word)
    for i in corpus.train:
        counter[i] += 1
    for i in corpus.valid:
        counter[i] += 1
    for i in corpus.test:
        counter[i] += 1
    return np.array(counter).astype(int)

def word_freq_ordered(corpus):
    # Given a word_freq_rank, we could find the word_idx of that word in the corpus
    counter = word_count(corpus)
    # idx_order: freq from large to small (from left to right)
    idx_order = np.argsort(-counter)
    return idx_order.astype(int)

def word_rank_dictionary(corpus):
    # Given a word_idx, we could find the frequency rank (0-N, the smaller the rank, the higher frequency the word) of that word in the corpus
    idx_order = word_freq_ordered(corpus)
    # Reverse
    rank_dictionary = np.zeros(len(idx_order))
    for rank, word_idx in enumerate(idx_order):
        rank_dictionary[word_idx] = rank
    return rank_dictionary.astype(int)

class Rank_Indexed_Corpus(object):
    # Corpus using word rank as index. Can be used to train in place of the Corpus class.
    def __init__(self, corpus, word_rank):
        self.dictionary = self.convert_dictionary(dictionary = corpus.dictionary, word_rank = word_rank)
        self.train = self.convert_tokens(tokens = corpus.train, word_rank = word_rank)
        self.valid = self.convert_tokens(tokens = corpus.valid, word_rank = word_rank)
        self.test = self.convert_tokens(tokens = corpus.test, word_rank = word_rank)

    def convert_tokens(self, tokens, word_rank):
        rank_tokens = torch.LongTensor(len(tokens))
        for i in range(len(tokens)):
            #print(word_rank[tokens[i]])

            rank_tokens[i] = int(word_rank[tokens[i]])
        return rank_tokens

    def convert_dictionary(self, dictionary, word_rank):
        rank_dictionary = Dictionary()
        rank_dictionary.idx2word = [''] * len(dictionary.idx2word)
        for idx, word in enumerate(dictionary.idx2word):
            #print(word_rank)
            #print(rank)
            rank = word_rank[idx]
            rank_dictionary.idx2word[rank] = word
            if word not in rank_dictionary.word2idx:
                rank_dictionary.word2idx[word] = rank
        return rank_dictionary

