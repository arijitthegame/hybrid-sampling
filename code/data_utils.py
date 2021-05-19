import os
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
    class Corpus(object):
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
