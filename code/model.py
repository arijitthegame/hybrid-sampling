import torch
import torch.nn as nn
from torch.autograd import Variable


## A simple LSTM or bilSTM model
class RNNModel(nn.Module):

    def __init__(self, ntoken, ninp, nhid, nlayers, directions, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp) # Token2Embeddings
        self.directions = directions
        
        if self.directions == 1: 
            self.rnn = nn.LSTM(ninp, nhid, nlayers, bidirectional=False, dropout=dropout) #(seq_len, batch_size, emb_size)
            if tie_weights:
                self.decoder = nn.Linear(nhid, ntoken, bias=False)
            else:
                self.decoder = nn.Linear(nhid, ntoken)
            
        elif self.directions == 2:
            self.rnn = nn.LSTM(ninp, nhid, nlayers, bidirectional=True, dropout=dropout) #(seq_len, batch_size, emb_size)
            if tie_weights:
                self.decoder = nn.Linear(2*nhid, ntoken, bias=False)
            else: 
                self.decoder = nn.Linear(2*nhid, ntoken)
                
        
        else:  
            print("Invalid number of directions. Can not initialize model.")
            
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
                
            self.decoder.weight = self.encoder.weight
            
       # self.decoder = nn.Linear(2*nhid, ntoken)
        self.init_weights()
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        # For tied weights initrange = .1
        initrange = 0.05
        self.encoder.weight.data.uniform_(-initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
       # nn.init.zeros_(self.decoder.bias)
       # self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        if self.decoder.bias is not None:
            torch.nn.init.zeros_(self.decoder.bias)
        #self.decoder.bias.data.fill_(0)

    def forward(self, input, hidden):
        # input size(bptt, bsz)
        embedding = self.encoder(input)
        emb = self.drop(embedding)
        # emb size(bptt, bsz, embsize)
        # hid size(layers, bsz, nhid)
        output, hidden = self.rnn(emb, hidden)
        # output size(bptt, bsz, nhid)
        output = self.drop(output)
        # decoder: nhid -> ntoken
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded, hidden


    def init_hidden(self, bsz, directions):
        if directions == 2:
            h, c = (Variable(torch.zeros(self.nlayers * 2, bsz, self.nhid)),
                Variable(torch.zeros(self.nlayers * 2, bsz, self.nhid)))
            
        else: 
            h, c = (Variable(torch.zeros(self.nlayers, bsz, self.nhid)),
                Variable(torch.zeros(self.nlayers, bsz, self.nhid)))
            
        return h, c


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
