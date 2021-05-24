import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
from data_utils import * 
import model
from sampled_softmax import *


parser = argparse.ArgumentParser(description='PyTorch RNN Language Model')
parser.add_argument('--data', type=str, default='./input', # /input
                    help='location of the data corpus, data path must contain the string ptb or wiki')
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--hidden_size', type=int, default=200)
parser.add_argument('--output_dim', type=int, default=128)
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--directions', type=int, default =2, help = 'Choose a bilSTM vs a LSTM')
parser.add_argument('--dropout', type=float, default=.5)
parser.add_argument('--lr', type=float, default=.001)
parser.add_argument('--weight_decay', type=float, default=.0001)
parser.add_argument('--clip', type=float, default=1.0)
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--interval', type=int, default=200)
parser.add_argument('--train_batch_size', type=int, default=32)
parser.add_argument('--eval_batch_size', type=int, default=10)
parser.add_argument('--bptt', type=int, default=64)
parser.add_argument('--softmax_type', type=str, default='full', help='full, sampled')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--softmax_nsampled', type=int,  default=100,
                    help='number of random sample generated for sampled softmax')
parser.add_argument('--save', type=str,  default='./output/model_test.pt')
parser.add_argument('--device', type=str, default='cpu', help='cpu, cuda')

args = parser.parse_args()

torch.manual_seed(42)
#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device(args.device)

if args.softmax_type=='full':
    corpus = Corpus(args.data)

if args.softmax_type=='sampled':
    corpus_raw = Corpus(args.data)

# word_rank: list idx is the word_idx, list content value is the frequency rank
    word_rank = word_rank_dictionary(corpus = corpus_raw)

# We use the word_rank as the input to the model
    corpus = Rank_Indexed_Corpus(corpus = corpus_raw, word_rank = word_rank)

train_data = batchify(corpus.train, args.train_batch_size) # size(total_len//bsz, bsz)
val_data = batchify(corpus.valid, args.eval_batch_size)
test_data = batchify(corpus.test, args.eval_batch_size)

ntokens = len(corpus.dictionary)

#Initilaize the model 
net = model.RNNModel(ntokens, args.hidden_size, args.output_dim, args.n_layers, args.directions,args.dropout)
net.to(device)
if args.softmax_type=='sampled':

    sampled_softmax = SampledSoftmax(ntokens, nsampled=args.softmax_nsampled, nhid=args.output_dim*args.directions, tied_weight = None)
    net.decoder = model.Identity()
    net.add_module("decoder_softmax", sampled_softmax)
### For more usage of the weight tying, please see the trainer.ipynb

opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
criterion = nn.CrossEntropyLoss()

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    with torch.no_grad():
        net.eval()
        total_loss = 0
        ntokens = len(corpus.dictionary)
        hidden = net.init_hidden(args.eval_batch_size, args.directions) #hidden size(nlayers, bsz, hdsize)
        for i in range(0, data_source.size(0) - 1, args.bptt):# iterate over every timestep
            data, targets = get_batch(data_source, i)
            data, targets = data.to(device), targets.to(device)
            output, hidden = net(data, hidden)
            # model input and output
            # inputdata size(bptt, bsz), and size(bptt, bsz, embsize) after embedding
            # output size(bptt*bsz, ntoken)
            if args.softmax_type=='sampled':
                logits = sampled_softmax.full(output)
                output = logits.view(-1, logits.size(-1))

            total_loss += len(data) * criterion(output, targets).data
            hidden = repackage_hidden(hidden)
        return total_loss / len(data_source)

def train():

    net.train()
    total_loss = 0
    start_time = time.time()
   
    hidden = net.init_hidden(args.train_batch_size, args.directions)
   
    # train_data size(batchcnt, bsz)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        data, targets = data.to(device), targets.to(device)
        
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        
      
        output, hidden = net(data, hidden)
    
        if args.softmax_type=='sampled':
            logits, new_targets = sampled_softmax(output, targets)
            
            loss = criterion(logits.view(-1, args.softmax_nsampled+1), new_targets)
        if args.softmax_type=='full':

            loss = criterion(output, targets)
        opt.zero_grad()
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
        opt.step()

        total_loss += loss.data
        

        if batch % args.interval == 0 and batch > 0:
            cur_loss = total_loss / args.interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // 64,
                elapsed * 1000 / args.interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

# Loop over epochs.

best_val_loss = None

try:
    for epoch in range(1, args.n_epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            
            with open(args.save, 'wb') as f:
                torch.save(net.state_dict(), f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
#             if args.opt == 'SGD' or args.opt == 'Momentum':
#                 lr /= 4.0
#                 for group in opt.param_groups:
#                     group['lr'] = lr
            pass

###TODO Anneal learning rate 
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')


with open(args.save, 'rb') as f:
    net.load_state_dict(torch.load(f))

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
