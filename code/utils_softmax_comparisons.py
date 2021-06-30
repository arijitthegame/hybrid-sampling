import torch 
from data_utils import *


#These functions needs a pretrained model. Load the model first and pass in the model in net. 
# You also need the dataset to be loaded as well. You will need the test set as the data source and the corpus for the whole dataset.


def get_model_embeddings(data_source, net):
    # Turn on evaluation mode which disables dropout.
    with torch.no_grad():
        net.eval()
       
        ntokens = len(corpus_raw.dictionary)
        hidden = net.init_hidden(eval_batch_size) #hidden size(nlayers, bsz, hdsize)
        model_out = []
        for i in range(0, data_source.size(0) - 1, 64):# iterate over every timestep
            data, targets = get_batch(data_source, i)
            data, targets = data.to(device), targets.to(device)
            
            emb = net.encoder(data)
          
            output, hidden = net.rnn(emb, hidden)
            
            model_out.append(output.reshape(-1, output.shape[-1])
            # model input and output
            # inputdata size(bptt, bsz), and size(bptt, bsz, embsize) after embedding
            # output size(bptt*bsz, ntoken)
            
        return torch.cat((model_out), dim=0)
      
      
def get_true_softmax_scores(data_source):
    # Turn on evaluation mode which disables dropout.
    with torch.no_grad():
        net.eval()
       
        ntokens = len(corpus_raw.dictionary)
        hidden = net.init_hidden(eval_batch_size) #hidden size(nlayers, bsz, hdsize)
        test_output = []
        for i in range(0, data_source.size(0) - 1, 64):# iterate over every timestep
            data, targets = get_batch(data_source, i)
            data, targets = data.to(device), targets.to(device)
          
            output, hidden = net(data, hidden)
            
            test_output.append(output.reshape(-1, ntokens))
            
        total_output = torch.cat((test_output), dim=0)
            
        return F.softmax(total_output, dim=1)
      
  def get_class_embeddings(net):
    
    classes = torch.tensor([list(corpus_raw.dictionary.word2idx.values())]).squeeze()
    embeddings = net.encoder(classes)
    
    return embeddings