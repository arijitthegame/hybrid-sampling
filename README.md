Code to run hybrid sampling schemes to compare against the softmax sampling. All experiments are run on langauge modeling tasks. The datasets used are Penn Tree Bank, Wikitext 2 and Wikitext 103. We will just use a LSTM/biLSTM for all the experiments. The goal here is to prove the efficacy of these hybrid schemes and not the model itself. 
<br/>
Added: sampled softmax. This is a simple reimplementation of https://www.tensorflow.org/api_docs/python/tf/nn/sampled_softmax_loss
<br/>

The code needs cleanup. Will combine the LSTMs in the near future by removing the directions arguments.
