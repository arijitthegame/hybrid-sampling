Some caveats about the LSTM models: For LM tasks we will never use BiLSTM. It is only there for completeness. 
<br/>
The initrange inside the model class needs to be .1 for tied weights and 0.05 otherwise. 
<br/>
The dropout should be .2 for tied weights and .5 otherwise. 
<br/>
TODO: Annealing of the learning rates as is recommended.
