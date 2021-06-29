Some caveats about the LSTM models: For LM tasks we will never use BiLSTM. It is only there for completeness. 
<br/>
The initrange inside the model class needs to be .1 for tied weights and 0.05 otherwise. 
<br/>
The dropout should be .2 for tied weights and .5 otherwise. 
<br/>
If weights are tied, embedding dim = output dim. Choose that dim = 200
<br/>
For the purposes of our paper, we will only use weight tied model. All other functionalities exist for completeness. 
<br/>
TODO: Pass a flag allowing us to choose initrange in the main.py
<br/>
TODO: Annealing of the learning rates as is recommended.
<br/>
Small bug in SWD computation. #TODO Push the fix. 
<br/>
Add the Walker's sampling method
