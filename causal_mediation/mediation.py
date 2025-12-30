#%% ### imports
import os
from getpass import getpass
from nnsight import CONFIG 
from nnsight import LanguageModel # instantiate the model using the LanguageModel class
from IPython.display import clear_output
import json, re
import random
import torch
import plotly.express as px

#%% ### setting the API key
CONFIG.API.APIKEY = os.getenv("NNSIGHT_API_KEY") 
clear_output() 
# %%
#%% ### instantiating the model
llm = LanguageModel("meta-llama/Meta-Llama-3.1-70B", device_map="auto")
# %% setting base and source prompts

with open("dataset_generation/paired_dataset.json") as f:
    data = json.load(f)

total = 1
subset = random.sample(data, k=total)
intervened_prob_diff = []

for example in subset:
    base_prompt = (
        "Count the number of words in the following list that match the given type, and put the numerical answer in parentheses.\n"
        f"Type: {example['Type']}\n"
        f"List: {example['BaseList']}\n"
        "Answer: ("
    )

    source_prompt = (
        "Count the number of words in the following list that match the given type, and put the numerical answer in parentheses.\n"
        f"Type: {example['Type']}\n"
        f"List: {example['SourceList']}\n"
        "Answer: ("
    )

    source_ids = llm.tokenizer(f" {example['SourceAnswer']}")["input_ids"][1]
    base_ids   = llm.tokenizer(f" {example['BaseAnswer']}")["input_ids"][1]
    num_layers = len(llm.model.layers)
    base_len = len(llm.tokenizer(base_prompt).input_ids)
    
    source_hs = []
    with torch.no_grad():
        with llm.trace(source_prompt, remote=True):
            source_hs = [layer.output[0].save() for layer in llm.model.layers] # save the hidden states of the last layer of each layer
            # source_output = llm.output.save() # saving the output of the model
            
    causal_effects = []
    intervened_prob_diff = []

    num_tokens = len(llm.tokenizer(base_prompt).input_ids) # get number of tokens in prompt

    for layer_idx in range(llm.config.num_hidden_layers): # iterating through all layers
        causal_effect_per_layer = []
        for token_idx in range(num_tokens): # iterating through all tokens
            with torch.no_grad():
                with llm.trace(base_prompt, remote=True):
                    # changing the value of the base activation to the source value
                    llm.model.layers[layer_idx].output[0][:, token_idx, :] = \
                        source_hs[layer_idx][:, token_idx, :]

                    # getting intervened output & comparing to base output
                    intervened_logits = llm.output.logits[:, -1, :]
                    intervened_probs = intervened_logits.softmax(dim=-1)

                    intervened_prob_diff = (intervened_probs[0, source_ids] - intervened_probs[0, base_ids]).item().save()

                causal_effect_per_layer.append(intervened_prob_diff)
                
        causal_effects.append(causal_effect_per_layer)
        
#%% plotting the causal effects
token_strs = [llm.tokenizer.decode([tid]) for tid in base_ids[:num_tokens]]

fig = px.imshow(
    causal_effects,
    x=token_strs,
    y=list(range(llm.config.num_hidden_layers)),
    template="simple_white",
)
fig.write_html("causal_mediation/causal_effects.png", scale=2)