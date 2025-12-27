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

total = 20
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

    num_tokens = len(llm.tokenizer(base_prompt).input_ids) # get number of tokens in prompt

    with torch.no_grad():
        with llm.trace(remote=True) as tracer:
            with tracer.invoke(source_prompt):
                source_hs = [layer.output[0].save() for layer in llm.model.layers] # save the hidden states of the last layer of each layer
                source_output = llm.output.save() # saving the output of the model

            source_id = llm.tokenizer(f" {example['SourceAnswer']}")["input_ids"][1]
            base_id   = llm.tokenizer(f" {example['BaseAnswer']}")["input_ids"][1]
            
        causal_effects = []
        for layer_idx in range(llm.config.num_hidden_layers): # iterating through all layers
            causal_effect_per_layer = []
            for token_idx in range(num_tokens): # iterating through all tokens
                    with tracer.invoke(base_prompt):
                        # changing the value of the base activation to the source value
                        llm.model.layers[layer_idx].output[0][:, token_idx, :] = \
                            source_hs[layer_idx][:, token_idx, :]

                        # getting intervened output & comparing to base output
                        intervened_logits = llm.output.logits[:, -1, :]
                        intervened_probs = intervened_logits.softmax(dim=-1)

                        intervened_prob_diff = (intervened_probs[0, source_id] - intervened_probs[0, base_id]).item().save()

                    causal_effect_per_layer.append(intervened_prob_diff)
            causal_effects.append(causal_effect_per_layer)
        

#%% plotting the causal effects
fig = px.imshow(
    causal_effects,
    x=llm.tokenizer(base_prompt).input_ids,
    y=list(range(llm.config.num_hidden_layers)),
    template='simple_white',
    color_continuous_scale=[[0, '#FFFFFF'], [1, "#A59FD9"]]
)

fig.update_layout(
    xaxis_title='token',
    yaxis_title='layer',
    yaxis=dict(autorange='min')
)

fig