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
# %% helper functions
def flatten_ids(ids):
    if isinstance(ids[0], list):
        return [id for sublist in ids for id in sublist]
    return ids

# %% CMA 

with open("dataset_generation/paired_dataset.json") as f:
    data = json.load(f)

total = 1
subset = random.sample(data, k=total)
intervened_prob_diff = []

for example in subset:
    prefix = (
        "Count the number of words in the following list that match the given type, and put the numerical answer in parentheses.\n"
        f"Type: {example['Type']}\n"
        "List: "
    )
    suffix = "\nAnswer: ("

    base_prompt   = prefix + f"{example['BaseList']}" + suffix
    source_prompt = prefix + f"{example['SourceList']}" + suffix

    # answer token ids (single token ids, with leading space)
    source_id = llm.tokenizer(f" {example['SourceAnswer']}")["input_ids"][-1]
    base_id   = llm.tokenizer(f" {example['BaseAnswer']}")["input_ids"][-1]

    # prompt token ids (robust to batched vs unbatched)
    base_prompt_ids   = flatten_ids(llm.tokenizer(base_prompt)["input_ids"])
    source_prompt_ids = flatten_ids(llm.tokenizer(source_prompt)["input_ids"])

    # start = tokens up through "List: "
    prefix_ids = flatten_ids(llm.tokenizer(prefix)["input_ids"]) # shared text
    start_idx = len(prefix_ids) # where the first list token appears

    # only iterate over token positions that exist in BOTH prompts
    num_tokens = min(len(base_prompt_ids), len(source_prompt_ids))
    token_strs = [llm.tokenizer.decode([tid]) for tid in base_prompt_ids[:num_tokens]]
    num_layers = llm.config.num_hidden_layers

    source_hs = []
    with torch.no_grad():
        with llm.trace(source_prompt, remote=True):
            source_hs = [layer.output[0] for layer in llm.model.layers].save() # save the hidden states of the last layer of each layer
            # source_output = llm.output.save() # saving the output of the model
        
    causal_effects = []
    intervened_prob_diff = []

    for layer_idx in range(0, num_layers, 10): # iterating through every 5 layers: llm.config.num_hidden_layers
        # print(f"Layer {layer_idx} of {num_layers}")
        causal_effect_per_layer = []
        for token_idx in range(start_idx, num_tokens): # iterating through all tokens: num_tokens
            with torch.no_grad():
                with llm.trace(base_prompt, remote=True):
                    # changing the value of the base activation to the source value
                    llm.model.layers[layer_idx].output[0][token_idx, :] = \
                        source_hs[layer_idx][token_idx, :]

                    # getting intervened output & comparing to base output
                    intervened_logits = llm.output.logits[:, -1, :]
                    intervened_probs = intervened_logits.softmax(dim=-1)

                    intervened_prob_diff = (intervened_probs[0, source_id] - intervened_probs[0, base_id]).item().save()

                causal_effect_per_layer.append(intervened_prob_diff)
                
        causal_effects.append(causal_effect_per_layer)
        
#%% plotting the causal effects
# token_strs = [llm.tokenizer.decode([tid]) for tid in base_prompt_ids[:num_tokens]]

y_labels = list(range(0, num_layers, 10))              # rows = layers you actually ran
x_labels = token_strs[:len(causal_effects[0])]              # cols = tokens you actually ran

# Normalize causal_effects to 0-1 range for proper color scaling
# Find min and max across all layers and tokens
all_values = [val for layer in causal_effects for val in layer]
min_val = min(all_values) if all_values else 0
max_val = max(all_values) if all_values else 1

# Normalize to 0-1 range (handle case where min == max)
if max_val - min_val > 0:
    causal_effects_normalized = [
        [(val - min_val) / (max_val - min_val) for val in layer]
        for layer in causal_effects
    ]
else:
    causal_effects_normalized = causal_effects

print(f"Causal effects range: [{min_val:.4f}, {max_val:.4f}]")
print(f"Normalized to [0.0, 1.0]")

fig = px.imshow(
    causal_effects_normalized,
    x=x_labels,
    y=y_labels,
    template="simple_white", 
    color_continuous_scale=[[0, '#FFFFFF'], [1, "#A59FD9"]],
    zmin=0,
    zmax=1
)

fig.update_layout(
    xaxis_title='token',
    yaxis_title='layer',
    yaxis=dict(autorange='min')
)

fig.write_html("cma/causal_effects/every10layers_normalized.html")