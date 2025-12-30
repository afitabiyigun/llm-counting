import os, json, random
import torch
import plotly.express as px
from nnsight import CONFIG, LanguageModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"
CONFIG.API.APIKEY = os.getenv("NNSIGHT_API_KEY")

llm = LanguageModel("meta-llama/Meta-Llama-3.1-70B", device_map="auto")

with open("dataset_generation/paired_dataset.json") as f:
    data = json.load(f)

example = random.choice(data)

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

# answer token ids (assumes single-token answers like " 0"..." 9")
source_id = llm.tokenizer(f" {example['SourceAnswer']}")["input_ids"][1]
base_id   = llm.tokenizer(f" {example['BaseAnswer']}")["input_ids"][1]

num_layers = llm.config.num_hidden_layers

# IMPORTANT: base/source prompts can tokenize to different lengths
base_ids_prompt   = llm.tokenizer(base_prompt).input_ids
source_ids_prompt = llm.tokenizer(source_prompt).input_ids
num_tokens = min(len(base_ids_prompt), len(source_ids_prompt))

# We'll fill this with SavedNodes (remote handles)
causal_effects = [[None for _ in range(num_tokens)] for _ in range(num_layers)]

with torch.no_grad():
    with llm.trace(remote=True) as tracer:

        # 1) Source run: save per-layer hidden states
        with tracer.invoke(source_prompt):
            source_hs = [layer.output[0].save() for layer in llm.model.layers]

        # 2) Patch and measure
        for layer_idx in range(num_layers):
            for token_idx in range(num_tokens):
                with tracer.invoke(base_prompt):
                    # patch: base resid at (layer_idx, token_idx) <- source resid
                    llm.model.layers[layer_idx].output[0][:, token_idx, :] = \
                        source_hs[layer_idx][:, token_idx, :]

                    logits = llm.output.logits[:, -1, :]
                    probs  = logits.softmax(dim=-1)

                    # assign directly so we never reference an undefined `diff`
                    causal_effects[layer_idx][token_idx] = \
                        (probs[0, source_id] - probs[0, base_id]).save()

# pull back to local floats
effects_matrix = torch.tensor(
    [[causal_effects[i][j].value for j in range(num_tokens)] for i in range(num_layers)],
    dtype=torch.float32
)

# x-axis token strings (from BASE prompt, truncated to num_tokens)
token_strs = [llm.tokenizer.decode([tid]) for tid in base_ids_prompt[:num_tokens]]

fig = px.imshow(
    effects_matrix.numpy(),
    x=token_strs,
    y=list(range(num_layers)),
    aspect="auto",
    template="simple_white",
    labels=dict(x="Prompt token", y="Layer", color="ΔP(source)-ΔP(base)")
)
fig.write_html("causal_mediation/causal_effects2.html")
print("Wrote causal_mediation/causal_effects2.html")