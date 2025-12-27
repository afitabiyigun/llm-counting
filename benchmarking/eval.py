#%% ### imports
import os
from getpass import getpass
from nnsight import CONFIG 
from nnsight import LanguageModel # instantiate the model using the LanguageModel class
from IPython.display import clear_output
import json, re
import random
#%% ### setting the API key
CONFIG.API.APIKEY = os.getenv("NNSIGHT_API_KEY") 
clear_output() 
# %%
#%% ### instantiating the model
llm = LanguageModel("meta-llama/Meta-Llama-3.1-70B", device_map="auto")
# %% generating outputs
with open("dataset_generation/dataset.json") as f:
    data = json.load(f)

# grabbing a random subset of TOTAL samples from the dataset each time
random.seed(42) 
total = 20
subset = random.sample(data, k=total) 

correct = 0
output_log = []

for example in subset:
    prompt = (
        "Count the number of words in the following list that match the given type, and put the numerical answer in parentheses.\n"
        f"Type: {example['Type']}\n"
        f"List: {example['List']}\n"
        "Answer: ("
    )
    expected = f"{example['Answer']})" if not str(example['Answer']).startswith("(") else str(example ['Answer'])

    # generating text using NNsight remotely 
    with llm.trace(prompt, remote=True):
        # should we also save internal activations here 
        output = llm.output.save() # saving the output of the model
    
    output_logits = output["logits"]
    # decode the final model output from output logits
    max_probs, tokens = output_logits[0].max(dim=-1) # max_probs is the maximum probability for each token, tokens is the token with the maximum probability
    word = [llm.tokenizer.decode(tokens.cpu()[-1])] # decode the token with the maximum probability using the tokenizer
    print("Model Output: ", word[0])

    match = re.search(r"\(?(\d+)", word[0]) # extracting the number from the model output, not as a list
    predicted = f"{match.group(1)})" if match else "N/A"

    print("\n" + "="*60)
    print("PROMPT:", prompt)
    print("\n RAW MODEL OUTPUT:", word)
    print("\n EXTRACTED ANSWER:", predicted)
    print("EXPECTED ANSWER:", expected)

    if predicted.strip() == expected.strip():
        correct += 1

    output_log.append({
        "Type": example["Type"],
        "List": example["List"],
        "Expected": expected,
        "RawOutput": word,
        "Predicted": predicted,
        "Match": predicted.strip() == expected.strip()
    })

with open("benchmarking/outputs/outputs20..json", "w") as out_file:
    json.dump(output_log, out_file, indent=2, ensure_ascii=False)

print(f"\nAccuracy: {correct}/{len(subset)} = {correct / len(subset):.2%}")
