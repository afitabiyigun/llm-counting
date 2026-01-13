# Category Counting via Causal Mediation Analysis

This project investigates whether LLMs internally represent a **running count** when solving a zero-shot category counting task. Using synthetic data, model benchmarking, and causal mediation analysis with activation patching, we identify which hidden states causally contribute to correct counting behavior.

---

## Task Description

We study the following zero-shot counting task:
  Count the number of words in the following list that match the given type, and put the numerical answer in parentheses.
  Type: fruit
  List: [dog apple cherry bus cat grape bowl]
  Answer: (


A sufficiently capable language model should complete the answer as `(3)` without any chain-of-thought or intermediate reasoning tokens.

---


A sufficiently capable language model should complete the answer as `(3)` without any chain-of-thought, reasoning tokens, or intermediate explanations.

---

## Project Goals

1. **Dataset generation**  
   Generate a large, controlled synthetic dataset of category counting prompts.
2. **Zero-shot benchmarking**  
   Evaluate open-weight language models on the task without prompting tricks or reasoning scaffolds.
3. **Causal mediation analysis**  
   Identify whether specific hidden layers encode a running count of matching items during list processing.

---

## Dataset Generation

The dataset is fully synthetic and designed to minimize lexical shortcuts and memorization effects.

### Categories

Prompts are generated using a fixed set of semantic categories, including:

- **Fruits** (e.g., apple, banana, grape)
- **Animals** (e.g., dog, cat, horse)
- **Vehicles** (e.g., car, bus, train)
- **Colors** (e.g., red, blue, green)
- **Household objects** (e.g., chair, bowl, table)
- **Clothing items** (e.g., shirt, shoe, jacket)

Each category has a curated vocabulary of 20–50 tokens.

### Prompt Structure

- List length: **5–10 tokens**
- Matching items per list: **0–5**
- Token order: **randomized**
- Categories are balanced across the dataset

### Dataset Size

- **Total prompts generated:** ~5,000
- **Train / evaluation split:** used only for benchmarking consistency (no training performed)

Each example includes:
- The full prompt text
- The target category
- The token list
- The ground-truth numerical count

---

## Benchmarking

Open-weight transformer models are evaluated on **exact-match accuracy** of the final numerical answer.

Evaluation conditions:
- Zero-shot
- Greedy decoding
- No chain-of-thought or intermediate reasoning tokens

Results show strong scaling behavior: smaller models fail consistently, while larger models achieve high accuracy, suggesting the emergence of implicit counting mechanisms rather than surface-level heuristics.

---

## Causal Mediation Analysis

To probe *how* models solve the task internally, we apply causal mediation analysis using **activation patching** with NNsight.

### Method Overview

1. Run the model on identical prompts with different internal trajectories (successful vs. unsuccessful counting).
2. Patch hidden activations from successful runs into failed runs at specific layers and token positions.
3. Measure changes in the probability of generating the correct numerical answer.

This allows estimation of **layer-wise and token-wise causal effects** on counting behavior.

---

## Results

Causal effect heatmaps (see `cma/causal_effects/`) reveal:

- **Strong mediation effects concentrated in mid-to-upper transformer layers** (approximately la
