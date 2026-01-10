import random
import json
import sys
from pathlib import Path

# Add project root to Python path to enable imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dataset_generation.dataset import category_wordbank

all_words = sum(category_wordbank.values(), [])  # global pool
dataset = []

while len(dataset) < 5000:
    category = random.choice(list(category_wordbank.keys()))
    in_cat = category_wordbank[category]
    not_in_cat = [w for w in all_words if w not in in_cat]

    # ---- creating BASE example (same as your original logic) ----
    correct_words = random.sample(in_cat, random.randint(1, 10))
    total_words = random.randint(max(2, len(correct_words)), 20)  # ensure enough room
    num_distractors = total_words - len(correct_words)

    distractors = random.sample(not_in_cat, num_distractors)
    base_list = correct_words + distractors
    random.shuffle(base_list)

    base_count = len(correct_words)

    # ---- creating SOURCE list by flipping exactly one position ----
    source_list = base_list.copy()

    # pickinh flip direction (you can force only one if you want)
    flip_nonmatch_to_match = random.random() < 0.5

    if flip_nonmatch_to_match:
        # findinh a position that is currently a distractor (non-match)
        nonmatch_positions = [i for i, w in enumerate(source_list) if w not in in_cat]
        if not nonmatch_positions:
            continue  # skip if impossible (rare)
        flip_index = random.choice(nonmatch_positions)

        # replacing with a matching word not already in the list (if possible)
        candidates = [w for w in in_cat if w not in source_list]
        replacement = random.choice(candidates) if candidates else random.choice(in_cat)

        source_list[flip_index] = replacement
        source_count = base_count + 1
        flip_direction = "nonmatch_to_match"

    else:
        # match -> nonmatch
        match_positions = [i for i, w in enumerate(source_list) if w in in_cat]
        if not match_positions:
            continue  # skip if impossible (rare)
        flip_index = random.choice(match_positions)

        candidates = [w for w in not_in_cat if w not in source_list]
        replacement = random.choice(candidates) if candidates else random.choice(not_in_cat)

        source_list[flip_index] = replacement
        source_count = base_count - 1
        flip_direction = "match_to_nonmatch"

    dataset.append({
        "Type": category,
        "BaseList": base_list,
        "SourceList": source_list,
        "BaseAnswer": base_count,
        "SourceAnswer": source_count,
        "FlipIndex": flip_index,
        "FlipDirection": flip_direction
    })

# Save paired dataset
output_file = "dataset_generation/paired_dataset.json"
with open(output_file, "w") as f:
    json.dump(dataset, f, indent=2)

print(f"Saved {len(dataset)} paired examples to {output_file}")

