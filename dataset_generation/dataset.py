import random
import json

# hardcoding 50 words for each selected category
category_wordbank = {
    "fruit": ["apple", "banana", "cherry", "grape", "pear", "peach", "plum", "orange", "kiwi", "mango", "apricot", 
        "nectarine", "papaya", "pineapple", "coconut", "blueberry", "raspberry", "strawberry", "blackberry", "pomegranate", 
        "fig", "guava", "lychee", "cranberry", "cantaloupe", "watermelon", "honeydew", "passionfruit", "persimmon", "quince", 
        "kumquat", "tangerine", "clementine", "mulberry", "olive", "currant", "boysenberry", "gooseberry", "date", "loganberry", 
        "starfruit", "ugli", "soursop", "jackfruit", "rambutan", "durian", "tamarind", "longan", "sapodilla"], 
    "animal": ["dog", "cat", "lion", "tiger", "elephant", "zebra", "monkey", "giraffe", "bear", "wolf", "fox", "rabbit", "deer", 
        "cow", "horse", "goat", "sheep", "camel", "donkey", "kangaroo", "panda", "leopard", "cheetah", "buffalo", "moose", "rat", 
        "mouse", "squirrel", "bat", "hedgehog", "otter", "seal", "whale", "dolphin", "shark", "crocodile", "alligator", "penguin", 
        "flamingo", "peacock", "eagle", "hawk", "falcon", "parrot", "owl", "chicken", "turkey", "duck", "goose"],
    "sport": ["soccer", "basketball", "tennis", "baseball", "hockey", "volleyball", "golf", "rugby", "cricket", "boxing", "wrestling", 
        "badminton", "cycling", "archery", "fencing", "skiing", "snowboarding", "skating", "rowing", "sailing", "canoeing", "kayaking", 
        "surfing", "diving", "swimming", "triathlon", "marathon", "judo", "karate", "taekwondo", "gymnastics", "billiards", "snooker", 
        "darts", "ping-pong", "racquetball", "polo", "handball", "lacrosse", "curling", "motocross", "nascar", "formula1", "bmx", 
        "speedskating", "climbing", "parkour", "horseback", "bowling", "paintball"], 
    "vehicle": ["car", "bus", "truck", "van", "scooter", "motorcycle", "bicycle", "train", "subway", "tram", "airplane", "jet", 
        "helicopter", "boat", "ship", "canoe", "kayak", "ferry", "yacht", "sailboat", "skateboard", "rollerblades", "segway", 
        "taxi", "rickshaw", "trolley", "minivan", "pickup", "camper", "caravan", "hovercraft", "submarine", "spaceship", 
        "glider", "gondola", "zeppelin", "moped", "dune buggy", "golf cart", "ATV", "snowmobile", "jet ski", "limousine", 
        "ambulance", "firetruck", "police car", "cruise", "freighter", "container ship", "rocket"], 
    "color": ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "black", "white", "cyan", "magenta", "maroon", 
        "navy", "teal", "lime", "indigo", "violet", "beige", "gray", "silver", "gold", "ivory", "peach", "tan", "lavender", "turquoise", 
        "salmon", "coral", "amber", "mustard", "plum", "mint", "aqua", "charcoal", "cream", "ruby", "sapphire", "emerald", "bronze", 
        "rose", "apricot", "mahogany", "ochre", "periwinkle", "fuchsia", "mauve", "olive", "copper", "khaki"], 
    "instrument": ["piano", "guitar", "drum", "violin", "flute", "trumpet", "saxophone", "harp", "cello", "clarinet", "trombone", "oboe",
        "bassoon", "tuba", "mandolin", "banjo", "ukulele", "harmonica", "xylophone", "accordion", "tambourine", "maracas", "sitar", "tabla", 
        "djembe", "bongo", "lute", "pan flute", "didgeridoo", "organ", "bugle", "zither", "lyre", "kalimba", "ocarina", "viola", "double bass", 
        "castanets", "steel drum", "clavichord", "melodica", "triangle", "conga", "bass drum", "snare drum", "gong", "bell", "glass harmonica", 
        "jaw harp", "hurdy-gurdy"], 
    "occupation": ["doctor", "engineer", "teacher", "lawyer", "nurse", "artist", "chef", "pilot", "writer", "musician", "actor", "director", 
        "plumber", "electrician", "mechanic", "farmer", "dentist", "architect", "accountant", "scientist", "pharmacist", "photographer", 
        "designer", "barber", "stylist", "receptionist", "librarian", "psychologist", "therapist", "analyst", "clerk", "banker", "cashier", 
        "firefighter", "police officer", "soldier", "judge", "journalist", "translator", "interpreter", "coach", "athlete", "dancer", "tailor", 
        "butcher", "baker", "janitor", "zoologist", "biologist", "economist"], 
    "furniture": ["chair", "table", "sofa", "couch", "bed", "desk", "cabinet", "dresser", "wardrobe", "nightstand", "bookshelf", "stool", "bench", 
        "armchair", "ottoman", "loveseat", "recliner", "rocking chair", "hutch", "sideboard", "coffee table", "end table", "vanity", "chest", 
        "trunk", "folding chair", "bar stool", "gaming chair", "bean bag", "futon", "crib", "changing table", "bunk bed", "murphy bed", "sofabed", 
        "tv stand", "entertainment center", "console table", "buffet", "china cabinet", "file cabinet", "workbench", "coat rack", "shoe rack", 
        "wine rack", "bookshelf ladder", "chaise lounge", "massage chair", "nesting table", "hall tree"], "plant": ["rose", "tulip", "daisy", 
        "orchid", "lily", "sunflower", "lavender", "bamboo", "cactus", "fern", "ivy", "mint", "basil", "thyme", "oregano", "cilantro", "parsley",
        "aloe", "chamomile", "poppy", "marigold", "daffodil", "peony", "azalea", "hibiscus", "geranium", "pansy", "begonia", "zinnia", "hydrangea",
        "snapdragon", "freesia", "amaryllis", "gardenia", "anemone", "aster", "bluebell", "camellia", "dahlia", "gladiolus", "heather", "jasmine", 
        "magnolia", "petunia", "rhododendron", "verbena", "wisteria", "yarrow", "lilac", "calendula"] 
    }


all_words = sum(category_wordbank.values(), []) # flatting all words into a global pool
dataset = []

while len(dataset) < 5000:
    category = random.choice(list(category_wordbank.keys()))
    correct_words = random.sample(category_wordbank[category], random.randint(1, 10))
    total_words = random.randint(2, 20)
    num_distractors = max(0, total_words - len(correct_words))
    distractor_pool = [w for w in all_words if w not in category_wordbank[category]]
    distractors = random.sample(distractor_pool, num_distractors)
    
    word_list = correct_words + distractors
    random.shuffle(word_list)

    dataset.append({
        "Type": category,
        "List": word_list,
        "Answer": len(correct_words)
    })

# Save dataset in the data_generation folder
output_file = "dataset_generation/dataset.json"
with open(output_file, "w") as f:
    json.dump(dataset, f, indent=2)
