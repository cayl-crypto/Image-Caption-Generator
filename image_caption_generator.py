import torch
import numpy as np
from utils import load_mscoco_annotations_val
from create_vocabulary import Voc, normalizeAllCaptions
from tokenization import tokenize, get_maximum_length_of_captions
from tqdm import tqdm

# Download dataset is done.

# Load dataset is done.

val_captions, val_image_names = load_mscoco_annotations_val()

# Tokenize captions is done.
voc = Voc(name="Vocabulary")

val_normalized_captions = normalizeAllCaptions(val_captions)

print()
print("Creating Vocabulary...")
for caption in tqdm(val_normalized_captions):

    voc.addCaption(caption=caption)

voc.trim(min_count=13)

tokenized_val_captions = tokenize(voc, val_normalized_captions)

max_len = get_maximum_length_of_captions(tokenized_val_captions)

# TODO pad tokenized captions.






