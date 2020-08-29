import torch
import numpy as np
from utils import load_mscoco_annotations_val
from create_vocabulary import Voc, normalizeAllCaptions
from tokenization import tokenize, pad_sequences
from tqdm import tqdm

# Download dataset is done.

# Load dataset is done.

val_captions, val_image_names = load_mscoco_annotations_val()

## Tokenize captions is done.
#voc = Voc(name="Vocabulary")

#val_normalized_captions = normalizeAllCaptions(val_captions)

#print()
#print("Creating Vocabulary...")
#for caption in tqdm(val_normalized_captions):

#    voc.addCaption(caption=caption)

#voc.trim(min_count=13)

#tokenized_val_captions = tokenize(voc, val_normalized_captions)

#val_captions_tokens = np.array(pad_sequences(tokenized_val_captions))
#print(val_captions_tokens.shape)
## Captions are padded.
#np.save('val_captions_tokens.npy', val_captions_tokens)
# Captions are saved.
print("Loading val_captions_tokens.npy ...")
val_captions_tokens = np.load('val_captions_tokens.npy')
print("Captions are loaded.")
# Captions are loaded.

# TODO extract image features.






