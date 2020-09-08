import torch
import torchvision
import torch.nn as nn
from torchvision.models import inception_v3

import numpy as np
from utils import load_mscoco_annotations_val, show_image, load_image, gray_to_RGB
from create_vocabulary import Voc, normalizeAllCaptions
from tokenization import tokenize, pad_sequences
from tqdm import tqdm
from train_utils import inception_v3

from torchvision import transforms
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
#print("Loading val_captions_tokens.npy ...")
#val_captions_tokens = np.load('val_captions_tokens.npy')
#print("Captions are loaded.")
# Captions are loaded.

# DONE load images

# DONE show images

# TODO extract image features.


#encoder = Encoder()
preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
all_images = []
for im in tqdm(val_image_names):
    im = load_image(im)
    im = gray_to_RGB(im)
    #im = preprocess(im)
    all_images.append(im)

all_images = np.array(all_images)
#image = load_image(val_image_names[0])
inception = inception_v3(pretrained=True)
inception.eval()


#input_tensor = preprocess(image)
#input_batch = input_tensor.unsqueeze(0) 
encoder = inception_v3(pretrained=True)
encoder.eval()
o = encoder.forward(all_images)
print(o.shape)

# Extract all features with batches






