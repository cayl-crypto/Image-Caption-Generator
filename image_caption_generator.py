import torch
import torchvision
import torch.nn as nn
from torchvision.models import inception_v3
torch.set_default_tensor_type('torch.cuda.FloatTensor')
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
print("Loading val_captions_tokens.npy ...")
val_captions_tokens = np.load('val_captions_tokens.npy')
print("Captions are loaded.")
# Captions are loaded.

# DONE load images

# DONE show images

# DONE extract image features.
encoder = inception_v3(pretrained=True)
encoder.eval()
encoder.cuda()
preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
feature_size = 2048


#all_image_features = torch.zeros(len(val_image_names), feature_size)
#feature_extraction_batch_size = 100
#image_batch = torch.zeros(feature_extraction_batch_size, 3, 299, 299)
#start_ids_of_batch = 0
#end_ids_of_batch = start_ids_of_batch + feature_extraction_batch_size

#def batch_extractor(model, batch):
#    with torch.no_grad():
#        features = model.forward(batch)
#    return features

# Extract all features with batches
#for ids, im_path in tqdm(enumerate(val_image_names)):
#    # completes in about 30 mins.
#    im = load_image(im_path)
#    im = gray_to_RGB(im)
#    im = preprocess(im)
#    batch_index = ids % feature_extraction_batch_size
#    image_batch[batch_index] = im

#    if ids == len(val_image_names) - 1:
#        batch_size = ids % 100     
#        image_batch_features = batch_extractor(encoder, image_batch[:batch_size+1])
#        all_image_features[start_ids_of_batch:ids+1] = image_batch_features
#        break

#    if batch_index == feature_extraction_batch_size - 1:   
#        image_batch_features = batch_extractor(encoder, image_batch)
#        all_image_features[start_ids_of_batch:end_ids_of_batch] = image_batch_features
#        start_ids_of_batch = end_ids_of_batch
#        end_ids_of_batch = start_ids_of_batch + feature_extraction_batch_size
        
#torch.save(all_image_features, 'val_image_features.pt')
print("Loading val_image_features.pt ...")
val_features = torch.load("val_image_features.pt")
print("Features are loaded.")
# TODO Design the model and batchify the dataset for training







