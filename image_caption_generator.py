import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import numpy as np
from utils import load_mscoco_annotations_val, show_image, load_image, gray_to_RGB
from create_vocabulary import Voc, normalizeAllCaptions
from tokenization import tokenize, pad_sequences
from tqdm import tqdm
from Inception import inception_v3

from torchvision import transforms
# Download dataset is done.

# Load dataset is done.

val_captions, val_image_names = load_mscoco_annotations_val()

# Tokenize captions is done.
print("Loading val_captions_tokens.npy ...")
val_captions_tokens = np.load('val_captions_tokens.npy')
print("Captions are loaded.")
voc = Voc(name="Vocabulary")
voc.load_vocabulary()
#s = str(0)
#print(voc.index2word[s])
# Vocabulary is loaded.
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
##np.save('val_captions_tokens.npy', val_captions_tokens)
## Captions are saved.
#voc.save_vocabulary()
## Vocabulary is saved.

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

print(val_captions_tokens.shape)
print(val_features.shape)

batch_size = 16
hidden_size  = 512
output_size  = 10 # num words
embed_size   = 128
feature_size = 2048

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, embed_size, feature_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.map = nn.Linear(feature_size, hidden_size)
        self.embedding = nn.Embedding(output_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        #input = input.unsqueeze(0)
        print(input.shape)
        output = self.embedding(input)
        print(output.shape)
        output = F.relu(output)
        print(output.shape)
        output, hidden = self.gru(output, hidden)
        print(output.shape)
        output = self.softmax(self.out(output[0]))
        print(output.shape)
        return output, hidden

dec = Decoder(hidden_size=hidden_size, output_size=output_size, embed_size=embed_size, feature_size=feature_size)
dec.cuda()
a,b = dec(input=torch.zeros(batch_size,1).to(torch.int64), hidden = torch.zeros(batch_size,1,hidden_size))
print("a")
print(a.shape)
print("b")
print(b.shape)




