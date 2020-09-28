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
import time
from torchvision import transforms
# Download dataset is done.

# Load dataset is done.

val_captions, val_image_names = load_mscoco_annotations_val()

# Tokenize captions is done.
print("Loading val_captions_tokens.npy ...")
#val_captions_tokens = np.load('val_captions_tokens.npy')
val_captions_tokens = torch.load('val_captions_tokens.pt')

print("Captions are loaded.")
voc = Voc(name="Vocabulary")
voc.load_vocabulary()
voc_size = len(voc.index2word)
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
#val_captions_tokens_tensor = torch.from_numpy(val_captions_tokens)
#torch.save(val_captions_tokens_tensor,'val_captions_tokens.pt')
#print("tensor saved.")
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
# DONE Design the model and batchify the dataset for training
val_captions_tokens = torch.transpose(val_captions_tokens,0,1)
val_captions_tokens = val_captions_tokens.type(torch.LongTensor)
print(val_captions_tokens.shape)

#val_features = torch.transpose(val_features,0,1).cuda()
#print(val_features.shape)

batch_size = 16
hidden_size  = 2048
output_size  = voc_size # num words
embed_size   = 128
feature_size = 2048
num_layers   = 3

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, embed_size, num_layers, feature_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(output_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers=num_layers)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        #input = input.unsqueeze(0)
        
        output = self.embedding(input)
        
        output = F.relu(output)
        
        output, hidden = self.gru(output, hidden)
        
        output = self.softmax(self.out(output[0]))
        
        return output, hidden

dec = Decoder(hidden_size=hidden_size, output_size=output_size, embed_size=embed_size,num_layers=num_layers, feature_size=feature_size)
dec.cuda()
dec.train()
print(dec)
hidden = torch.zeros(num_layers,batch_size,hidden_size).cuda()
inp = torch.zeros(1,batch_size).to(torch.int64).cuda()
a,b = dec(inp, hidden)




def train_step(tokens_tensor, feature_tensor, decoder, decoder_optimizer, criterion):
  
    decoder_optimizer.zero_grad()
    sequence_length = tokens_tensor.size(0)
    batch_size = tokens_tensor.size(1)
    
  
    loss = 0
    decoder_hidden = torch.zeros(num_layers,batch_size,hidden_size).cuda()
    for i in range(num_layers):
        decoder_hidden[i] = feature_tensor

    #decoder_hidden = feature_tensor.unsqueeze(0)
    for seq in range(sequence_length-1):
        input = tokens_tensor[seq]
        input = input.unsqueeze(0)
        output = tokens_tensor[seq+1]

        decoder_output, decoder_hidden = decoder(input, decoder_hidden)
        loss += criterion(decoder_output, output)

    
  
    loss.backward()
    decoder_optimizer.step()
    return loss.item() / sequence_length

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)




def generate_caption(feature, max_len=30):
    "Does not work..."
    decoder_hidden = torch.zeros(num_layers,batch_size,hidden_size).cuda()
    for i in range(num_layers):
        decoder_hidden[i] = feature

    input = voc.word2index["soc"].type(torch.LongTensor)
    tokens = []
    for i in range(max_len):
        out,decoder_hidden = dec(input,decoder_hidden)
        tokens.append(out)
        input = out

    print(tokens)


def train(decoder, batch_size=64, n_iters=20, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
   
    criterion = nn.NLLLoss()
    batch_index = 0
    data_sample = val_features.shape[0]
   
    for iter in tqdm(range(1, n_iters + 1)):
        dec.eval()
        generate_caption(val_features[0,:])
        dec.train()
        if batch_index + batch_size > data_sample:
            tokens = val_captions_tokens[:,batch_index:].cuda()
            features = val_features[batch_index:,:].cuda()
            batch_index = 0
        else:
            tokens = val_captions_tokens[:,batch_index:batch_index+batch_size].cuda()
            features = val_features[batch_index:batch_index+batch_size,:].cuda()
            batch_index = batch_index+batch_size
       
        loss = train_step(tokens, features, decoder, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)

train(dec)

# TODO complete generate caption and analyze results check if the system is working.

