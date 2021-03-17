import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor

torch.set_default_tensor_type('torch.cuda.FloatTensor')
import numpy as np
from utils import load_mscoco_annotations_train, show_image, load_image, gray_to_RGB
from create_vocabulary import Voc, normalizeAllCaptions
from tokenization import tokenize, pad_sequences
from tqdm import tqdm
from Inception import inception_v3
import time
from torchvision import transforms

"""
If there is too much padding the system will memorize padding 
and will not generate a meaningful caption.
"""
# Download dataset is done.

# Load dataset is done.

# val_captions, val_image_names = load_mscoco_annotations_val()
# train_captions, train_image_names = load_mscoco_annotations_train()

# Tokenize captions is done.
# print("Loading val_captions_tokens.pt ...")
##val_captions_tokens = np.load('val_captions_tokens.npy')
train_captions_tokens = torch.load('train_captions_tokens.pt')

# print("Captions are loaded.")
voc = Voc(name="Vocabulary")
voc.load_vocabulary()
voc_size = len(voc.index2word)

##print(voc.index2word[s])
## Vocabulary is loaded.
# train_normalized_captions = normalizeAllCaptions(train_captions)
#
# print()
# print("Creating Vocabulary...")
# for caption in tqdm(train_normalized_captions):
#     voc.addCaption(caption=caption)
#
# voc.trim(min_count=77)
#
# tokenized_train_captions = tokenize(voc, train_normalized_captions)
# voc_size = len(voc.index2word)
# # Before go any further extract captions that have 16 or more tokens.
# core_img_path = []
# core_img_capt_tokens = []
# for img_path, img_cap in zip(train_image_names, tokenized_train_captions):
#
#     if 12 >= len(img_cap) >= 6:
#         core_img_path.append(img_path)
#         core_img_capt_tokens.append(img_cap)
#
# print("LEN of core im path")
# print(len(core_img_path))
# print(len(core_img_capt_tokens))
#
# # prints tokenized captions lengths for best padding the 15 will be fine.
# len_captions = np.zeros((70,), dtype=int)
# for caption in tqdm(tokenized_train_captions):
#     len_captions[len(caption)] += 1
#
# print(len_captions)
#
# # pad captions
# print("shape of train_captions_tokens")
# train_captions_tokens = np.array(pad_sequences(core_img_capt_tokens))
# print(train_captions_tokens.shape)
# print(train_captions_tokens.shape)
# # Captions are padded.
#
# train_captions_tokens = torch.from_numpy(train_captions_tokens)
# torch.save(train_captions_tokens, 'train_captions_tokens.pt')
# print("tensor saved.")
# train_image_names = core_img_path
# # Captions are saved.
# voc.save_vocabulary()
## Vocabulary is saved.

# Captions are loaded.

# DONE load images

# DONE show images

# DONE extract image features.
encoder = inception_v3(pretrained=True)
encoder.eval()
encoder.cuda()
preprocess = transforms.Compose([
    transforms.Resize(80),
    transforms.CenterCrop(80),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# feature_size = 2048

# all_image_features = torch.zeros(len(train_image_names), feature_size)
# feature_extraction_batch_size = 100
# image_batch = torch.zeros(feature_extraction_batch_size, 3, 80, 80)
# start_ids_of_batch = 0
# end_ids_of_batch = start_ids_of_batch + feature_extraction_batch_size


# def batch_extractor(model, batch):
#     with torch.no_grad():
#         features = model.forward(batch)
#     return features
#
#
# # Extract all features with batches
# for ids, im_path in tqdm(enumerate(train_image_names)):
#     # completes in about 30 mins.
#     im = load_image(im_path)
#     im = gray_to_RGB(im)
#     im = preprocess(im)
#     batch_index = ids % feature_extraction_batch_size
#     image_batch[batch_index] = im
#
#     if ids == len(train_image_names) - 1:
#         batch_size = ids % 100
#         image_batch_features = batch_extractor(encoder, image_batch[:batch_size + 1])
#         all_image_features[start_ids_of_batch:ids + 1] = image_batch_features
#         break
#
#     if batch_index == feature_extraction_batch_size - 1:
#         image_batch_features = batch_extractor(encoder, image_batch)
#         all_image_features[start_ids_of_batch:end_ids_of_batch] = image_batch_features
#         start_ids_of_batch = end_ids_of_batch
#         end_ids_of_batch = start_ids_of_batch + feature_extraction_batch_size
#
# train_features = all_image_features.to('cpu')
# torch.save(train_features, 'train_image_features.pt')
# val_features=all_image_features

print("Loading val_image_features.pt ...")
train_features = torch.load("train_image_features.pt")
print("Features are loaded.")
# DONE Design the model and batchify the dataset for training
train_captions_tokens = torch.transpose(train_captions_tokens, 0, 1)
train_captions_tokens = train_captions_tokens.type(torch.LongTensor)
print(train_captions_tokens.shape)

# val_features = torch.transpose(val_features,0,1).cuda()
# print(val_features.shape)
batch_size = 512
hidden_size = 2048
output_size = voc_size  # num words
embed_size = 128
feature_size = 2048
num_layers = 1


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, embed_size, num_layers, feature_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers=num_layers)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # input = input.unsqueeze(0)

        output = self.embedding(input)

        output = F.relu(output)

        output, hidden = self.gru(output, hidden)

        output = self.softmax(self.out(output[0]))

        return output, hidden


dec = Decoder(hidden_size=hidden_size, output_size=output_size, embed_size=embed_size, num_layers=num_layers,
              feature_size=feature_size)
dec.cuda()
dec.train()


# print(dec)
# hidden = torch.zeros(num_layers,batch_size,hidden_size).cuda()
# inp = torch.zeros(1,batch_size).to(torch.int64).cuda()
# a,b = dec(inp, hidden)


def train_step(tokens_tensor, feature_tensor, decoder, decoder_optimizer, criterion):
    decoder_optimizer.zero_grad()
    sequence_length = tokens_tensor.size(0)
    batch_size = tokens_tensor.size(1)

    loss = 0
    decoder_hidden = torch.zeros(num_layers, batch_size, hidden_size).cuda()
    for i in range(num_layers):
        decoder_hidden[i] = feature_tensor

    # decoder_hidden = feature_tensor.unsqueeze(0)
    for seq in range(sequence_length - 1):
        input = tokens_tensor[seq]
        input = input.unsqueeze(0)
        output = tokens_tensor[seq + 1]

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


def generate_caption(deco, feature, max_len=30):
    "Does not work..."
    decoder_hidden = torch.zeros(num_layers, 1, hidden_size).cuda()
    for i in range(num_layers):
        decoder_hidden[i] = feature

    # input = torch.tensor(voc.word2index["soc"]).type(torch.LongTensor).cuda()
    input = torch.ones(1, 1).type(torch.LongTensor).cuda()
    caption = ""
    for i in range(max_len):
        out, decoder_hidden = deco(input, decoder_hidden)
        out = out.argmax(dim=1)

        caption += voc.index2word[str(int(out))] + " "

        input = out.unsqueeze(0)

    print(caption)


def train(decoder, batch_size=128, n_iters=20, print_every=1000, learning_rate=0.01):
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    compare_loss = 99999999
    criterion = nn.NLLLoss()
    batch_index = 0
    data_sample = train_features.shape[0]

    for iter in tqdm(range(1, n_iters + 1)):

        if batch_index + batch_size > data_sample:
            tokens = train_captions_tokens[:, batch_index:].cuda()
            features = train_features[batch_index:, :].cuda()
            batch_index = 0
        else:
            tokens = train_captions_tokens[:, batch_index:batch_index + batch_size].cuda()
            features = train_features[batch_index:batch_index + batch_size, :].cuda()
            batch_index = batch_index + batch_size

        loss = train_step(tokens, features, decoder, decoder_optimizer, criterion)
        print_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(print_loss_avg)
            print()
            decoder.eval()
            if print_loss_avg < compare_loss:
                torch.save(decoder, "best_decoder.pt")
                compare_loss = print_loss_avg
            generate_caption(deco=decoder, feature=train_features[0, :])
            decoder.train()


train(decoder=dec, n_iters=2000, print_every=500, learning_rate=0.01)
# Model class must be defined somewhere
# model = torch.load("best_decoder.pt")

# TODO complete generate caption and analyze results check if the system is working.
