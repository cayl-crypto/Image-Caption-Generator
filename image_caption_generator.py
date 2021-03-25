from pathlib import Path

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
from autoencoder import ConvAutoencoder

torch.set_default_tensor_type('torch.cuda.FloatTensor')
import numpy as np
from utils import *
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
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
'''
dataset_paths = ["Resized", "quant_learning/2", "quant_learning/4", "quant_learning/8", "quant_learning/16"]
project_root = str(Path().absolute().parent)
# Load dataset is done.
annotation_root = project_root + "/Datasets/annotations/"
val_ann_path = annotation_root + "captions_val2017.json"
train_ann_path = annotation_root + "captions_train2017.json"

resized_dataset_folder = project_root + "/" + dataset_paths[0]
two_colored_dataset_folder = project_root + "/" + dataset_paths[1]
four_colored_dataset_folder = project_root + "/" + dataset_paths[2]
eight_colored_dataset_folder = project_root + "/" + dataset_paths[3]
sixteen_colored_dataset_folder = project_root + "/" + dataset_paths[4]

resized_val_captions, resized_val_image_names = load_mscoco_annotations_from_path(dataset_folder=resized_dataset_folder,
                                                                                  ann_path=val_ann_path,
                                                                                  image_path="val2017")
resized_train_captions, resized_train_image_names = load_mscoco_annotations_from_path(
    dataset_folder=resized_dataset_folder, ann_path=train_ann_path, image_path="train2017")

two_val_captions, two_val_image_names = load_mscoco_annotations_from_path(dataset_folder=two_colored_dataset_folder,
                                                                          ann_path=val_ann_path, image_path="val2017")
two_train_captions, two_train_image_names = load_mscoco_annotations_from_path(dataset_folder=two_colored_dataset_folder,
                                                                              ann_path=train_ann_path,
                                                                              image_path="train2017")

four_val_captions, four_val_image_names = load_mscoco_annotations_from_path(dataset_folder=four_colored_dataset_folder,
                                                                            ann_path=val_ann_path, image_path="val2017")
four_train_captions, four_train_image_names = load_mscoco_annotations_from_path(
    dataset_folder=four_colored_dataset_folder, ann_path=train_ann_path, image_path="train2017")

eight_val_captions, eight_val_image_names = load_mscoco_annotations_from_path(
    dataset_folder=eight_colored_dataset_folder, ann_path=val_ann_path, image_path="val2017")
eight_train_captions, eight_train_image_names = load_mscoco_annotations_from_path(
    dataset_folder=eight_colored_dataset_folder, ann_path=train_ann_path, image_path="train2017")

sixteen_val_captions, sixteen_val_image_names = load_mscoco_annotations_from_path(
    dataset_folder=sixteen_colored_dataset_folder, ann_path=val_ann_path, image_path="val2017")
sixteen_train_captions, sixteen_train_image_names = load_mscoco_annotations_from_path(
    dataset_folder=sixteen_colored_dataset_folder, ann_path=train_ann_path, image_path="train2017")
    '''
# val_captions, val_image_names = load_mscoco_annotations_val()
# train_captions, train_image_names = load_mscoco_annotations_train()

# Tokenize captions is done.
# print("Loading val_captions_tokens.pt ...")
# val_captions_tokens = np.load('val_captions_tokens.npy')
# train_captions_tokens = torch.load('train_captions_tokens.pt')

# print("Captions are loaded.")
voc = Voc(name="Vocabulary")
voc.load_vocabulary()
voc_size = len(voc.index2word)


# print(voc.index2word[s])
#  Vocabulary is loaded.
# val_normalized_captions = normalizeAllCaptions(resized_val_captions)
# train_normalized_captions = normalizeAllCaptions(resized_train_captions)
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


def get_core_dataset(image_names, tokenized_captions):
    # Before go any further extract captions that have 16 or more tokens.
    core_img_path = []
    core_img_capt_tokens = []
    for img_path, img_cap in zip(image_names, tokenized_captions):

        if 17 >= len(img_cap) >= 8:
            core_img_path.append(img_path)
            core_img_capt_tokens.append(img_cap)

    print("LEN of core im path")
    print(len(core_img_path))
    print(len(core_img_capt_tokens))
    return core_img_path, core_img_capt_tokens


# resized_core_train_img_path, resized_core_train_img_capt_tokens = get_core_dataset(resized_train_image_names,
#                                                                                    tokenized_train_captions)
#
# two_core_train_img_path, two_core_train_img_capt_tokens = get_core_dataset(two_train_image_names,
#                                                                            tokenized_train_captions)
#
# four_core_train_img_path, four_core_train_img_capt_tokens = get_core_dataset(four_train_image_names,
#                                                                              tokenized_train_captions)
#
# eight_core_train_img_path, eight_core_train_img_capt_tokens = get_core_dataset(eight_train_image_names,
#                                                                                tokenized_train_captions)
# sixteen_core_train_img_path, sixteen_core_train_img_capt_tokens = get_core_dataset(sixteen_train_image_names,
#                                                                                    tokenized_train_captions)

# prints tokenized captions lengths for best padding the 15 will be fine.
# len_captions = np.zeros((70,), dtype=int)
# for caption in tqdm(tokenized_train_captions):
#     len_captions[len(caption)] += 1
#
# print(len_captions)
#
# pad captions
# print("shape of train_captions_tokens")
# train_captions_tokens = np.array(pad_sequences(resized_core_train_img_capt_tokens))
# print(train_captions_tokens.shape)
# print(train_captions_tokens.shape)
# Captions are padded.
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
# model = inception_v3(pretrained=True)
# model.eval()
# model.cuda()
preprocess = transforms.Compose([
    transforms.Resize(160),
    transforms.CenterCrop(160),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
feature_size = 256


def batch_extractor(encoder, batch):
    with torch.no_grad():
        features = encoder.encode(batch)
    return features


def extract_features(encoder, train_image_names, path_name: str):
    all_image_features = torch.zeros(len(train_image_names), feature_size)
    feature_extraction_batch_size = 100
    image_batch = torch.zeros(feature_extraction_batch_size, 3, 160, 160)
    start_ids_of_batch = 0
    end_ids_of_batch = start_ids_of_batch + feature_extraction_batch_size
    # Extract all features with batches
    for ids, im_path in tqdm(enumerate(train_image_names)):
        # completes in about 30 mins.
        im = load_image(im_path)
        im = gray_to_RGB(im)
        im = preprocess(im)
        batch_index = ids % feature_extraction_batch_size
        image_batch[batch_index] = im

        if ids == len(train_image_names) - 1:
            batch_size = ids % 100
            image_batch_features = batch_extractor(encoder, image_batch[:batch_size + 1])
            all_image_features[start_ids_of_batch:ids + 1] = image_batch_features
            break

        if batch_index == feature_extraction_batch_size - 1:
            image_batch_features = batch_extractor(encoder, image_batch)
            all_image_features[start_ids_of_batch:end_ids_of_batch] = image_batch_features
            start_ids_of_batch = end_ids_of_batch
            end_ids_of_batch = start_ids_of_batch + feature_extraction_batch_size

    train_features = all_image_features.to('cpu')
    print(train_features.shape)
    torch.save(train_features, path_name + '.pt')
    # val_features=all_image_features


# encoder_resized = torch.load(project_root + "/autoenc_quant/best_original.pt", map_location=device)
# encoder_2 = torch.load(project_root + "/autoenc_quant/best_2.pt", map_location=device)
# encoder_4 = torch.load(project_root + "/autoenc_quant/best_4.pt", map_location=device)
# encoder_8 = torch.load(project_root + "/autoenc_quant/best_8.pt", map_location=device)
# encoder_16 = torch.load(project_root + "/autoenc_quant/best_16.pt", map_location=device)

# extract_features(encoder=encoder_resized, train_image_names=resized_core_train_img_path, path_name="original_image_train_features")
# extract_features(encoder=encoder_resized, train_image_names=resized_val_image_names, path_name="original_image_val_features")
#
# extract_features(encoder=encoder_2, train_image_names=two_core_train_img_path, path_name="two_image_train_features")
# extract_features(encoder=encoder_2, train_image_names=two_val_image_names, path_name="two_image_val_features")
#
# extract_features(encoder=encoder_4, train_image_names=four_core_train_img_path, path_name="four_image_train_features")
# extract_features(encoder=encoder_4, train_image_names=four_val_image_names, path_name="four_image_val_features")
#
# extract_features(encoder=encoder_8, train_image_names=eight_core_train_img_path, path_name="eight_image_train_features")
# extract_features(encoder=encoder_8, train_image_names=eight_val_image_names, path_name="eight_image_val_features")

# extract_features(encoder=encoder_16, train_image_names=sixteen_core_train_img_path, path_name="sixteen_image_train_features")
# extract_features(encoder=encoder_16, train_image_names=sixteen_val_image_names, path_name="sixteen_image_val_features")

# DONE Design the model and batchify the dataset for training


# val_features = torch.transpose(val_features,0,1).cuda()
# print(val_features.shape)
batch_size = 512
hidden_size = 256
output_size = voc_size  # num words
embed_size = 32
# feature_size = 256
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


# print(dec)
# hidden = torch.zeros(num_layers,batch_size,hidden_size).cuda()
# inp = torch.zeros(1,batch_size).to(torch.int64).cuda()
# a,b = dec(inp, hidden)


def train_step(tokens_tensor, feature_tensor, decoder, decoder_optimizer, criterion):
    decoder_optimizer.zero_grad()
    sequence_length = tokens_tensor.size(0)
    batch_size = tokens_tensor.size(1)

    loss = 0.0
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


# train_captions_tokens = torch.load("train_captions_tokens.pt")
# train_captions_tokens = torch.transpose(train_captions_tokens, 0, 1)
# train_captions_tokens = train_captions_tokens.type(torch.LongTensor)
# print(train_captions_tokens.shape)
# print("Loading train_image_features.pt ...")
# train_features = torch.load("original_image_train_features.pt")
# print("Features are loaded.")
#
# dec = Decoder(hidden_size=hidden_size, output_size=output_size, embed_size=embed_size, num_layers=num_layers,
#               feature_size=feature_size)
# dec.cuda()
# dec.train()
# train(decoder=dec, batch_size=2048, n_iters=200000, print_every=250, learning_rate=0.01)
# Model class must be defined somewhere
# model = torch.load("best_decoder.pt")


decoder = torch.load("best_decoder.pt")
encoder = torch.load('/home/eva/Desktop/ozkan/Image Caption Generator/autoenc_quant/best_original.pt')

im_path = "/home/eva/Desktop/ozkan/Image Caption Generator/Resized/train2017/000000000294.jpg"
im = load_image(im_path)
im = gray_to_RGB(im)
im = preprocess(im)
encoder_input = torch.zeros(1, 3, 160, 160).cuda()
encoder_input[0] = im
features = encoder.encode(encoder_input)

generate_caption(decoder, features)
show_image(load_image(im_path))
# TODO complete generate caption and analyze results check if the system is working.
