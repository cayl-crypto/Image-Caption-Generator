import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
from utils import *


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


preprocess = transforms.Compose([
    transforms.Resize(160),
    transforms.CenterCrop(160),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

decoder = torch.load("best_decoder.pt")
encoder = torch.load('/home/eva/Desktop/ozkan/Image Caption Generator/autoenc_quant/best_original.pt')

im_path = "/home/eva/Desktop/ozkan/Image Caption Generator/Resized/val2017/000000000139.jpg"
im = load_image(im_path)
im = gray_to_RGB(im)
im = preprocess(im)
encoder_input = torch.zeros(1, 3, 160, 160).cuda()
encoder_input[0] = im
features = encoder.encode(encoder_input)

