import torch
from Inception import inception_v3

import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn


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


encoder = inception_v3(pretrained=True)
decoder = torch.load("best_decoder.pt")


class Generator(nn.Module):

    def __init__(self, e, d):
        super().__init__()
        self.encoder = e
        self.decoder = d

    def forward(self, inputImage):
        features = self.encoder.forward(inputImage)

        decoder_hidden = torch.zeros(1, 1, 2048)
        for i in range(1):
            decoder_hidden[i] = features

        inputtoken = torch.ones(1, 1).type(torch.LongTensor)
        output = torch.zeros(30)
        for i in range(30):
            out, decoder_hidden = self.decoder(inputtoken, decoder_hidden)
            out = out.argmax(dim=1)
            output[i] = out
            inputtoken = out.unsqueeze(0)
        return output


encoder.eval()
decoder.eval()
decoder.cpu()

generator = Generator(e=encoder, d=decoder)
encoder_input = torch.zeros(1, 3, 80, 80)
generator_input = torch.zeros(1, 3, 80, 80)

decoder_input1 = torch.tensor([[0]])
decoder_input2 = torch.zeros(1, 1, 2048)

# dynamic quantization can be applied to the decoder for its nn.Linear parameters
quantized_decoder = torch.quantization.quantize_dynamic(decoder, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)

traced_encoder = torch.jit.trace(encoder, encoder_input)
traced_generator = torch.jit.trace(generator, generator_input)
traced_decoder = torch.jit.trace(quantized_decoder, (decoder_input1, decoder_input2))

from torch.utils.mobile_optimizer import optimize_for_mobile

# traced_encoder_optimized = optimize_for_mobile(traced_encoder)
# traced_encoder_optimized.save("optimized_encoder_150k.pth")
traced_encoder.save("encoder.pth")
traced_generator.save("generator.pth")

# traced_decoder_optimized = optimize_for_mobile(traced_decoder)
# traced_decoder_optimized.save("optimized_decoder_150k.pth")
traced_decoder.save("decoder.pth")
