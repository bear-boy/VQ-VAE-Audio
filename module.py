import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Encoder(nn.Module):
    def __init__(self, out_channels, in_channels=256, stride=2, kernel_size=4):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        for out_channel in out_channels:
            self.layers.append(nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=out_channel,
                          kernel_size=kernel_size,stride=stride,padding=0),
                nn.Tanh()
            ))
            in_channels = out_channel

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

class Decoder(nn.Module):
    '''WaveNet Decoder'''
    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self, inputs):
        pass

class VectorQuantizer(nn.Module):
    def __init__(self, num_embedding, embedding_dim, beta):
        super(VectorQuantizer,self).__init__()
        self.num_embedding = num_embedding
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings=num_embedding, embedding_dim=embedding_dim)
        self.embedding.weight.data.uniform_(-1.0/self.num_embedding, 1.0/self.num_embedding)
        self.beta = beta

    def forward(self, inputs):
        # inputs shape: B x C x T
        inputs_shape = inputs.shape
        inputs = inputs.permute(0,2,1).contiguous()

        flat_input = inputs.view(-1, inputs.shape[2])           # (BxT) x C, equals D x K in paper

        distance = torch.sum(flat_input**2, dim=1, keepdim=True) +\
                   torch.sum(self.embedding.weight.data, dim=1) - \
                   2 * torch.matmul(flat_input, self.embedding.weight.t())          # torch.sum()按指定维度挤压

        encoding_ind = torch.argmin(distance, dim=1).unsqueeze(1)       # (BxT) x 1
        encodings = torch.zeros((encoding_ind.shape[0], self.num_embedding)).to(device)
        encodings.scatter_(1, encodings, 1)

        quantizer = torch.matmul(encodings, self.embedding.weight.data).view(inputs_shape)

        loss = F.mse_loss(inputs.detach(), quantizer) + self.beta * F.mse_loss(inputs, quantizer.detach())
        quantized = inputs + (quantizer.detach() - inputs)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized.permute(0,2,1).contiguous(), perplexity

class VQVAE_Audio(nn.Module):
    def __init__(self):
        super(VQVAE_Audio, self).__init__()
        self.encoder = Encoder()
        self.vqvae = VectorQuantizer()
        self.decoder = Decoder()

    def forward(self, x):
        audio = x
        encoder_output = self.encoder(audio)

        quantized_loss, quantized_embedding, perplexity = self.vqvae(encoder_output)

        decoder_output = self.decoder(quantized_embedding)

        return quantized_loss, decoder_output
