import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_utils import TrainSet
from module import VQVAE_Audio

def train(model, epoch, trainset, lr):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    trainloader = DataLoader(trainset)
    Recon_Loss = torch.nn.CrossEntropyLoss()
    model.train()

    for _ in enumerate(epoch):
        for i,batch in enumerate(trainloader):
            audio, target, speaker_id = batch
            quantized_loss, decoder_output = model(audio)
            recon_loss = Recon_Loss(decoder_output, target)
            loss = recon_loss + quantized_loss

            optim.zero_grad()
            loss.backward()
            optim.step()

if __name__ == "__main__":
    model = VQVAE_Audio()
