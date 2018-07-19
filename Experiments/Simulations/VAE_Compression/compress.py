import os
import json
import torch
import argparse
import numpy as np
import torch.nn as nn
from multiprocessing import cpu_count
from model import SentenceVAE
from utils import to_var, idx2word, interpolate
from collections import OrderedDict, defaultdict
from ptb import PTB
from torch.utils.data import DataLoader

epochs = 10
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
learning_rate = 0.001

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(args.latent_size, 16),
            nn.BatchNorm1d(16),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU())
        self.fc = (nn.Linear(64, args.latent_size))
        
    def forward(self, x):
        # print(x.size())
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.fc(out)
        return out



def main(args):

    def loss_fn(logp, target, length, mean, logv):

        # cut-off unnecessary padding from target, and flatten
        target = target[:, :torch.max(length).item()].contiguous().view(-1)
        logp = logp.view(-1, logp.size(2))
            
        # _, gt = torch.max(target, 1)
        _, pred = torch.max(logp, 1)

        acc = float(torch.sum(target==pred).item())/float(target.size(0))
        # print(pred)
        # Negative Log Likelihood
        NLL_loss = NLL(logp, target)

        # KL Divergence
        KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        KL_weight = 0

        return NLL_loss, KL_loss, KL_weight, acc


    batch_size = args.batch_size

    with open(args.data_dir+'/ptb.vocab.json', 'r') as file:
        vocab = json.load(file)

    w2i, i2w = vocab['w2i'], vocab['i2w']

    model = SentenceVAE(
        vocab_size=len(w2i),
        sos_idx=w2i['<sos>'],
        eos_idx=w2i['<eos>'],
        pad_idx=w2i['<pad>'],
        max_sequence_length=args.max_sequence_length,
        embedding_size=args.embedding_size,
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional
        )



    if not os.path.exists(args.load_checkpoint):
        raise FileNotFoundError(args.load_checkpoint)

    model.load_state_dict(torch.load(args.load_checkpoint))
    print("Model loaded from %s"%(args.load_checkpoint))

    if torch.cuda.is_available():
        model = model.cuda()

    ####Load Data
    splits = ['train', 'valid']

    datasets = OrderedDict()
    for split in splits:
        datasets[split] = PTB(
            data_dir=args.data_dir,
            split=split,
            create_data=args.create_data,
            max_sequence_length=args.max_sequence_length,
            min_occ=args.min_occ
        )

    NLL = torch.nn.NLLLoss(size_average=False, ignore_index=datasets['train'].pad_idx)

    
    model.eval()
    data_loader = DataLoader(
                dataset=datasets['train'],
                batch_size=1,
                shuffle=True,
                num_workers=cpu_count(),
                pin_memory=torch.cuda.is_available()
            )

    accuracy = []
    for j, i in enumerate(data_loader):
        target = i
        # print(target)

        for k, v in target.items():
            if torch.is_tensor(v):
                l = [to_var(v)]*args.batch_size
                target[k] = torch.cat(l, 0)
                # target[k] = to_var(v)

        gen = generator().to(device)

        
        # Loss and optimizer
        optimizer = torch.optim.Adam(gen.parameters(), lr=learning_rate)
        gen.train()

        for epoch in range(epochs):
            inp = to_var(torch.randn([args.batch_size, args.latent_size]))
            z_var = gen(inp)
            # print(target['input'].size())
            logp, mean, logv, z = model.decompress(target['input'], target['length'], z_var)
            # loss calculation
            NLL_loss, KL_loss, KL_weight, acc = loss_fn(logp, target['target'],
                target['length'], mean, logv)

            loss = (NLL_loss)/batch_size

            # backward + optimization
            if split == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1


            # bookkeepeing
            # print((loss.data))
            # print(tracker['ELBO'].size())
            
            # if True:
            #     print("%s Batch %04d/%i, Loss %9.4f, NLL-Loss %9.4f, KL-Loss %9.4f, acc %6.8f, KL-Weight %6.3f"
            #         %(split.upper(), epoch, len(data_loader)-1, loss.item(), NLL_loss.item()/batch_size, KL_loss.item()/batch_size, acc, KL_weight))

        accuracy.append(acc)
        print(j)
        if j==100:
            break

    print("Accuracy is ", np.mean(accuracy))

    # samples, z = model.inference(n=args.num_samples)
    # print('----------SAMPLES----------')
    # print(idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']))

    # z1 = torch.randn([args.latent_size]).numpy()
    # z2 = torch.randn([args.latent_size]).numpy()
    # z = to_var(torch.from_numpy(interpolate(start=z1, end=z2, steps=8)).float())
    # samples, _ = model.inference(z=z)
    # print('-------INTERPOLATION-------')
    # print(idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--load_checkpoint', default='bin/2018-Jul-19-02:13:50/E8.pytorch', type=str)
    parser.add_argument('-n', '--num_samples', type=int, default=10)
    parser.add_argument('--create_data', action='store_true')
    parser.add_argument('--min_occ', type=int, default=1)
    parser.add_argument('-dd', '--data_dir', type=str, default='data')
    parser.add_argument('-bs', '--batch_size', type=int, default=512)
    parser.add_argument('-ms', '--max_sequence_length', type=int, default=40)
    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0.5)
    parser.add_argument('-ls', '--latent_size', type=int, default=16)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')

    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']

    main(args)
