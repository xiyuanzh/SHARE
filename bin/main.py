import argparse
import datetime
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from semantichar.seq2seq import Encoder, Decoder
from semantichar.exp import trainer, evaluate
from semantichar.dataset import prepare_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='dataset to use')
parser.add_argument('--data_path', default='./', help='path to store the data')

parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--early_stopping', type=int, default=50, help='early stopping patient steps')
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate, default=0.0002')
parser.add_argument('--prob', type=float, default=0.4, help='probabilty for label augmentation')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--visualize', action='store_true', help='visualize training loss curve on wandb')
parser.add_argument('--run_tag', default='', help='tags for the current run')
parser.add_argument('--model_path', default='./model/', help='path to save the model')
opt = parser.parse_args()

print(opt)

if opt.manualSeed is None:
    opt.manualSeed = 2023
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
np.random.seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("You have a cuda device, so you might want to run with --cuda as option")
device = torch.device("cuda:0" if opt.cuda else "cpu")

data_root = opt.data_path + '/dataset/' + opt.dataset
config_file = opt.data_path + '/configs/' + opt.dataset + '.json'
with open(config_file, 'r') as config_file:
    data = json.load(config_file)
    label_dictionary = {int(k): v for k, v in data['label_dictionary'].items()}

tr_data = np.load(data_root + '/X_train.npy')
tr_label = np.load(data_root + '/y_train.npy')

test_data = np.load(data_root + '/X_test.npy')
test_label = np.load(data_root + '/y_test.npy')

seq_len, dim, class_num, vocab_size, break_step, word_list, pred_dict, seqs, \
    tr_data, test_data, \
    tr_label, test_label, \
    tr_text, test_text, \
    len_text, test_len_text = \
    prepare_dataset(tr_data, tr_label, test_data, test_label, label_dictionary)

enc = Encoder(d_input=dim, d_model=128, d_output=128, seq_len=seq_len).to(device)
dec = Decoder(embed_dim=1024, decoder_dim=128, vocab=word_list, encoder_dim=128, device=device).to(device)
params = list(enc.parameters()) + list(dec.parameters())
optimizer = optim.Adam(params, lr=1e-4)

cross_entropy = nn.CrossEntropyLoss().to(device)

if opt.visualize:
    import wandb
    print('log')
    date = datetime.datetime.now().strftime("%d-%m-%y_%H:%M")
    wandb.init(
        project='SemanticHAR',
        config=vars(opt),
        name=f"{opt.run_tag}_{opt.dataset}_seed_{opt.manualSeed}_{date}" 
        )

best_acc = 0

for epoch in range(opt.epochs):

    loss = trainer(
        opt, # configs
        enc, # encoder
        dec, # decoder
        cross_entropy, # loss
        optimizer, # optimizer
        tr_data, # training input
        tr_label, # training labels
        tr_text, # training label text sequence
        len_text, # training label text sequence length
        break_step, # max training label text sequence length
        vocab_size, # vocabulary size
        device, # device
    )

    print("epoch: %d total loss: %.4f" % (epoch + 1, loss))
    if opt.visualize:
        wandb.log({"Loss": loss})

    torch.save(enc.state_dict(), opt.model_path + opt.run_tag + 'enc')
    torch.save(dec.state_dict(), opt.model_path + opt.run_tag + 'dec')

acc, prec, rec, f1 = evaluate(
    opt, # configs
    enc, # encoder
    dec, # decoder
    test_data, # test input
    test_label, # test labels
    test_text, # test label text sequence
    test_len_text, # test label text sequence length
    pred_dict,  # mapping from label token-id sequence to label id
    seqs, # label token-id sequence for all classes
    break_step, # max training label text sequence length
    class_num, # number of classes
    vocab_size, # vocabulary size
    device # device
)

print('Test Acc: %.4f Macro-Prec: %.4f Macro-Rec: %.4f Macro-F1: %.4f' % (acc, prec, rec, f1))
if opt.visualize:
    wandb.log({"Acc": acc, "Macro-Prec": prec, "Macro-Rec": rec, "Macro-F1": f1})