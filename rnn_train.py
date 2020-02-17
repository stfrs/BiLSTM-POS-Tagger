#!/usr/bin/env python3

import sys
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from Data import Data
from TaggerModel import TaggerModel

parser = argparse.ArgumentParser(description = "LSTM-Trainer")

parser.add_argument('trainfile')
parser.add_argument('devfile')
parser.add_argument('paramfile')
parser.add_argument('--num_epochs', default=20, type=int)
parser.add_argument('--num_words', default=10000, type=int)
parser.add_argument('--emb_size', default=200, type=int)
parser.add_argument('-rnn_size', default=200, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--learning_rate', default=0.5, type=float)

args = parser.parse_args()

print("*** Prepare Network ***")
data = Data(args.trainfile, args.devfile, args.num_words, args.paramfile+'.io')
num_tags = data.numTags
num_words = len(data.word_index)

tagger = TaggerModel(args.num_words+1, data.numTags, args.emb_size, args.rnn_size, args.dropout_rate)
loss_function = nn.CrossEntropyLoss(size_average=False).cuda()
optimizer = optim.SGD(tagger.parameters(), lr = args.learning_rate)
best_accuracy = 0.00

print("*** Start Training ***")
for epoch in range(args.num_epochs):

    ### Network-Training ###
    print("++ Training ++")
    random.shuffle(data.trainSentences)
    tagger = tagger.cuda().train()
    for words, tags in data.trainSentences:
        tagger.zero_grad()
        optimizer.zero_grad()
        tagger.hidden = tagger.init_hidden()
        wordIDs = data.words2IDs(words)
        tagIDs = data.tags2IDs(tags)
        scores = tagger(torch.cuda.LongTensor(wordIDs))
        loss = loss_function(scores, torch.cuda.LongTensor(tagIDs))
        loss.backward()
        optimizer.step()

    ### Network-Evalutation ###
    print("++ Eval ++")
    tagger = tagger.cpu().eval()
    total_tags = 0
    right_classified = 0
    for words, tags in data.devSentences:
        wordIDs = data.words2IDs(words)
        scores = tagger(torch.LongTensor(wordIDs))
        _, pred_tagIDs = scores.max(dim=-1)
        pred_tags = data.IDs2tags(pred_tagIDs)
        for n in range(len(tags)):
            if tags[n] == pred_tags[n]:
                right_classified += 1
        total_tags += len(tags)

    ### Calculating Accuracy ###
    accuracy = right_classified / total_tags
    print("+ Accuracy: ", accuracy, " +", sep='')
    if accuracy > best_accuracy:
        torch.save(tagger, args.paramfile+'.rnn')   # save network if best accuracy
        best_accuracy = accuracy

print("++ Best Accuracy: ", best_accuracy)


