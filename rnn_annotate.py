#!/usr/bin/env python3

import argparse
import torch
from Data import Data
from TaggerModel import TaggerModel

parser = argparse.ArgumentParser(description = "LSTM-Tagger")
parser.add_argument('path_param')
parser.add_argument('filename')
args = parser.parse_args()

data = Data(args.path_param+'.io')          # Reading the symbol mapping tables
model = torch.load(args.path_param+'.rnn')  # Reading the model
model = model.cpu()
for sentence in data.sentences(args.filename):   # Extracting sentences from file
    wordIDs = data.words2IDs(sentence)
    with torch.no_grad():
        scores = model(torch.LongTensor(wordIDs))   # Calculating scores with LSTM
    _, tagIDs = scores.max(dim=-1)
    tags = data.IDs2tags(tagIDs)

    for n in range(len(sentence)):
        print(sentence[n], "[", tags[n], "] ", sep='', end='') # Printing words with predicted tags
    print()
