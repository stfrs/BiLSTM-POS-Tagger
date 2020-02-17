#!/usr/bin/env python3

from Data import Data
import torch
import torchvision
import torch.nn as nn

###### Class TaggerModel ######

class TaggerModel(nn.Module):

    def __init__(self, numWords, numTags, embSize, rnnSize, dropoutRate):
        super(TaggerModel, self).__init__()
        self.rnnSize = rnnSize
        self.embedding = nn.Embedding(numWords, embSize).cuda()
        self.dropout = nn.Dropout(dropoutRate).cuda()
        self.lstm = nn.LSTM(embSize, self.rnnSize, num_layers = 1, bidirectional = True).cuda()
        self.linear = nn.Linear(self.rnnSize*2, numTags).cuda()
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return(torch.zeros(1, 1, self.rnnSize), torch.zeros(1, 1, self.rnnSize))
	
    def forward(self, wordIDs):
        #wordIDs = torch.autograd.Variable(wordIDs)
        embedding = self.embedding(wordIDs)
        embedding_d = self.dropout(embedding)
        hidden, _ = self.lstm(embedding_d.unsqueeze(0))
        hidden_d = self.dropout(hidden.squeeze(0))
        output = self.linear(hidden_d)
        return output


######## Test-Funktion ########

def run_test():
    tagger = TaggerModel(100, 100, 20, 20, 0.5)
    wordIDs = [2,5,20,4,3]
    scores = tagger(torch.cuda.LongTensor(wordIDs))
    print(scores)


####### Main (Test) #######

if __name__ == "__main__":
    run_test()

