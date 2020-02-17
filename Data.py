#!/usr/bin/env python3

from collections import defaultdict
from collections import Counter
import operator
import json

def prepare_sentences(filename):	# Extracting words and tags from file
	sentences = []
	sentence = []
	tags = []
	with open(filename) as f:
		for line in f:
			line = line.rstrip()
			if line:
				word, tag = line.rstrip().split('\t')
				sentence.append(word)
				tags.append(tag)
			else:
				sentences.append((sentence, tags))
				sentence = []
				tags = []
	return sentences


class Data():
	def __init__(self, *args):
		if len([*args]) == 1:
			self.init_test(*args)
		else:
			self.init_train(*args)

	def init_train(self, train_data, dev_data, numWords, paramfile):
		### Prepare Training Data ###
		self.trainSentences = prepare_sentences(train_data)

		### Prepare Development Data ###
		self.devSentences = prepare_sentences(dev_data)

		### Create Index Dictionaries ###
		word_freq = Counter()
		tagset = set()
		for words, tags in self.trainSentences:
			word_freq.update(words)
			tagset.update(tags)
		ID2tag = sorted(tagset)
		tag2ID = {tag:id for id,tag in enumerate(ID2tag)}
		word_list, rest = zip(*word_freq.most_common(numWords))
		w_index = {}
		n = 1
		for word in word_list:
			w_index[word] = n
			n += 1
		self.word_index = w_index
		self.tag_index = tag2ID
		self.numTags = len(self.tag_index)
		self.store_parameters(paramfile)

	def init_test(self, paramfile):		# Reading Index-Dicts from file
		with open(paramfile) as f:
			self.word_index, self.tag_index = json.load(f)
		self.numTags = len(self.tag_index)
		

	def words2IDs(self, words):
		IDs = []
		for word in words:
			if word in self.word_index:
				IDs.append(self.word_index[word])
			else:
				IDs.append(0)
		return IDs

	def tags2IDs(self, tags):
		IDs = []
		for tag in tags:
			IDs.append(self.tag_index[tag])
		return IDs

	def IDs2tags(self, IDs):
		tags = []
		for ID in IDs:
			for tag, id in self.tag_index.items():
				if id == ID:
					tags.append(tag)
					break
		return tags

	def numTags(self):
		return len(self.tag_index)

	def sentences(self, filename):
		sentence = []
		with open(filename) as f:
			for line in f:
				word = line.rstrip()
				if word:
					sentence.append(word)
				else:
					yield sentence
					sentence = []
			yield sentence

	def store_parameters(self, paramfile):	# Writing Index-Dicts to file
		with open(paramfile, 'w') as f:
			f.write(json.dumps([self.word_index, self.tag_index]))


######## Testfunktion ########

def run_test():
	f = open("test_train", "w")			# Creating Test-Files
	n = 0
	with open("train.tagged") as t:
		for line in t:
			if n < 50000:
				f.write(line)
				n += 1
			else:
				break
	f.close()

	f = open("test_dev", "w")
	n = 0
	with open("dev.tagged") as t:
		for line in t:
			if n < 50000:
				f.write(line)
				n += 1
			else:
				break
	f.close()
	
	error = False

	numWords = 2000
	paramfile = 'parfile1'
	data = Data('test_train', 'test_dev', numWords, paramfile)		# Calling Constructor

	for words, tags in data.trainSentences:         # Check if trainSentences works well
		wordIDs = data.words2IDs(words)
		tagIDs = data.tags2IDs(tags)

		tags_back = data.IDs2tags(tagIDs)           # Check if tag2IDs and IDs2tags work well
		if tags_back != tags:
			print("* Modul-Test: Error in Tag-ID-Functions *")
			error = True

	for words, tags in data.devSentences:           # Check if devSentences works well
		wordIDs = data.words2IDs(words)

	for id in wordIDs:								# Check if wordIDs are type integer
		if type(id) != int:
			print("* Modul-Test: Error: Non-Integer Values in Word_IDs *")
			error = True

	if error == False:
		print("*** Modul-Test: Modul works well ***")

	
	data2 = Data('parfile1')
	IDs = data2.words2IDs(['gut', 'ist', 'morgen'])
	for ID in IDs:
		print(ID)
	IDS = data2.words2IDs(['ist'])
	print(IDS)

######### Main #########

if __name__ == '__main__':
	run_test()
