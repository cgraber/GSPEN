import arff
import torch
import sys

if len(sys.argv) != 5:
    print("USAGE: python convert2tensor.py [bibtex, bookmarks] <input file> <output features> <output_labels>")
    sys.exit(1)

if sys.argv[1] == 'bibtex':
    split_ind = 159
elif sys.argv[1] == 'bookmarks':
    split_ind = 208
else:
    raise ValueError("Mode must be one of [bibtex, bookmarks]")
print("CONVERTING FILE: ",sys.argv[2])
with open(sys.argv[2], "r") as fin:
    info = arff.load(fin)

features = []
labels = []
for datum in info['data']:
    datum = torch.IntTensor([int(val) for val in datum])
    features.append(datum[:-split_ind])
    labels.append(datum[-split_ind:])

torch.save(torch.stack(features), sys.argv[3])
torch.save(torch.stack(labels), sys.argv[4])