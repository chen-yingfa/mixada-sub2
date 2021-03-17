import csv
import nltk
from .data import PTBDataset
import benepar
from .augmenters import *

sentence = "(3 (2 It) (4 (4 (2 's) (4 (3 (2 a) (4 (3 lovely) (2 film))) (3 (2 with) (4 (3 (3 lovely) (2 performances)) (2 (2 by) (2 (2 (2 Buy) (2 and)) (2 Accorsi))))))) (2 .)))"

tree = nltk.Tree.fromstring(sentence)

def tree_to_str(tree):
    if isinstance(tree, str):
        return tree
    children = [tree_to_str(c) for c in tree]
    ret = f"({tree.label()} {' '.join(children)})"
    return ret


def sent_to_trees():
    # load examples
    with open('../data/sst/train.tsv') as fin:
        reader = csv.reader(fin, delimiter='\t')
        examples = list(reader)
    
    for i, example in enumerate(examples):
        sent = example[0]
        words = nltk.word_tokenize(sent)
        sent = benepar.InputSentence(words=words)
        tree = benepar.Parser("benepar_en3").parse(sent)
        s = tree_to_str(tree)
        examples[i][0] = s

    # save trees to file
    with open("../data/trees/benepar/train.tsv", 'w') as fout:
        writer = csv.writer(fout, delimiter="\t")
        
        for example in examples:
            writer.writerow(example)


def augment():
    filename = "train.tsv"
    sst_dataset = PTBDataset(
        f'../data/trees/benepar/{filename}', use_spans=True, span_min_length=4
    )

    for d in sst_dataset:
        print(d)

    print("# spans:", len(sst_dataset))
    print("# examples:", len(sst_dataset.trees))

    print(">>> init augmenter")
    sst_augmenter = Sub2Augmenter(sst_dataset)
    print(">>> augmenting...")
    sst_augmented_dataset = sst_augmenter.augment()

    print("# spans:", len(sst_augmented_dataset))
    print("# examples:", len(sst_augmented_dataset.trees))

    for i in range(len(sst_dataset.trees)):
        print(' '.join(sst_dataset.trees[i].leaves()))
        print(' '.join(sst_augmented_dataset.trees[i].leaves()))
    print("------------------")
    for i in range(len(sst_dataset.trees), len(sst_augmented_dataset.trees)):
        print(' '.join(sst_augmented_dataset.trees[i].leaves()))


if __name__ == '__main__':
    augment()