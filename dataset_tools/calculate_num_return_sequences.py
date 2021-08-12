import sys
import json
import spacy
from tqdm import tqdm
import math


if __name__ == '__main__':
    """
    Run this on the scifact claim generation file to calculate the number of return sequences 
    """
    infile = sys.argv[1]
    nlp = spacy.load('en_core_sci_md')

    with open(infile) as f:
        citances = [json.loads(l) for l in f]

    with open(f"{infile}", 'wt') as f:
        for citance in tqdm(citances):
            citance['num_return_sequences'] = int(math.ceil(len(list(nlp(citance['claims']).noun_chunks)) / 2.))
            f.write(json.dumps(citance) + '\n')
