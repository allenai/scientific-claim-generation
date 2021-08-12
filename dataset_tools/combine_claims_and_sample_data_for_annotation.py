import json
import sys
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import random
import pandas as pd
import argparse
from pathlib import Path


def generate_annotation_files(dataset, annotator_id, out_file):
    """
    Generate an annotation sheet
    :param dataset:
    :param annotator_id:
    :param out_file:
    :return:
    """
    final_data = []
    for d in dataset:
        claim_map = defaultdict(list)
        for claim,method in zip(np.array(d['claims']), np.array(d['methods'])):
            claim_map[claim].append((claim,method))
        keys = list(claim_map.keys())
        random.shuffle(keys)
        j = 0
        for k in keys:
            for (c,m) in claim_map[k]:
                final_data.append(
                    [d['id'], m, annotator_id, d['original_sentence'] if j == 0 else None, c, None, None, None, d['context'] if j == 0 else None, d['original_sentence'] if j == 0 else None, c,
                     None, None])
                j += 1
        final_data.append([None]*len(final_data[-1]))
    out_data = pd.DataFrame(final_data, columns=['ID', 'Method', 'annotator', 'Original Sentence', 'Claim', 'Fluency', 'De-Contextualized', 'Atomicity', 'Context', 'Original Sentence', 'Claim', 'Faithfulness', 'Notes'])
    out_data.to_csv(out_file, index=None)

if __name__ == '__main__':
    """
    Create a set of annotation sheets from a given ste of claims files. Claims files are any number of output jsonl files from curriculum learning 
    (the correct output file is output_test_claims.jsonl). Each claim file should contain generated claims for the exact same set of starting citances.
    Can specify the number of annotators, how many unique citatnces they should annotator, and how many citances should be shared for IAA
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--claims_files", type=str, nargs='+', help="Files containing generated claims", default=[])
    parser.add_argument("--shared_size", type=int, help="The number of citances which are shared between all annotators", default=10)
    parser.add_argument("--set_sizes", type=int, nargs='+', help="The number of unique citances to give each annotator", default=[30, 30, 30])
    parser.add_argument("--annotator_names", type=str, nargs='+', help="The names of the annotators", default=['bailey', 'madeleine', 'sophie'])
    parser.add_argument("--output_dir", type=str, help="Output directory", default=None)
    args = parser.parse_args()

    random.seed(1000)
    np.random.seed(1000)

    joint_data = defaultdict(lambda: {'original_sentence': '', 'claims': [], 'scores': [], 'methods': []})
    shared_size = args.shared_size #10
    set_sizes = args.set_sizes #[30, 30, 30]
    ann_ids = args.annotator_names#['bailey', 'madeleine', 'sophie']

    assert len(ann_ids) == len(set_sizes), "Number of annotators must be the same as the number of set sizes"

    # Iterate through all of the claim files
    for claims_file in args.claims_files:
        with open(claims_file) as f:
            cf = Path(claims_file).parts
            # Methods and iterations are sorted into directories "{method_name}/{iteration}/output_test_claims.jsonl
            method = f"{cf[-3]}_{cf[-2]}"
            for l in tqdm(f):
                d = json.loads(l)
                id_ = d['id']
                id_ = id_[:id_.rfind('_')]
                joint_data[id_]['original_sentence'] = d['citance']
                joint_data[id_]['context'] = d['context']
                joint_data[id_]['id'] = id_
                joint_data[id_]['paper_id'] = d['paper_id']
                joint_data[id_]['claims'].append(d['generated_claim'])
                joint_data[id_]['methods'].append(method)


    # Shuffle and sample from the citances
    keys = list(joint_data.keys())
    random.shuffle(keys)
    paper_ids = set()

    # Get shared samples for IAA
    shared_samples = []
    j = 0
    while len(shared_samples) < shared_size and j < len(keys):
        if joint_data[keys[j]]['paper_id'] not in paper_ids:
            shared_samples.append(joint_data[keys[j]])
            paper_ids.add(joint_data[keys[j]]['paper_id'])
        j += 1

    # Generate individual annotation files
    for i,size in enumerate(set_sizes):
        curr = []
        while len(curr) < size and j < len(keys):
            if joint_data[keys[j]]['paper_id'] not in paper_ids:
                curr.append(joint_data[keys[j]])
                paper_ids.add(joint_data[keys[j]]['paper_id'])
            j += 1
        generate_annotation_files(shared_samples + curr, i, f"{args.output_dir}/ann_{i}.csv")