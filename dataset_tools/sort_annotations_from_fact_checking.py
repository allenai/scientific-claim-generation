import krippendorff
from sklearn.metrics import cohen_kappa_score
import pandas as pd
import argparse
import ipdb
import numpy as np
from collections import defaultdict
import json
import math
import ipdb
from scipy.special import softmax
from scipy.stats import pearsonr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_files", nargs='+', help="List of annotation files", default=[])
    parser.add_argument("--predictions_file", type=str, help="Predictions for each claim", default=None)
    parser.add_argument("--original_claims", type=str, help="The original claims for evidence IDs", default=None)
    parser.add_argument("--output_claims", type=str, help="Output file for scifact-style claims", default=None)
    args = parser.parse_args()
    annotations = [pd.read_csv(f).dropna(how='all').fillna(np.nan) for f in args.annotation_files]

    with open(args.predictions_file) as f:
        preds = [json.loads(l) for l in f]
    orig_claim_map = {}
    with open(args.original_claims) as f:
        for l in f:
            c = json.loads(l)
            orig_claim_map[c['id']] = c

    # Combine claims for dataset, separate multi-annotated claims from singly annotated claims
    split_point = 0
    while all(a.values[split_point,0] == annotations[0].values[split_point,0] for a in annotations):
        split_point += 1
    annotations_multi = [a[:split_point] for a in annotations]

    # TODO combine annotations better
    all_annotations = pd.concat([annotations[0][:split_point]] + [ann[split_point:] for ann in annotations])

    # Fix the ids
    ids = all_annotations['ID'].to_numpy()
    orig_sent = all_annotations['Original Sentence'].to_numpy()
    orig_sent1 = all_annotations['Original Sentence.1'].to_numpy()
    context = all_annotations['Context'].to_numpy()
    k = 1
    ids[0] = ids[0] + f'_{k}'
    for i in range(1,all_annotations.shape[0]):
        if ids[i] in ids[i-1]:
            k += 1
        else:
            k = 1
        if not isinstance(orig_sent[i], str):
            orig_sent[i] = orig_sent[i-1]
            orig_sent1[i] = orig_sent1[i - 1]
            context[i] = context[i - 1]
        ids[i] = ids[i] + f"_{k}"
    all_annotations['ID'] = ids
    all_annotations['Original Sentence'] = orig_sent
    all_annotations['Original Sentence.1'] = orig_sent1
    all_annotations['Context'] = context

    # Get the scores
    id_score = {}
    # for p in preds:
    #     score = 0
    #     if len(p['evidence']) == 0:
    #         score = -100
    #     else:
    #         for e in p['evidence']:
    #             ev = p['evidence'][e]
    #             cited_docs = orig_claim_map[p['id']]['cited_doc_ids']
    #             if ev['label'] == 'SUPPORT' and e in cited_docs:
    #                 score += 1 + softmax(ev['score'])[1]
    #
    #             elif ev['label'] != 'SUPPORT':
    #                 score -= 1
    #     id_score[p['id']] = score
    for p in preds:
        score = 0
        cited_docs = orig_claim_map[p['id']]['cited_doc_ids']
        found = False
        for e in p['evidence']:
            if e in cited_docs:
                found = True
                ev = p['evidence'][e]
                probs = softmax(ev['score'])
                # SUPPORTS prob - CONTRADICTS prob
                score += probs[1] - probs[2]
        id_score[p['id']] = score if found else -100
    final_scores = [id_score[id] for id in all_annotations['ID'].to_numpy()]
    all_annotations['scores'] = final_scores
    sorted_annotations = all_annotations.sort_values(by='scores', ascending=False)
    sorted_annotations.to_csv(args.output_claims, index=None)

    # calculate pearson correlation
    fluencies = sorted_annotations[(sorted_annotations['Fluency'].notnull()) & (sorted_annotations['scores'] > -100)]
    fluency = [int(f[0]) for f in fluencies['Fluency']]
    print(f"Fluency: {pearsonr(fluency, fluencies['scores'])}")

    decon = sorted_annotations[(sorted_annotations['De-Contextualized'].notnull()) & (sorted_annotations['scores'] > -100)]
    dec = [int(f[0]) for f in decon['De-Contextualized']]
    print(f"De-Contextualized: {pearsonr(dec, decon['scores'])}")

    atom = sorted_annotations[
        (sorted_annotations['Atomicity'].notnull()) & (sorted_annotations['scores'] > -100)]
    atomic = [int(f[0]) for f in atom['Atomicity']]
    print(f"Atomicity: {pearsonr(atomic, atom['scores'])}")

    faith = sorted_annotations[
        (sorted_annotations['Faithfulness'].notnull()) & (sorted_annotations['scores'] > -100)]
    faithful = [int(f[0]) for f in faith['Faithfulness']]
    print(f"Faithfulness: {pearsonr(faithful, faith['scores'])}")

