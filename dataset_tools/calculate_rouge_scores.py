import argparse
import json
from collections import defaultdict
from rouge_score import rouge_scorer
import numpy as np
from tqdm import tqdm
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
from pathlib import Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--claims_files", type=str, nargs='+', help="Files containing generated claims", default=[])
    parser.add_argument("--gold_file", type=str, help="File with gold claims", default=None)
    args = parser.parse_args()

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    # Load the gold claims
    with open(args.gold_file) as f:
        gold_claims = [json.loads(l) for l in f]
    gold_claim_map = defaultdict(list)
    for c in gold_claims:
        gold_claim_map[c['context']].append(c['claims'])

    plt_r1 = []
    plt_r2 = []
    plt_rL = []
    # Go through predictions files
    for pred_file in args.claims_files:
        context_claim_map = defaultdict(list)
        with open(pred_file) as f:
            pred_claims = [json.loads(l) for l in f]
        # Go through gold claims, calculate top rouge score
        r1s = []
        r2s = []
        rLs = []
        for pc in tqdm(pred_claims):
            candidates = gold_claim_map[pc['context']]
            assert len(candidates) > 0
            r1Curr = []
            r2Curr = []
            rLCurr = []
            for cand in candidates:
                scores = scorer.score(pc['generated_claim'], cand)
                r1Curr.append(scores['rouge1'].fmeasure)
                r2Curr.append(scores['rouge2'].fmeasure)
                rLCurr.append(scores['rougeL'].fmeasure)
            r1s.append(max(r1Curr))
            r2s.append(max(r2Curr))
            rLs.append(max(rLCurr))
        print(f"rouge1: {np.mean(r1s)}\trouge2: {np.mean(r2s)}\trougeL: {np.mean(rLs)}")
        plt_r1.append(np.mean(r1s))
        plt_r2.append(np.mean(r2s))
        plt_rL.append(np.mean(rLs))

    fig = plt.figure(figsize=(5,5))
    ax = sns.lineplot(list(range(len(plt_r1))), plt_r1)
    ax.set_title('rouge1')
    fig.tight_layout()
    fig.savefig(f"{Path(args.claims_files[0]).parent.parent}/rouge1.png")
    fig.clf()

    fig = plt.figure(figsize=(5, 5))
    ax = sns.lineplot(list(range(len(plt_r1))), plt_r2)
    ax.set_title('rouge2')
    fig.tight_layout()
    fig.savefig(f"{Path(args.claims_files[0]).parent.parent}/rouge2.png")
    fig.clf()

    fig = plt.figure(figsize=(5, 5))
    ax = sns.lineplot(list(range(len(plt_r1))), plt_rL)
    ax.set_title('rougeL')
    fig.tight_layout()
    fig.savefig(f"{Path(args.claims_files[0]).parent.parent}/rougeL.png")
    fig.clf()