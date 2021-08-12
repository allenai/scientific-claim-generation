import json
import argparse
import random
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
import math
import spacy


if __name__ == '__main__':
    """
    Starting with a dataset from CiteWorth, create a train/dev/test split formatted for use with curriculum learning scripts
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--citance_file", type=str, help="File containing citeworth citances", default=None)
    parser.add_argument("--corpus_file", type=str, help="Corpus to filter down valid citances. Valid citances will have all cited documents available in this corpus", default=None)
    parser.add_argument("--internal_corpus_file", type=str, help="Corpus with all paragraphs from papers containing the citances", default=None)
    parser.add_argument("--old_citance_files", nargs='+', help="List of old annotation files so we can ignore these citances (for sampling new data)", default=[])
    parser.add_argument("--output_dir", type=str, help="Output citance file", default=None)
    args = parser.parse_args()

    random.seed(1000)
    np.random.seed(1000)

    nlp = spacy.load('en_core_sci_md')

    with open(args.citance_file) as f:
        citeworth = [json.loads(l) for l in f]

    corpus_ids = set()
    with open(args.corpus_file) as f:
        for l in f:
            abstract = json.loads(l)
            corpus_ids.add(abstract['doc_id'])

    internal_corpus = defaultdict(list)
    with open(args.internal_corpus_file) as f:
        for l in f:
            abstract = json.loads(l)
            paper_id = abstract['doc_id'].split('_')[0]
            internal_corpus[paper_id].append(abstract)

    ignore_ids = set()
    for file in args.old_citance_files:
        with open(file) as f:
            ignore_ids.update(set([json.loads(l)['doc_id'] for l in f]))

    # Filter out citances we don't care about
    citance_paper_map = defaultdict(list)
    corpus = []
    all_papers = []
    for j,doc in enumerate(citeworth):
        for i, s in enumerate(doc['samples']):
            id_ = f"evaluate_{j}_{i}"
            if s['label'] in ['check-worthy', 'context-only'] and doc['mag_field_of_study'][0] in ['Biology',
                                                                                                   'Medicine'] \
                    and s['ref_ids'] is not None and all(ref[1] in corpus_ids for group in s['ref_ids'] for ref in group) \
                    and id_ not in ignore_ids:
                # Get evidence
                main_paper_id = doc['paper_id'].split('_')[0]
                evidence = [v[1] for group in s['ref_ids'] for v in group]
                # Get context
                prev_sent = '' if i == 0 else doc['samples'][i - 1]['text']
                next_sent = '' if i == len(doc['samples']) - 1 else doc['samples'][i + 1]['text']
                other_sents = [other['text'] for other in doc['samples'] if other['text'] not in [prev_sent, next_sent, s['text']]]
                n_ret_seq = int(math.ceil(len(list(nlp(s['text']).noun_chunks)) / 2.))
                citance_paper_map[main_paper_id].append({'doc_id': id_, 'text': s['text'],
                                           'context': f"{prev_sent} {s['text']} {next_sent}",
                                           'paper_id': doc['paper_id'], 'evidence': evidence,
                                 'generation_context': f"{prev_sent} {next_sent} || {s['text']} ||",
                                 'other_sentences': other_sents, 'num_return_sequences': n_ret_seq})
                corpus.append({'doc_id': doc['paper_id'], 'title': "", 'abstract': [sent['text'] for sent in doc['samples']]})
                all_papers.append(main_paper_id)
    # Sample and split the data
    train_samples = 5000
    dev_samples = 1000
    test_samples = 1000

    # Do it based on paper so there is no overlap between train/dev/test papers
    all_papers, train_papers = train_test_split(all_papers, test_size=train_samples, random_state=1000)
    train_set = [random.sample(citance_paper_map[paper_id], k=1)[0] for paper_id in train_papers]
    with open(f"{args.output_dir}/train.jsonl", 'wt') as f:
        for c in train_set:
            f.write(json.dumps(c) + '\n')

    all_papers, dev_papers = train_test_split(all_papers, test_size=dev_samples, random_state=1000)
    dev_set = [random.sample(citance_paper_map[paper_id], k=1)[0] for paper_id in dev_papers]
    with open(f"{args.output_dir}/dev.jsonl", 'wt') as f:
        for c in dev_set:
            f.write(json.dumps(c) + '\n')

    all_papers, test_papers = train_test_split(all_papers, test_size=test_samples, random_state=1000)
    test_set = [random.sample(citance_paper_map[paper_id], k=1)[0] for paper_id in test_papers]
    with open(f"{args.output_dir}/test.jsonl", 'wt') as f:
        for c in test_set:
            f.write(json.dumps(c) + '\n')

    # Internal corpus points to paragraphs from the same documents as the citances
    with open(f"{args.output_dir}/internal_corpus.jsonl", 'wt') as f:
        for paper_id in train_papers + dev_papers + test_papers:
            for abstract in internal_corpus[paper_id]:
                f.write(json.dumps(abstract) + '\n')

