import json
import spacy
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import math

nlp = spacy.load('en_core_sci_md')

with open('../data/scifact/claims_with_citances.jsonl') as f:
    data = [json.loads(l) for l in f]

out_data = []
for d in tqdm(data):
    citance = d['citance']
    sentences = list(nlp(d['citation_paragraph']).sents)
    sent_idx = -1
    for i,sent in enumerate(sentences):
        if sent.text in citance or citance in sent.text:
            sent_idx = i
            break

    if sent_idx == -1:
        print(citance)
        for sent in sentences:
            print(sent.text)
    assert sent_idx >= 0
    prev = sentences[sent_idx - 1].text if sent_idx > 0 else None
    next = sentences[sent_idx + 1].text if sent_idx < len(sentences) - 1 else None

    for c in d['claims']:
        if not c['is_negation']:
            n_ret_seq = int(math.ceil(len(list(nlp(c['text']).noun_chunks)) / 2.))
            out_data.append({'context': f"{prev} {next} || {citance} ||", 'claims': c['text'], 'num_return_sequences': n_ret_seq})


train,dev = train_test_split(out_data, train_size=0.8)
with open('../data/scifact/claim_generation_data_train.jsonl','wt') as f:
    for d in train:
        f.write(json.dumps(d) + '\n')

with open('../data/scifact/claim_generation_data_dev.jsonl','wt') as f:
    for d in dev:
        f.write(json.dumps(d) + '\n')