import json
import random
from sklearn.model_selection import train_test_split

seed = 1000
random.seed(1000)

with open('../data/bioasq/training9b.json') as f:
    dset = json.load(f)

match_qs = [q for q in dset['questions'] if q['type'] == 'factoid']

out_rows = []
out_q2c = []
for q in match_qs:
    if len(q['exact_answer']) > 1:
        continue
    ans = q['exact_answer'][0]
    question = q['body']
    contexts = []
    for c in q['snippets']:
        if ans.lower() in c['text'].lower():
            contexts.append(c['text'])
    questions = [question] * len(contexts)
    answers = [ans] * len(contexts)
    out_rows.extend([{'question': q, 'context': c, 'answers': [{'text': a}]} for q,c,a in zip(questions, contexts, answers)])
    out_q2c.append({'question': question, 'answer': ans, 'turker_answer': q['ideal_answer'][0]})

# Question generation data
train,val = train_test_split(out_rows, test_size=0.2, random_state=seed)
with open('../data/bioasq/qg_data_train.json', 'wt') as f:
    for row in train:
        f.write(json.dumps(row) + '\n')

with open('../data/bioasq/qg_data_val.json', 'wt') as f:
    for row in val:
        f.write(json.dumps(row) + '\n')

# Claim generation data
train,val = train_test_split(out_q2c, test_size=0.2, random_state=seed)
with open('../data/bioasq/q2c_data_train.json', 'wt') as f:
    for row in train:
        f.write(json.dumps(row) + '\n')

with open('../data/bioasq/q2c_data_val.json', 'wt') as f:
    for row in val:
        f.write(json.dumps(row) + '\n')