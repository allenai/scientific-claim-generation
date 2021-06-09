import argparse
import torch
import spacy
import json
from tqdm import tqdm

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_loc", help="Location of the citance data", required=True, type=str)
    parser.add_argument("--output_dir", help="Location of the output directory", required=True, type=str)

    args = parser.parse_args()

    # scibert = spacy.load('en_core_sci_scibert')
    # craft = spacy.load('en_ner_craft_md')
    # jnlpba = spacy.load('en_ner_jnlpba_md')
    # bc5cdr = spacy.load('en_ner_bc5cdr_md')
    # bionlp13cg = spacy.load('en_ner_bionlp13cg_md')
    nlp = spacy.load('en_core_sci_md')

    # Load data
    data = []
    with open(f'{args.data_loc}/train.jsonl') as f:
        data.extend([json.loads(l) for l in f])
    with open(f'{args.data_loc}/dev.jsonl') as f:
        data.extend([json.loads(l) for l in f])

    citances = [s['text'] for doc in data for s in doc['samples'] if s['label'] == 'check-worthy' and doc['mag_field_of_study'][0] in ['Biology', 'Medicine']]

    # Run prediction on data
    with open(f'{args.output_dir}/citance_ner.jsonl', 'wt') as f:#, open(f'{args.output_dir}/citance_noun_chunks.jsonl', 'wt') as g:
        for citance in tqdm(citances):
            entities = []
            entity_text = []
            doc = nlp(citance)
            entities.extend(list(doc.ents))
            entity_text.extend([e.text for e in doc.ents])
            # doc = jnlpba(citance)
            # entities.extend([e for e in doc.ents if e.text not in entity_text])
            # entity_text.extend([e.text for e in doc.ents if e.text not in entity_text])
            # doc = bc5cdr(citance)
            # entities.extend([e for e in doc.ents if e.text not in entity_text])
            # entity_text.extend([e.text for e in doc.ents if e.text not in entity_text])
            # doc = bionlp13cg(citance)
            # entities.extend([e for e in doc.ents if e.text not in entity_text])
            # entity_text.extend([e.text for e in doc.ents if e.text not in entity_text])

            answers = [{'text': ent.text, 'type': ent.label_, 'start': ent.start_char, 'pos': [t.pos_ for t in ent]} for ent in entities]
            f.write(json.dumps({'context': citance, 'answers': answers, 'question': ''}) + '\n')

            # Get noun chunks
            # doc = scibert(citance)
            # answers = [{'text': np.text, 'start': np.start_char} for
            #            np in doc.noun_chunks]
            # g.write(json.dumps({'context': citance, 'answers': answers, 'question': ''}) + '\n')