import argparse
import torch
import spacy
import scispacy
from scispacy.linking import EntityLinker
import json
from tqdm import tqdm

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_loc", help="Location of the citance data", required=True, type=str)
    parser.add_argument("--output_fname", help="Name of the output file", required=True, type=str)
    parser.add_argument("--check_linker", help="Whether to check if entities are linked in UMLS and at a low enough depth", action="store_true", default=False)

    args = parser.parse_args()

    # scibert = spacy.load('en_core_sci_scibert')
    # craft = spacy.load('en_ner_craft_md')
    # jnlpba = spacy.load('en_ner_jnlpba_md')
    # bc5cdr = spacy.load('en_ner_bc5cdr_md')
    # bionlp13cg = spacy.load('en_ner_bionlp13cg_md')
    nlp = spacy.load('en_core_sci_md')
    if args.check_linker:
        nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
        linker = nlp.get_pipe("scispacy_linker")

    # Load data
    # data = []
    # with open(f'{args.data_loc}/train.jsonl') as f:
    #     curr = [json.loads(l) for l in f]
    #     for j,d in enumerate(curr):
    #         d['split'] = f'train_{j}'
    #     data.extend(curr)
    # with open(f'{args.data_loc}/dev.jsonl') as f:
    #     curr = [json.loads(l) for l in f]
    #     for j,d in enumerate(curr):
    #         d['split'] = f'dev_{j}'
    #     data.extend(curr)
    # #citances = {f"{doc['split']}_{i}": {'text': s['text'], 'context': doc['original_text']} for j,doc in enumerate(data) for i,s in enumerate(doc['samples']) if s['label'] == 'check-worthy' and doc['mag_field_of_study'][0] in ['Biology', 'Medicine']}
    # citances = {}
    # for j,doc in enumerate(data):
    #     for i, s in enumerate(doc['samples']):
    #         if s['label'] in ['check-worthy', 'context-only'] and doc['mag_field_of_study'][0] in ['Biology', 'Medicine']:
    #             # Get evidence
    #             evidence = []
    #             if s['ref_ids'] != None:
    #                 evidence = [v[1] for group in s['ref_ids'] for v in group]
    #             # Get context
    #             prev_sent = '' if i == 0 else doc['samples'][i-1]['text']
    #             next_sent = '' if i == len(doc['samples']) - 1 else doc['samples'][i+1]['text']
    #             citances[f"{doc['split']}_{i}"] = {'text': s['text'], 'context': f"{prev_sent} {s['text']} {next_sent}", 'paper_id': doc['paper_id'], 'evidence': evidence}
    with open(args.data_loc) as f:
        citances = [json.loads(l) for l in f]

    # Run prediction on data
    with open(f'{args.output_fname}', 'wt') as f:#, open(f'{args.output_dir}/citance_noun_chunks.jsonl', 'wt') as g:
        for citance_dict in tqdm(citances):
            #citance = citances[id_]['text']
            citance = citance_dict['text']
            entities = []
            entity_text = []
            doc = nlp(citance)
            entities.extend(list(doc.ents))
            if args.check_linker:
                final_entities = []
                for ent in entities:
                    if len(ent._.kb_ents) > 0:
                        max_depth = max(linker.kb.semantic_type_tree.get_node_from_id(t).level for t in linker.kb.cui_to_entity[ent._.kb_ents[0][0]].types)
                        if max_depth > 5:
                            entity_text.append(ent.text)
                            final_entities.append(ent)
                entities = final_entities
            else:
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
            for ent in entities:
                answers = [{'text': ent.text, 'type': ent.label_, 'start': ent.start_char, 'pos': [t.pos_ for t in ent]}]
                #f.write(json.dumps({'id': id_, 'paper_id': citances[id_]['paper_id'], 'paragraph': citances[id_]['context'], 'context': citance, 'answers': answers, 'question': ''}) + '\n')
                f.write(json.dumps({'id': citance_dict['doc_id'], 'paper_id': citance_dict['paper_id'], 'context': citance_dict['context'], 'citance': citance, 'answers': answers, 'question': '', 'evidence': citance_dict['evidence']}) + '\n')

            # Get noun chunks
            # doc = scibert(citance)
            # answers = [{'text': np.text, 'start': np.start_char} for
            #            np in doc.noun_chunks]
            # g.write(json.dumps({'context': citance, 'answers': answers, 'question': ''}) + '\n')