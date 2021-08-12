import argparse
import random
import numpy as np
import torch
import spacy
import scispacy
import json
import os
import pandas as pd

from spacy.training import Example
from tqdm import tqdm
from datasets import Dataset
from functools import partial
from custom_trainer import CustomTrainer
import ipdb
from collections import defaultdict
from scipy.special import softmax
from spacy.util import minibatch, compounding
from generate_claim_variants import generate_negative_claims_using_linker
from transformers import pipeline


import transformers
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    HfArgumentParser,
    set_seed,
    PreTrainedTokenizerBase,
    PreTrainedModel,
    DataCollatorForSeq2Seq,
    AutoModelForCausalLM
)

from ParagraphJointModel.paragraph_model_dynamic import JointParagraphClassifier
from ParagraphJointModel.dataset import SciFactParagraphBatchDataset
from ParagraphJointModel.scifact_joint_paragraph_dynamic_prediction import predict, post_process_stance
from ParagraphJointModel.util import stance2json, rationale2json, merge_json


def enforce_reproducibility(seed=1000):
    # Sets seed manually for both CPU and CUDA
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For atomic operations there is currently
    # no simple way to enforce determinism, as
    # the order of parallel operations is not known.
    # CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # System based
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)


def qg_data_preprocess(tokenizer, dset, examples):
    """
    Data preprocessor for QG model input
    :param tokenizer: QG model tokenizer
    :param dset: Dataset name, either 'local' for citances or a dataset such as squad
    :param examples: The actual data to preprocess
    :return: Tokenizer encoded inputs to QG model
    """
    if dset == 'local':
        inputs = [ctx + ' ' + ans[0]['text'] for ctx, ans in zip(examples['context'], examples['answers'])]
    else:
        inputs = [ctx + ' ' + ans['text'][0] for ctx, ans in zip(examples['context'], examples['answers'])]
    targets = [q for i,q in enumerate(examples['question'])]
    model_inputs = tokenizer(inputs, max_length=tokenizer.model_max_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=tokenizer.model_max_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def q2c_data_preprocess(tokenizer, dset, examples):
    """
        Data preprocessor for claim generation model input
        :param tokenizer: claim generation model tokenizer
        :param dset: Dataset name, either 'citeworth' for citances or a dataset such as squad
        :param examples: The actual data to preprocess
        :return: Tokenizer encoded inputs to claim generation model
        """

    if dset == 'citeworth':
        inputs = [ctx + ' ' + ans['text'] for ctx, ans in zip(examples['generated_question'], examples['answer'])]
        targets = [''] * len(inputs)
    else:
        inputs = [q + ' ' + a for q,a in zip(examples['question'], examples['answer'])]
        targets = [a for a in examples['turker_answer']]
    model_inputs = tokenizer(inputs, max_length=tokenizer.model_max_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=tokenizer.model_max_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def sort_fc_claims(preds, original_claims):
    """
    Scores each claim using the formula:
    $$ s = p[support] - p[contradict] $$
    Returns the claims sorted by this score in descending order
    :param preds: The raw logits from ParagraphJointModel for each evidence sample for each claim
    :param original_claims: The original generated claims
    :return: Sorted claims with their fact checking score
    """
    orig_claim_map = {c['id']: c for c in original_claims}
    for p in preds:
        all_probs = [softmax(p['evidence'][e]['score']) for e in p['evidence']]
        score = max(p[1] - p[2] for p in all_probs)
        orig_claim_map[p['id']]['score'] = score

    return list(sorted([v for v in orig_claim_map.values()], key=lambda x: x['score'], reverse=True))


def save_ner_model(output_dir, nlp, new_model_name):
    """
    Save a spacy model
    :param output_dir: Where to save the model
    :param nlp: The scispacy model to save
    :param new_model_name: New name for the spacy model
    :return:
    """
    output_dir = f'ner_models/{output_dir}'
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        nlp.meta["name"] = new_model_name
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)


def get_named_entities(citances, nlp):
    """
    Extract named entities from a set of citances
    :param citances:
    :param nlp:
    :return: List of dicts containing input to question generation model
    """
    question_gen_input = defaultdict(list)
    for citance_dict in tqdm(citances):
        citance = citance_dict['text'] if 'text' in citance_dict else citance_dict['claims']
        entities = []
        entity_text = []
        doc = nlp(citance)
        entities.extend(list(doc.ents))
        entity_text.extend([e.text for e in doc.ents])
        for ent in entities:
            answers = [{'text': ent.text, 'type': ent.label_, 'start': ent.start_char, 'pos': [t.pos_ for t in ent]}]
            if 'doc_id' in citance_dict:
                sample = {'id': citance_dict['doc_id'], 'paper_id': citance_dict['paper_id'],
                          'context': citance_dict['context'], 'citance': citance, 'answers': answers, 'question': '',
                          'evidence': citance_dict['evidence']}
            else:
                sample = {'id': '', 'paper_id': '',
                          'context': citance_dict['context'], 'citance': citance, 'answers': answers, 'question': '',
                          'evidence': ''}
            for k in sample:
                question_gen_input[k].append(sample[k])
    return question_gen_input


def run_question_generation(trainer, dset, model, tokenizer, device, num_beams):
    """
    Generate a set of questions from a source text and list of answers (named entities)
    :param trainer: HuggingFace trainer
    :param dset: The dataset to generate questions from
    :param model: Question generation model
    :param tokenizer: Tokenizer for the provided model
    :param device: torch device to run on
    :param num_beams: Number of beams for beam search
    :return: A list of dicts containing input to the claim generation model
    """
    dl = trainer.get_test_dataloader(dset)
    all_samples = []
    for b in tqdm(dl):
        input_ids = b['input_ids'].to(device)
        samples = model.generate(
            input_ids,
            num_beams=num_beams,
            max_length=tokenizer.model_max_length,
            early_stopping=True
        )
        all_samples.extend(list(samples.detach().cpu().numpy()))
    claim_gen_input = defaultdict(list)
    for id, con, ans, q, citance, paper_id, evidence in zip(dset['id'], dset['context'], dset['answers'],
                                                            all_samples, dset['citance'], dset['paper_id'],
                                                            dset['evidence']):
        gen_question = tokenizer.decode(q, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        sample = {'id': id, 'paper_id': paper_id, 'context': con, 'answer': ans[0], 'generated_question': gen_question,
                  'citance': citance, 'evidence': evidence}
        for k in sample:
            claim_gen_input[k].append(sample[k])

    return claim_gen_input


def run_claim_generation(trainer, dset, model, tokenizer, device, num_beams):
    """
    Generate a set of claims from a question and list of answers (named entities)
    :param trainer: HuggingFace trainer
    :param dset: The dataset to generate claims from
    :param model: Claim generation model
    :param tokenizer: Tokenizer for the provided model
    :param device: torch device to run on
    :param num_beams: Number of beams for beam search
    :return: A list of dicts containing the generated claims and a list of dicts containing the input to external fact
    checking model
    """
    dl = trainer.get_test_dataloader(dset)
    all_samples = []
    for b in tqdm(dl):
        input_ids = b['input_ids'].to(device)
        samples = model.generate(
            input_ids,
            num_beams=num_beams,
            max_length=tokenizer.model_max_length,
            early_stopping=True
        )
        all_samples.extend(list(samples.detach().cpu().numpy()))

    generated_claims = []
    fc_claim_inputs = []
    count = defaultdict(int)
    for id, con, ans, q, claim, citance, paper_id, evidence in zip(dset['id'], dset['context'],
                                                                   dset['answer'], dset['generated_question'],
                                                                   all_samples, dset['citance'],
                                                                   dset['paper_id'], dset['evidence']):
        gen_claim = tokenizer.decode(claim, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        n = count[id]
        generated_claims.append(
            {'id': f"{id}_{n}", 'paper_id': paper_id, 'context': con, 'citance': citance, 'answer': ans,
             'generated_question': q,
             'generated_claim': gen_claim, 'evidence': evidence})
        fc_claim_inputs.append({'id': f"{id}_{n}", 'claim': gen_claim, 'evidence': {}, 'cited_doc_ids': evidence,
                                'retrieved_doc_ids': evidence})
        count[id] += 1
    return generated_claims, fc_claim_inputs


def retrain_ner_model(ner_data, nlp):
    """
    Run NER training starting from a given spacy model
    :param ner_data: NER training data
    :param nlp: Spacy model to start from
    :return: Trained spacy model
    """
    print(len(ner_data))
    random.shuffle(ner_data)
    N = int(0.8*len(ner_data))
    #Use 20% for validation
    ner_training_data = ner_data[:N]
    ner_validation_data = ner_data[N:]

    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    best_f = 0.0
    patience = 10
    pcounter = 0
    with nlp.disable_pipes(*unaffected_pipes):

        # Training for 100 iterations w/ early stopping
        for iteration in range(100):

            # shuufling examples  before every iteration
            random.shuffle(ner_training_data)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(ner_training_data, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                #texts, annotations = zip(*batch)
                nlp.update(
                    batch,  # batch of annotations
                    drop=0.1,  # dropout - make it harder to memorise data
                    losses=losses,
                )
                #print("Losses", losses)
            # Get validation scores
            f1 = nlp.evaluate(ner_validation_data)['ents_f']
            print(f"Eval f1: {f1}")
            if f1 > best_f:
                best_f = f1
                save_ner_model("curriculum_learning", nlp, "cl-model")
                pcounter = 0
            else:
                pcounter += 1
                if pcounter == patience:
                    break

    return spacy.load("ner_models/curriculum_learning")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_citances", help="Location of the citance data", required=True, type=str)
    parser.add_argument("--val_citances", help="Location of the validation citance data", required=True, type=str)
    parser.add_argument("--test_citances", help="Location of the test citance data", required=True, type=str)
    parser.add_argument("--qg_model_name", help="Name of the model to use for question generation", required=True, type=str)
    parser.add_argument("--q2c_model_name", help="Name of the model to use for question generation", required=True, type=str)
    parser.add_argument("--fc_model_name", help="Name of the fact checking model", required=True,
                        default='roberta-large')
    parser.add_argument("--fc_model_checkpoint", help="Name of the fact checking model", required=False,
                        default=None)
    parser.add_argument("--external_corpus_file", help="Evidence corpus file", required=True,
                        type=str)
    parser.add_argument("--internal_corpus_file", help="Other paragraphs from citance documents", required=True,
                        type=str)
    parser.add_argument("--seed", help="Random seed", type=int, default=1000)
    parser.add_argument("--num_beams", help="Number of beams for beam search", type=int, default=1)
    parser.add_argument("--output_dir", help="Directory to output files", required=True, type=str)


    args = parser.parse_args()

    enforce_reproducibility(args.seed)
    # See if CUDA available
    device = torch.device("cpu")
    if torch.cuda.is_available():
        print("Training on GPU")
        device = torch.device("cuda:0")

    # Setup
    nlp = spacy.load('en_core_sci_md')
    # QG model setup
    qg_model = args.qg_model_name
    qg_tokenizer = AutoTokenizer.from_pretrained(qg_model)
    qg_model = AutoModelForSeq2SeqLM.from_pretrained(qg_model)
    # Q2C model setup
    q2c_model = args.q2c_model_name
    q2c_tokenizer = AutoTokenizer.from_pretrained(q2c_model)
    q2c_model = AutoModelForSeq2SeqLM.from_pretrained(q2c_model)
    # FC model setup
    fc_tokenizer = AutoTokenizer.from_pretrained(args.fc_model_name)

    fc_model = JointParagraphClassifier(args.fc_model_name, 1024,
                                     0.0)
    state_dict = torch.load(args.fc_model_checkpoint)
    # strict = false because of bert.embeddings.position_ids mismatch
    fc_model.load_state_dict(state_dict, strict=False)

    # Language model for negative claim generation
    lm = AutoModelForCausalLM.from_pretrained('gpt2')
    lm_tk = AutoTokenizer.from_pretrained('gpt2')

    ########### Run NER on input
    with open(args.train_citances) as f:
        citances = [json.loads(l) for l in f]
    with open(args.val_citances) as f:
        val_citances = [json.loads(l) for l in f]
    with open(args.test_citances) as f:
        test_citances = [json.loads(l) for l in f]
    ner_data = []
    output_claims = []
    n_epochs = 6
    for epoch in range(n_epochs):
        if not os.path.exists(f"{args.output_dir}/{epoch}"):
            os.makedirs(f"{args.output_dir}/{epoch}")

        save_dir = f"{args.output_dir}/{epoch}"

        question_gen_input = get_named_entities(citances, nlp)
        val_question_gen_input = get_named_entities(val_citances, nlp)
        test_question_gen_input = get_named_entities(test_citances, nlp)

        ############ Generate questions from NER
        qg_model.to(device)
        preprocessor = partial(qg_data_preprocess, qg_tokenizer, 'local')
        gen_dset_base = Dataset.from_dict(question_gen_input)
        val_gen_dset_base = Dataset.from_dict(val_question_gen_input)
        test_gen_dset_base = Dataset.from_dict(test_question_gen_input)

        # Filter missing NER
        #gen_dset_base = gen_dset_base.filter(lambda example: len(example['answers']) > 0)
        gen_dset = gen_dset_base.map(preprocessor, batched=True)
        val_gen_dset = val_gen_dset_base.map(preprocessor, batched=True)
        test_gen_dset = test_gen_dset_base.map(preprocessor, batched=True)

        data_collator = DataCollatorForSeq2Seq(
            qg_tokenizer,
            model=qg_model,
            label_pad_token_id=-100,
            padding='longest'
        )
        qg_trainer = CustomTrainer(
            model=qg_model,
            tokenizer=qg_tokenizer,
            data_collator=data_collator
        )
        claim_gen_input = run_question_generation(qg_trainer, gen_dset, qg_model, qg_tokenizer, device, args.num_beams)
        val_claim_gen_input = run_question_generation(qg_trainer, val_gen_dset, qg_model, qg_tokenizer, device, args.num_beams)
        test_claim_gen_input = run_question_generation(qg_trainer, test_gen_dset, qg_model, qg_tokenizer, device, args.num_beams)

        qg_model.to('cpu')

        ############ Generate claims from questions
        q2c_model.to(device)
        preprocessor = partial(q2c_data_preprocess, q2c_tokenizer, 'citeworth')
        gen_dset_base = Dataset.from_dict(claim_gen_input)
        val_gen_dset_base = Dataset.from_dict(val_claim_gen_input)
        test_gen_dset_base = Dataset.from_dict(test_claim_gen_input)

        gen_dset = gen_dset_base.map(preprocessor, batched=True)
        val_gen_dset = val_gen_dset_base.map(preprocessor, batched=True)
        test_gen_dset = test_gen_dset_base.map(preprocessor, batched=True)
        data_collator = DataCollatorForSeq2Seq(
            q2c_tokenizer,
            model=q2c_model,
            label_pad_token_id=-100,
            padding='longest'
        )
        q2c_trainer = CustomTrainer(
            model=q2c_model,
            tokenizer=q2c_tokenizer,
            data_collator=data_collator
        )

        generated_claims, fc_claim_inputs = run_claim_generation(q2c_trainer, gen_dset, q2c_model, q2c_tokenizer, device, args.num_beams)
        val_generated_claims, _ = run_claim_generation(q2c_trainer, val_gen_dset, q2c_model, q2c_tokenizer,
                                                                 device, args.num_beams)
        test_generated_claims, _ = run_claim_generation(q2c_trainer, test_gen_dset, q2c_model, q2c_tokenizer, device,
                                                                args.num_beams)

        with open(f"{save_dir}/output_test_claims.jsonl", 'wt') as f:
            for c in test_generated_claims:
                f.write(json.dumps(c) + '\n')
        with open(f"{save_dir}/output_scifact_dev_claims.jsonl", 'wt') as f:
            for c in val_generated_claims:
                f.write(json.dumps(c) + '\n')

        q2c_model.to('cpu')
        # Run FC model
        fc_model.to(device)
        #TODO get the data into the right format
        fc_dev_set = SciFactParagraphBatchDataset(args.external_corpus_file, fc_claim_inputs,
                                                  sep_token=fc_tokenizer.sep_token, k=0, train=False)

        rationale_predictions, stance_preds, stance_scores = predict(fc_model, fc_dev_set, 16, args.fc_model_name, fc_tokenizer, device)
        rationale_json = rationale2json(fc_dev_set.samples, rationale_predictions)
        stance_json = stance2json(fc_dev_set.samples, stance_preds, stance_scores)
        stance_json = post_process_stance(rationale_json, stance_json)
        merged_json = merge_json(rationale_json, stance_json)
        fc_model.to('cpu')
        # Rank predictions
        sorted_fc_claims = sort_fc_claims(merged_json, generated_claims)
        # Get new entities
        citance_entity_map = defaultdict(lambda: {'text': '', 'entities': []})
        original_claims = [c for c in sorted_fc_claims if c['score'] > 0.5]
        for c in original_claims:
            citance_entity_map[c['id']]['text'] = c['citance']
            citance_entity_map[c['id']]['entities'].append(
                (c['answer']['start'], c['answer']['start'] + len(c['answer']['text']), 'ENTITY'))
        ner_data.extend([Example.from_dict(nlp.make_doc(citance_entity_map[c]['text']),
                                      {'entities': citance_entity_map[c]['entities']}) for c in citance_entity_map])
        # Continue NER training starting from initial entity recognizer
        nlp = retrain_ner_model(ner_data, spacy.load('en_core_sci_md'))
        output_claims.extend(original_claims)
        citances = [c for c in citances if c['doc_id'] not in citance_entity_map]


        if epoch == n_epochs - 1:
            output_claims.extend([c for c in sorted_fc_claims if c['score'] <= 0.5])

        with open(f"{save_dir}/added_claims.jsonl", 'wt') as f:
            for c in output_claims:
                f.write(json.dumps(c) + '\n')

        csv_out = []
        for c in output_claims:
            csv_out.append([c['context'], c['citance'], c['generated_claim'], c['score']])
        csv_pd = pd.DataFrame(csv_out, columns=['Context', 'Original Sentence', 'Claim', 'Score'])
        csv_pd.to_csv(f"{save_dir}/ranked_claims.csv", index=None)

        # Generate training data for fact checking
        nli = pipeline('sentiment-analysis', model='roberta-large-mnli', return_all_scores=True, device=0)

        # Generate data for scifact training/evaluation
        for claim_set in tqdm(test_generated_claims):
            neg_claims = generate_negative_claims_using_linker([claim_set['generated_claim']], nli, lm, lm_tk, device, 3)
            claim_set['neg_claim'] = neg_claims[0][2] if neg_claims[0] is not None else None
        # Get corpus so we can pick negative samples for NEI
        paper_id_to_paragraph = defaultdict(list)
        with open(args.internal_corpus_file) as f:
            for l in f:
                data = json.loads(l)
                paper_id = data['doc_id'].split('_')[0]
                paper_id_to_paragraph[paper_id].append(data)


        # Pick 1/3 to be supports, 1/3 to be contradicts, and 1/3 to be NEI
        def incgen():
            val = 0
            while True:
                val += 1
                yield val


        inc = incgen()
        base_claims_and_evidence = []
        for claim_set in test_generated_claims:
            # Remove ID suffix to get original paper ID
            original_doc_id = claim_set['id']
            original_doc_id = original_doc_id[:original_doc_id.rfind('_')]

            pos_claim = claim_set['generated_claim']
            neg_claim = claim_set['neg_claim']
            type = random.randint(0, 2)
            if type == 0 or neg_claim == None:
                base_claims_and_evidence.append({
                    'id': next(inc),
                    'claim': pos_claim,
                    'evidence': {str(doc_id): [{'sentences': [0], 'label': 'SUPPORT'}] for doc_id in
                                 claim_set['evidence']},
                    'cited_doc_ids': claim_set['evidence']
                })
            elif type == 1:
                base_claims_and_evidence.append({
                    'id': next(inc),
                    'claim': neg_claim,
                    'evidence': {str(doc_id): [{'sentences': [0], 'label': 'CONTRADICT'}] for doc_id in
                                 claim_set['evidence']},
                    'cited_doc_ids': claim_set['evidence']
                })
            elif type == 2:
                nei_type = random.randint(0, 1)
                if nei_type == 0:
                    base_claims_and_evidence.append({
                        'id': next(inc),
                        'claim': pos_claim,
                        'evidence': {},
                        'cited_doc_ids': [original_doc_id]
                    })
                else:
                    base_claims_and_evidence.append({
                        'id': next(inc),
                        'claim': neg_claim,
                        'evidence': {},
                        'cited_doc_ids': [original_doc_id]
                    })
        with open(f"{save_dir}/scifact_claims.jsonl", 'wt') as f:
            for c in base_claims_and_evidence:
                f.write(json.dumps(c) + '\n')
