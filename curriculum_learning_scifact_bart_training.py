import argparse
import random
import numpy as np
import torch
import spacy
import scispacy
import json
import os
import logging
import pandas as pd
import sys

from tqdm import tqdm
from datasets import Dataset
from functools import partial
from dataclasses import dataclass, field
from custom_trainer import CustomTrainer
import ipdb
from collections import defaultdict
from scipy.special import softmax
from typing import Optional, Union
from datasets import load_dataset, load_metric, concatenate_datasets
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


logger = logging.getLogger(__name__)


@dataclass
class DataAndModelArguments:
    """
    Arguments for setting up training/evaluation which aren't captured by huggingface
    """
    model_name: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    output_claim_dir: Optional[str] = field(
        default=None, metadata={"help": "Head directory to output all generated files"}
    )
    external_corpus_file: Optional[str] = field(
        default=None, metadata={"help": "Corpus for fact checking model"}
    )
    internal_corpus_file: Optional[str] = field(
        default=None, metadata={"help": "Other paragraphs in citance documents"}
    )
    fc_model_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "Location of pretrained fact checking model"}
    )
    fc_model_name: Optional[str] = field(
        default=None, metadata={"help": "Name of fact checking base model"}
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )
    train_dset: str = field(
        default='squad', metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    val_dset: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use for validation."}
    )
    predict_dset: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use for final generation."}
    )
    test_dset: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use for generating claims for human evaluation."}
    )
    should_log: bool = field(
        default=True,
        metadata={"help": "Whether or not to log"},
    )


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


def data_preprocess(tokenizer, dset, examples):
    inputs = [c for c in examples['context']]
    targets = [c for c in examples['claims']]
    model_inputs = tokenizer(inputs, max_length=tokenizer.model_max_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=tokenizer.model_max_length, truncation=True)

    # # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # # padding in the loss.
    # if padding == "max_length" and data_args.ignore_pad_token_for_loss:
    #     labels["input_ids"] = [
    #         [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    #     ]

    model_inputs["labels"] = labels["input_ids"]
    if 'doc_id' not in examples:
        model_inputs['doc_id'] = [''] * len(examples['context'])
        model_inputs['paper_id'] = [''] * len(examples['context'])
        model_inputs['evidence'] = [['']] * len(examples['context'])
        #model_inputs['score'] = [[0.0]] * len(examples['context'])
        model_inputs['score'] = [0.0] * len(examples['context'])
        model_inputs['orig_context'] = [''] * len(examples['context'])
        model_inputs['citance'] = [''] * len(examples['context'])
        if 'num_return_sequences' not in examples:
            model_inputs['num_return_sequences'] = [0] * len(examples['context'])
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


def compute_metrics(tokenizer, metric, eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [lab.strip() for lab in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    #result = {"rouge": result["score"]}
    result = {
        'rouge1_low_p': result['rouge1'].low.precision,
        'rouge1_low_r': result['rouge1'].low.recall,
        'rouge1_low_fmeasure': result['rouge1'].low.fmeasure,
        'rouge1_mid_p': result['rouge1'].mid.precision,
        'rouge1_mid_r': result['rouge1'].mid.recall,
        'rouge1_mid_fmeasure': result['rouge1'].mid.fmeasure,
        'rouge1_high_p': result['rouge1'].high.precision,
        'rouge1_high_r': result['rouge1'].high.recall,
        'rouge1_high_fmeasure': result['rouge1'].high.fmeasure,
        'rouge2_low_p': result['rouge2'].low.precision,
        'rouge2_low_r': result['rouge2'].low.recall,
        'rouge2_low_fmeasure': result['rouge2'].low.fmeasure,
        'rouge2_mid_p': result['rouge2'].mid.precision,
        'rouge2_mid_r': result['rouge2'].mid.recall,
        'rouge2_mid_fmeasure': result['rouge2'].mid.fmeasure,
        'rouge2_high_p': result['rouge2'].high.precision,
        'rouge2_high_r': result['rouge2'].high.recall,
        'rouge2_high_fmeasure': result['rouge2'].high.fmeasure,
        'rougeL_low_p': result['rougeL'].low.precision,
        'rougeL_low_r': result['rougeL'].low.recall,
        'rougeL_low_fmeasure': result['rougeL'].low.fmeasure,
        'rougeL_mid_p': result['rougeL'].mid.precision,
        'rougeL_mid_r': result['rougeL'].mid.recall,
        'rougeL_mid_fmeasure': result['rougeL'].mid.fmeasure,
        'rougeL_high_p': result['rougeL'].high.precision,
        'rougeL_high_r': result['rougeL'].high.recall,
        'rougeL_high_fmeasure': result['rougeL'].high.fmeasure,
        'rougeLsum_low_p': result['rougeLsum'].low.precision,
        'rougeLsum_low_r': result['rougeLsum'].low.recall,
        'rougeLsum_low_fmeasure': result['rougeLsum'].low.fmeasure,
        'rougeLsum_mid_p': result['rougeLsum'].mid.precision,
        'rougeLsum_mid_r': result['rougeLsum'].mid.recall,
        'rougeLsum_mid_fmeasure': result['rougeLsum'].mid.fmeasure,
        'rougeLsum_high_p': result['rougeLsum'].high.precision,
        'rougeLsum_high_r': result['rougeLsum'].high.recall,
        'rougeLsum_high_fmeasure': result['rougeLsum'].high.fmeasure,
    }

    #prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    #result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 6) for k, v in result.items()}
    return result


def generate_claims(model, gen_dset, dl, tokenizer, device):
    """
    Run generation using the given model on the given dataset
    :param model: BART model to use for generation
    :param gen_dset: The original dataset
    :param dl: A dataloader to use for generation
    :param tokenizer: A tokenizer for the given model
    :param device: What device to run on
    :return: The set of generated claims to use for the generative model, and the same claims formatted
    for input to the fact checking model
    """
    # Predict
    all_samples = []
    j = 0
    for b in tqdm(dl):
        input_ids = b['input_ids'].to(device)
        # Get the number of return sequences for this batch
        n_return_sequences = gen_dset['num_return_sequences'][j:j+input_ids.shape[0]]
        n_gen_seq = max(n_return_sequences)
        j += input_ids.shape[0]
        # Generate the max number of needed sequences for the batch
        samples = model.generate(
            input_ids,
            max_length=tokenizer.model_max_length,
            early_stopping=True,
            do_sample=True,
            num_return_sequences=n_gen_seq,
            top_k=n_gen_seq
        )
        samples = samples.reshape((input_ids.shape[0], n_gen_seq, -1))
        # Just get the number of sequences needed for each sample
        all_samples.extend([s[:n_seq] for s,n_seq in zip(list(samples.detach().cpu().numpy()), n_return_sequences)])
    fc_claim_inputs = []
    generated_claims = []
    count = defaultdict(int)
    for id, context, claims, evidence, orig_context, citance, paper_id, num_return_sequences in zip(gen_dset['doc_id'],
                                                                             gen_dset['context'],
                                                                             all_samples,
                                                                             gen_dset['evidence'],
                                                                             gen_dset['orig_context'],
                                                                             gen_dset['citance'],
                                                                             gen_dset['paper_id'],
                                                                             gen_dset['num_return_sequences']):
        gen_claims = set([tokenizer.decode(c, skip_special_tokens=True, clean_up_tokenization_spaces=False) for c in claims])
        for c in gen_claims:
            n = count[id]
            generated_claims.append(
                {'id': f"{id}_{n}", 'context': context,
                 'generated_claim': c, 'evidence': evidence, 'orig_context': orig_context, 'citance': citance,
                 'paper_id': paper_id, 'num_return_sequences': num_return_sequences})
            fc_claim_inputs.append({'id': f"{id}_{n}", 'claim': c, 'evidence': {}, 'cited_doc_ids': evidence,
                                    'retrieved_doc_ids': evidence})
            count[id] += 1

    return generated_claims, fc_claim_inputs

if __name__ == '__main__':
    parser = HfArgumentParser((DataAndModelArguments, Seq2SeqTrainingArguments))
    dm_args, training_args = parser.parse_args_into_dataclasses()

    # See if CUDA available
    device = torch.device("cpu")
    if torch.cuda.is_available():
        print("Training on GPU")
        device = torch.device("cuda:0")

    seed = training_args.seed
    model_name = dm_args.model_name
    n_gpu = training_args.n_gpu
    should_log = dm_args.should_log
    train_dset_name = dm_args.train_dset
    val_dset_name = dm_args.val_dset
    enforce_reproducibility(seed)
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if should_log else logging.WARN)
    # Set the verbosity to info of the Transformers logger (on main process only):
    if should_log:
        transformers.utils.logging.set_verbosity_info()
    logger.info(f"Training/evaluation parameters {training_args}")

    # FC model setup
    fc_tokenizer = AutoTokenizer.from_pretrained(dm_args.fc_model_name)

    fc_model = JointParagraphClassifier(dm_args.fc_model_name, 1024,
                                     0.0)
    state_dict = torch.load(dm_args.fc_model_checkpoint)
    fc_model.load_state_dict(state_dict, strict=False)

    # NLI model for negative claim generation
    nli = pipeline('sentiment-analysis', model='roberta-large-mnli', return_all_scores=True, device=0)
    # Language model for negative claim generation
    lm = AutoModelForCausalLM.from_pretrained('gpt2')
    lm_tk = AutoTokenizer.from_pretrained('gpt2')

    nlp = spacy.load('en_core_sci_md')

    # Load data
    # Create train/val datasets and processor
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_preprocessor = partial(data_preprocess, tokenizer, train_dset_name)
    train_dset_base = load_dataset('json', data_files=[train_dset_name])
    train_dset_base = train_dset_base.map(train_preprocessor, batched=True)['train']

    val_preprocessor = partial(data_preprocess, tokenizer, val_dset_name)
    val_dset_base = load_dataset('json', data_files=[val_dset_name])
    val_dset = val_dset_base.map(val_preprocessor, batched=True)['train']
    metric = load_metric('rouge')

    with open(dm_args.predict_dset) as f:
        citances = [json.loads(l) for l in f]
    # Prepare prediction input
    prediction_data = defaultdict(list)
    for citance in tqdm(citances):
        prediction_data['doc_id'].append(citance['doc_id'])
        prediction_data['paper_id'].append(citance['paper_id'])
        prediction_data['orig_context'].append(citance['context'])
        prediction_data['citance'].append(citance['text'])
        prediction_data['context'].append(citance['generation_context'])
        prediction_data['claims'].append("")
        prediction_data['evidence'].append(citance['evidence'])
        prediction_data['num_return_sequences'].append(citance['num_return_sequences'])

    # Create predict dset
    pred_preprocessor = partial(data_preprocess, tokenizer, 'citeworth')
    gen_dset_base = Dataset.from_dict(prediction_data)
    gen_dset = gen_dset_base.map(pred_preprocessor, batched=True)

    # Run on the final claims for annotation
    with open(dm_args.test_dset) as f:
        test_citances = [json.loads(l) for l in f]
    # Prepare prediction input
    test_data = defaultdict(list)
    for citance in tqdm(test_citances):
        test_data['doc_id'].append(citance['doc_id'])
        test_data['paper_id'].append(citance['paper_id'])
        test_data['orig_context'].append(citance['context'])
        test_data['citance'].append(citance['text'])
        test_data['context'].append(citance['generation_context'])
        test_data['claims'].append("")
        test_data['evidence'].append(citance['evidence'])
        test_data['num_return_sequences'].append(citance['num_return_sequences'])

    # Create test dset
    test_preprocessor = partial(data_preprocess, tokenizer, 'citeworth')
    test_dset_base = Dataset.from_dict(test_data)
    test_dset = test_dset_base.map(test_preprocessor, batched=True)

    added_training_data = None
    final_claims = []
    n_epochs = 6
    for epoch in range(n_epochs):
        if len(gen_dset) == 0:
            break
        if not os.path.exists(f"{dm_args.output_claim_dir}/{epoch}"):
            os.makedirs(f"{dm_args.output_claim_dir}/{epoch}")

        save_dir = f"{dm_args.output_claim_dir}/{epoch}"
        # Create the model
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=-100,
            padding='longest'
        )
        # Concatenate base training data and added training data
        if added_training_data is not None:
            train_dset = concatenate_datasets([train_dset_base, added_training_data])
        else:
            train_dset = train_dset_base

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dset,
            eval_dataset=val_dset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=partial(compute_metrics, tokenizer,
                                    metric) if training_args.do_train or training_args.do_eval else None
        )

        # Train BART
        train_result = trainer.train()
        trainer.save_model()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        # Run claim generation on dev set, test set, and scifact dev citances
        dl = trainer.get_test_dataloader(gen_dset)
        generated_claims, fc_claim_inputs = generate_claims(model, gen_dset, dl, tokenizer, device, dm_args.num_beams)


        test_dl = trainer.get_test_dataloader(test_dset)
        test_gen_claims, _ = generate_claims(model, test_dset, test_dl, tokenizer, device, dm_args.num_beams)
        with open(f"{save_dir}/output_test_claims.jsonl", 'wt') as f:
            for c in test_gen_claims:
                f.write(json.dumps(c) + '\n')

        scifact_dev_claims, _ = generate_claims(model, val_dset, trainer.get_test_dataloader(val_dset), tokenizer, device, dm_args.num_beams)
        with open(f"{save_dir}/output_scifact_dev_claims.jsonl", 'wt') as f:
            for c in scifact_dev_claims:
                f.write(json.dumps(c) + '\n')

        model.to('cpu')
        # Run FC model to get scores for generated claims
        fc_model.to(device)
        fc_dev_set = SciFactParagraphBatchDataset(dm_args.external_corpus_file, fc_claim_inputs,
                                                  sep_token=fc_tokenizer.sep_token, k=0, train=False)

        rationale_predictions, stance_preds, stance_scores = predict(fc_model, fc_dev_set, 16, dm_args.fc_model_name, fc_tokenizer, device)
        rationale_json = rationale2json(fc_dev_set.samples, rationale_predictions)
        stance_json = stance2json(fc_dev_set.samples, stance_preds, stance_scores)
        stance_json = post_process_stance(rationale_json, stance_json)
        merged_json = merge_json(rationale_json, stance_json)
        fc_model.to('cpu')

        # Rank predictions and save them
        sorted_fc_claims = sort_fc_claims(merged_json, generated_claims)
        with open(f"{save_dir}/sorted_fc_claims.jsonl", 'wt') as f:
            for c in sorted_fc_claims:
                f.write(json.dumps(c) + '\n')

        # Take all claims with score > 0.5 and add to training data, remove from test data
        if epoch < n_epochs - 1:
            add_to_data = [c for c in sorted_fc_claims if c['score'] > 0.5]
        else:
            add_to_data = sorted_fc_claims

        # Add selected claims to the training set, make the remaining claims the test set
        add_data = defaultdict(list)
        add_data_map = set()
        for claim in add_to_data:
            id_ = claim['id']
            id_ = id_[:id_.rfind('_')]
            add_data_map.add(id_)
            add_data['doc_id'].append(id_)
            add_data['paper_id'].append(claim['paper_id'])
            add_data['context'].append(claim['context'])
            add_data['claims'].append(claim['generated_claim'])
            add_data['evidence'].append(claim['evidence'])
            add_data['orig_context'].append(claim['orig_context'])
            add_data['citance'].append(claim['citance'])
            add_data['score'].append(claim['score'])
            add_data['num_return_sequences'].append(claim['num_return_sequences'])

        remaining = [c for c in sorted_fc_claims if c['score'] <= 0.5]
        remain_data = defaultdict(list)
        remain_ids = set()
        for claim in remaining:
            id_ = claim['id']
            id_ = id_[:id_.rfind('_')]
            if id_ not in add_data_map and id_ not in remain_ids:
                remain_ids.add(id_)
                remain_data['doc_id'].append(id_)
                remain_data['paper_id'].append(claim['paper_id'])
                remain_data['context'].append(claim['context'])
                remain_data['claims'].append("")
                remain_data['evidence'].append(claim['evidence'])
                remain_data['orig_context'].append(claim['orig_context'])
                remain_data['citance'].append(claim['citance'])
                remain_data['num_return_sequences'].append(claim['num_return_sequences'])

        # Create predict dset
        if added_training_data is None:
            added_training_data_base = Dataset.from_dict(add_data)
            added_training_data = added_training_data_base.map(pred_preprocessor, batched=True)
        else:
            new_training_data_base = Dataset.from_dict(add_data)
            added_training_data = concatenate_datasets([added_training_data, new_training_data_base.map(pred_preprocessor, batched=True)])

        gen_dset_base = Dataset.from_dict(remain_data)
        gen_dset = gen_dset_base.map(pred_preprocessor, batched=True)


        csv_out = []
        original_claim_data = []
        prev_id = ''
        j = 0
        # Save added claims to a jsonl for analysis
        with open(f"{save_dir}/added_claims.jsonl", 'wt') as f:
            for id_,paper_id,context,citance,claim,score in zip(added_training_data['doc_id'],
                                                                         added_training_data['paper_id'],
                                                                         added_training_data['orig_context'],
                                                                         added_training_data['citance'],
                                                                         added_training_data['claims'],
                                                                         added_training_data['score']):
                if len(claim.strip()) == 0 or claim.strip() == '.':
                    claim = citance

                original_claim_data.append({
                    'doc_id': id_,
                    'paper_id': paper_id,
                    'context': context,
                    'citance': citance,
                    'claims': [claim],
                    'scores': score
                })
                f.write(json.dumps(original_claim_data[-1]) + '\n')

                csv_out.append([f"{id_}_{j}", context, citance, claim, score])
                j += 1

        # Save the scored added claims for this round of curriculum learning
        csv_pd = pd.DataFrame(csv_out, columns=['ID', 'Context', 'Original Sentence', 'Claim', 'Score'])
        csv_pd.to_csv(f"{save_dir}/ranked_claims.csv", index=None)

        # Generate negative claims
        for claim_set in tqdm(test_gen_claims):
            neg_claims = generate_negative_claims_using_linker([claim_set['generated_claim']], nli, lm, lm_tk, device, 3)
            claim_set['neg_claim'] = neg_claims[0][2] if neg_claims[0] is not None else None

        # Pick 1/3 to be supports, 1/3 to be contradicts, and 1/3 to be NEI
        def incgen():
            val = 0
            while True:
                val += 1
                yield val
        inc = incgen()
        base_claims_and_evidence = []
        for claim_set in test_gen_claims:
            # Remove ID suffix to get original paper ID
            original_doc_id = claim_set['id']
            original_doc_id = original_doc_id[:original_doc_id.rfind('_')]

            pos_claim = claim_set['generated_claim']
            neg_claim = claim_set['neg_claim']
            type = random.randint(0,2)
            if type == 0 or neg_claim == None:
                base_claims_and_evidence.append({
                    'id': next(inc),
                    'claim': pos_claim,
                    'evidence': {str(doc_id): [{'sentences': [0], 'label': 'SUPPORT'}] for doc_id in claim_set['evidence']},
                    'cited_doc_ids': claim_set['evidence']
                })
            elif type == 1:
                base_claims_and_evidence.append({
                    'id': next(inc),
                    'claim': neg_claim,
                    'evidence': {str(doc_id): [{'sentences': [0], 'label': 'CONTRADICT'}] for doc_id in claim_set['evidence']},
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