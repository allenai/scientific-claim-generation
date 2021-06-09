import argparse
import numpy as np
import logging
import random
import torch
import sys
from functools import partial
from dataclasses import dataclass, field
from typing import Optional, Union
from tqdm import tqdm
import json


from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    HfArgumentParser,
    set_seed,
    PreTrainedTokenizerBase,
    PreTrainedModel,
    DataCollatorForSeq2Seq
)
from transformers.file_utils import PaddingStrategy

from custom_trainer import CustomTrainer


logger = logging.getLogger(__name__)


@dataclass
class DataAndModelArguments:
    """
    Arguments for setting up training/evaluation
    """
    model_name: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
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
    output_predictions: Optional[str] = field(
        default=None, metadata={"help": "The output file to store predictions"}
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
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
    if dset == 'citeworth':
        inputs = [ctx + ' ' + ans for ctx, ans in zip(examples['generated_question'], examples['answer'])]
        targets = [''] * range(len(inputs))
    else:
        inputs = [q + ' ' + a for q,a in zip(examples['question'], examples['answer'])]
        targets = [a for a in examples['turker_answer']]
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
    return model_inputs

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



if __name__ == '__main__':
    parser = HfArgumentParser((DataAndModelArguments, Seq2SeqTrainingArguments))
    dm_args, training_args = parser.parse_args_into_dataclasses()

    seed = training_args.seed
    model_name = dm_args.model_name
    n_gpu = training_args.n_gpu
    should_log = dm_args.should_log
    train_dset_name = dm_args.train_dset
    val_dset_name = dm_args.val_dset
    gen_dset_name = dm_args.predict_dset
    enforce_reproducibility(seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if should_log else logging.WARN)

    # See if CUDA available
    device = torch.device("cpu")
    if torch.cuda.is_available():
        print("Training on GPU")
        device = torch.device("cuda:0")

    # Set the verbosity to info of the Transformers logger (on main process only):
    if should_log:
        transformers.utils.logging.set_verbosity_info()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Create the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def qa2d_filter(example):
        return example['dataset'] == 'SQuAD' \
               and example['question'] != None \
               and example['answer'] != None \
               and example['turker_answer'] != None

    train_dset = None
    if training_args.do_train:
        # Create dataset and processor
        preprocessor = partial(data_preprocess, tokenizer, train_dset_name)
        train_dset_base = load_dataset('csv', data_files=[train_dset_name], delimiter='\t')
        train_dset_base = train_dset_base.filter(qa2d_filter)
        train_dset = train_dset_base.map(preprocessor, batched=True)

    val_dset = None
    if training_args.do_eval:
        preprocessor = partial(data_preprocess, tokenizer, val_dset_name)
        val_dset_base = load_dataset('csv', data_files=[val_dset_name], delimiter='\t')
        val_dset_base = val_dset_base.filter(qa2d_filter)
        val_dset = val_dset_base.map(preprocessor, batched=True)

    pred_dset = None
    if training_args.do_predict:
        preprocessor = partial(data_preprocess, tokenizer, 'citeworth')
        gen_dset_base = load_dataset('json', data_files=gen_dset_name)
        # Filter missing NER
        gen_dset_base = gen_dset_base.filter(lambda example: len(example['answers']) > 0)
        gen_dset = gen_dset_base.map(preprocessor, batched=True)['train']

    metric = load_metric('rouge')

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding='longest'
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dset,
        eval_dataset=val_dset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=partial(compute_metrics, tokenizer, metric) if training_args.do_train or training_args.do_eval else None
    )

    if training_args.do_train:
        # Train
        train_result = trainer.train()
        trainer.save_model()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        # Validation
        eval_metrics = trainer.evaluate(
            max_length=tokenizer.model_max_length,
            num_beams=dm_args.num_beams,
            metric_key_prefix='eval'
        )
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        dl = trainer.get_test_dataloader(gen_dset)
        all_samples = []
        for b in tqdm(dl):
            input_ids = b['input_ids'].to(device)
            samples = model.generate(
                input_ids,
                num_beams=dm_args.num_beams,
                max_length=tokenizer.model_max_length,
                early_stopping=True
            )
            all_samples.extend(list(samples.detach().cpu().numpy()))
        print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in all_samples])
        with open(dm_args.output_predictions, 'wt') as f:
            for con,ans,q,claim in zip(gen_dset['context'], gen_dset['answer'], gen_dset['generated_question'], all_samples):
                gen_claim = tokenizer.decode(q, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                f.write(json.dumps({'context': con, 'answer': ans, 'generated_question': q, 'generated_claim': gen_claim}))


