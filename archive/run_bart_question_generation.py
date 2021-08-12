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
from pathlib import Path


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


@dataclass
class DataCollatorForSeq2SeqMinPadding:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`

            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                feature["labels"] = (
                    feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                )

        max_length = max(len(feature['input_ids']) for feature in features)
        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features


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
    if dset == 'local':
        inputs = [ctx + ' ' + ans[0]['text'] for ctx, ans in zip(examples['context'], examples['answers'])]
    else:
        inputs = [ctx + ' ' + ans['text'][0] for ctx, ans in zip(examples['context'], examples['answers'])]
    targets = [q for i,q in enumerate(examples['question'])]
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

    train_dset = None
    if training_args.do_train:
        # Create dataset and processor
        if Path(train_dset_name).exists():
            preprocessor = partial(data_preprocess, tokenizer, 'local')
            # Load from local file
            train_dset_base = load_dataset('json', data_files=[train_dset_name])['train']
        else:
            preprocessor = partial(data_preprocess, tokenizer, train_dset_name)
            train_dset_base = load_dataset(train_dset_name, split='train')
        train_dset = train_dset_base.map(preprocessor, batched=True)

    val_dset = None
    if training_args.do_eval:
        if Path(val_dset_name).exists():
            preprocessor = partial(data_preprocess, tokenizer, 'local')
            val_dset_base = load_dataset('json', data_files=[val_dset_name])['train']
        else:
            preprocessor = partial(data_preprocess, tokenizer, val_dset_name)
            val_dset_base = load_dataset(val_dset_name, split='validation')
        val_dset = val_dset_base.map(preprocessor, batched=True)

    pred_dset = None
    if training_args.do_predict:
        preprocessor = partial(data_preprocess, tokenizer, 'local')
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
            for id,con,ans,q,citance,paper_id,evidence in zip(gen_dset['id'],gen_dset['context'], gen_dset['answers'], all_samples,gen_dset['citance'],gen_dset['paper_id'], gen_dset['evidence']):
                gen_question = tokenizer.decode(q, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                f.write(json.dumps({'id': id, 'paper_id': paper_id, 'context': con, 'answer': ans[0], 'generated_question': gen_question, 'citance': citance, 'evidence': evidence}) + '\n')


