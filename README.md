# Scientific Claim Generation
This repository contains data and experiments associated with the ACL 2022 paper "Generating Scientific Claims for Zero-Shot Scientific Fact Checking." We explore methods for generating scientific claims from citation sentences and heuristic assignment of fact checking labels to improve zero-shot scientific fact checking.

## Method
We approached claim generation in two settings:

### Zero-shot
This base method comes from this [paper](https://aclanthology.org/2021.acl-short.61/) on zero-shot fact verification. Claims are generated using a pipeline with the following components:

- Named entity recognition: Extract all of the named entities for a given source sentence
- Question generation: Train a question generation model from [BART](https://arxiv.org/abs/1910.13461) to generate a question given an original sentence and extracted named entity. The model is trained on SQuAD.
- Claim generation: Train a claim generation model from BART to generate a claim given a question and the answer (i.e. the named entity). The model is trained on the [QA2D](https://github.com/kelvinguu/qanli) dataset, which contains questions from SQuAD paired with declarative forms on the questions with their correct answers

### Supervised
In the supervised setting, we train a BART model to directly generate claims from citances using the manually written claims from SciFact (claim generation data can be found [here](https://scifact.s3-us-west-2.amazonaws.com/release/latest/claims_with_citances.jsonl)). We use a very simple input format for claim generation:

`{context} || {citance} || {claim}`

In this case, the context consists of the surounding two sentences from the original citance. During generation, we sample multiple generations from the language model for a given citance. We use half of the number of noun chunks in the original citance as the number of sample generations.

### Curriculum learning
We attempted to improve over the base models by using curriculum learning. We gradually introduce more data to either re-train the NER module (in the zero-shot case) or BART (in the supervised case) by selecting claims which are easily verifiable by an external pre-trained fact checking model. We use the [ParagraphJointModel](https://arxiv.org/abs/2012.14500) pre-trained on FEVER and SciFact to rank claims. Generated claims are paired with the abstracts of the documents which their original citances reference, which gives a set of probabilities $P = {p_0,...,p_N}$ for each generated claim. The score for a given claim is then $s = max_{i}(p_i[SUPPORT] - p_i[CONTRADICT])$. We then rank all claims using this score and select all claims where $s > 0.5$ to re-train the model. For the supervised setting this is straight-forward (pair the claim with the original citance). In the NER case, we take the *entity* from the accepted claims, pair them with their original citances, and re-train an NER model starting from the scispacy `en_core_sci_md` model, which is trained on MedMentions.


### Claim variants
#### SUPPORT
`SUPPORT` claims are created by pairing a generated claim as-is with the abstracts of documents that are cited in the original citance. Since rationales are not labeled in this setup, one should train a model such as LongChecker using these claims (which can perform inference using the entire abstract).

#### CONTRADICT
`CONTRADICT` claims are generated as follows:

- Extract named entities from the original claim and link these to UMLS concepts. Each named entity will be linked to a concept $c_i$
- Get all concepts which belong to the same semantic type and the same relation type. For a concept $c_i$ this yields a set ${r_0^i ... r_K^i} \in R_i$
- Rank all of these concepts by measuring the cosine distance to the original concept using cui2vec vectors ($s_k^i = cosdist(r_k^i, c_i)$)
- Sample the top N concepts from these concepts
- For each concept $r_k^i$, replace the selected named entitiy with either the canonical name or an alias for $r_k^i$ listed in UMLS and select the name which produces the sentence with minimum perplexity using GPT2
- Select as the negation the sentence which produces the strongest contradiction with the original claim as measured by an external NLI model

Negative claims are also paired with the abstract of documents cited in the original citance.

### NOT ENOUGH INFO
`NOT ENOUGH INFO` claims are created by selecting either the original claim or negation and pairing this claim with the abstract of the document where the citance came from.

## Setup
The best way to set up the environment is through [anaconda](https://www.anaconda.com/products/individual). First, install anaconda. Then run the following:

`conda env create -f environment.yml`

You should also download the scispacy `en_core_sci_md` model by running the following:

`pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_md-0.4.0.tar.gz`

Finally, pull the submodules so you have the ParagraphJointModel code:

`git submodule update --recursive --remote`

## Download data and models
1. Download the unlabeled citances and corpora files:

`aws s3 cp --recursive s3://claim-generation-data-dustinw/ claim-generation-data`

This will contain the following files:
- claim-generation-data/citeworth/citeworth\_citances.jsonl: Original unlabeled citances from citeworth
- claim-generation-data/citeworth/external\_corpus.jsonl: The abstracts for all papers references by the citances in citeworth\_citances.jsonl (which are available in S2ORC)
- claim-generation-data/citeworth/internal\_corpus.jsonl: All paragraphs from the documents which the citances in citeworth\_citances.jsonl come from
- claim-generation-data/sampled_citances/: Train, dev, and test splits of sample citances which have already been pre-processed for use with the curriculum learning scripts (these are the samples used in the initial experiments we ran)
- claim-generation-data/models/: Trained models for question generation (question\_gen\_squad) trained on the SQuAD dataset and claim generation (claim\_gen) trained on QA2D

2. Download the original claim generation data from SciFact:

```
cd claim-generation-data
wget https://scifact.s3-us-west-2.amazonaws.com/release/latest/claims_with_citances.jsonl -P scifact_claim_data
```

3. Download the original SciFact data [here](https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz) or by following the instructions in the [SciFact repo](https://github.com/allenai/scifact)

4. Download the checkpoint for the ParagraphJointModel [here](https://drive.google.com/file/d/1hMrQzFe1EaJpCN9s3pF27Wu3amBbekiI/view?usp=sharing) and place it under claim-generation-data/models

5. Download UMLS either [here](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html) or by asking Lucy. You only need the MRCONSO file.

6. Download cui2vec vectors [here](https://figshare.com/s/00d69861786cd0156d81)

## Running the code
To format the SciFact citances to be used as training data for claim generation, run the following (assuming SciFact data is located under `scifact/data`):

```
python dataset_tools/create_claim_generation_dset_from_scifact.py \
  --claims_with_citances claim-generation-data/scifact_claim_data/claims_with_citances.jsonl \
  --scifact_train_split scifact/data/claims_train.jsonl \
  --scifact_dev_split scifact/data/claims_dev.jsonl \
  --output_dir claim-generation-data/scifact_claim_data
```

To create a custom train/dev/test split of unlabeled citances, run the following:

```
python dataset_tools/create_train_test_splits_from_citances.py \
  --citance_file claim-generation-data/citeworth/citeworth_citances.jsonl \
  --external_corpus_file claim-generation-data/citeworth/external_corpus.jsonl \
  --internal_corpus_file claim-generation-data/citeworth/internal_corpus.jsonl \
  --output_dir claim-generation-data/new_sampled_citances
```

Before running either curriculum learning method, edit the paths in `generate_claim_variants.py` at lines 31 and 39 to point to where you downloaded the UMLS MRCONSO file and cui2vec vectors. Then, to run curriculum learning for the zero-shot method, execute the following:

```
python curriculum_learning_ner_pipeline.py \
  --train_citances claim-generation-data/sampled_citances/train.jsonl \
  --test_citances claim-generation-data/sampled_citances/test.jsonl \
  --qg_model_name claim-generation-data/models/question_gen_squad/ \
  --q2c_model_name claim-generation-data/models/claim_gen \
  --fc_model_name roberta-large \
  --fc_model_checkpoint claim-generation-data/models/scifact_roberta_joint_paragraph_dynamic_test_best.model \
  --external_corpus_file claim-generation-data/citeworth/external_corpus.jsonl \
  --internal_corpus_file claim-generation-data/citeworth/internal_corpus.jsonl \
  --output_dir claim-generation-data/output/zero_shot_cl_5_rounds
```

To run curriculum learning for the supervised method, execute the following:

```
python curriculum_learning_scifact_bart_training.py \
  --model_name facebook/bart-base \
  --train_dset claim-generation-data/scifact_claim_data/claim_generation_data_train.jsonl \
  --val_dset claim-generation-data/scifact_claim_data/claim_generation_data_dev.jsonl \
  --predict_dset claim-generation-data/sampled_citances/train.jsonl \
  --test_dset claim-generation-data/sampled_citances/test.jsonl \
  --num_beams 3 \
  --output_dir claim-generation-data/models/curriculum_scifact_model \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 4 \
  --logging_first_step \
  --learning_rate 2e-5 \
  --weight_decay 0.01 \
  --num_train_epochs 3 \
  --warmup_steps 200 \
  --seed 1000 \
  --run_name bart_scifact_curriculum \
  --should_log \
  --save_strategy epoch \
  --eval_accumulation_steps 2 \
  --output_claim_dir claim-generation-data/output/supervised_cl_5_rounds/ \
  --fc_model_name roberta-large \
  --fc_model_checkpoint claim-generation-data/models/scifact_roberta_joint_paragraph_dynamic_test_best.model \
  --external_corpus_file claim-generation-data/citeworth/external_corpus.jsonl \
  --internal_corpus_file claim-generation-data/citeworth/internal_corpus.jsonl
```

Both of these methods will create the following output files ({0-5} indicates that a new set of files is generated for each round of curriculum learning):

```
claim-generation-data/output/{method}/{0-5}
  | added_claims.jsonl: The claims which were added on that round for curriculum learning
  | output_scifact_dev_claims.jsonl: Claims generated from the dev split of the scifact claim generation data
  | output_test_claims.jsonl: Claims generated from the test portion of the unlabeled citances (test_citances and test_dset above)
  | ranked_claims.csv: Claims ranked by their score s, in a csv file for easier viewing
  | scifact_claims.jsonl: Claim file generated from the test citances which can be used as training data for a fact checking model which does not require rationales (i.e. LongChecker)
  | sorted_fc_claims.jsonl: Claims ranked by their score s in their original json format
```

To then sample a set of citances and combine/shuffle claims and prepare a set of sheets for annotation, run the following:

```
python dataset_tools/combine_claims_and_sample_data_for_annotation.py \
  --claims_files claim-generation-data/output/supervised_cl_5_rounds/0/output_test_claims.jsonl claim-generation-data/output/supervised_cl_5_rounds/1/output_test_claims.jsonl ... \
  --shared_size 10 \
  --set_sizes 30 30 30 \
  --annotator_names ann_0 ann_1 ann_2 \
  --outupt_dir claim-generation-data/output/annotations
```

This will combine all of the claims for each citance provided by the list of claim files and randomly sample 30 citances for 3 annotators, as well as 10 citances for all 3 to annotate for calculating IAA (so 100 citances total). After the annotation sheets have been filled out, calculate IAA and get evaluation metrics by running the following:

```
python dataset_tools/get_annotator_agreement_and_evaluate.py --annotation_files claim-generation-data/output/annotations/ann_0.csv claim-generation-data/output/annotations/ann_1.csv claim-generation-data/output/annotations/ann_2.csv
  
```

## Initial results

CSVs with human annotations for the implemented methods are under `claim-generation-data/main-experiment-annotations/{0,1,2}.csv`. These files consist of the following fields:

```
ID: ID of the citance, should be the same for all claims which come from the same citance
Method: The name of the method and which round of curriculum learning the claim was generated from in the format {method}_{cl round} (0 indicates no curriculum learning)
annotator: The annotator ID
Original Sentence: The citance the claim was generated from (only present for the first claim in a set, otherwise blank)
Claim: The generated claim
Fluency: The fluency score
De-Contextualized: De-contextualized rating
Atomicity: Atomicity rating
Context: The surrounding context for the citance (only present for the first claim in a set, otherwise blank)
Original Sentence: Repeated original sentence for the annotators' convenience (only present for the first claim in a set, otherwise blank)
Claim: Repeated claim for the annotators' convenience
Faithfulness: The faithfulness score
Notes: Any notes the annotator took
```

To get IAA, perform evaluation, and generated graphs from this data, execute the following:

```
python dataset_tools/get_annotator_agreement_and_evaluate.py \
  --annotation_files annotation_data/main_experiment_1/0.csv \
                     annotation_data/main_experiment_1/1.csv \
                     annotation_data/main_experiment_1/2.csv \
  --do_eval
```

This produced the following yield:

| Method        | # generated claims | # accepted claims | precision |
|---------------|:------------------:|:-----------------:|:---------:|
|Entity centric |893                 |111                |12.43      |
|BART Generation|156                 |34                 |21.79      |
|  + 1 round CL |147                 |36                 |24.49      |
|  + 2 round CL |145                 |35                 |24.14      |
|  + 3 round CL |146                 |38                 |26.03      |
|  + 4 round CL |143                 |33                 |23.08      |
|  + 5 round CL |148                 |34                 |22.97      |

