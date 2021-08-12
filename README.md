# Scientific Claim Generation
Repo for scientific claim generation project (Dustin internship Summer 2021)

# Method
We approached claim generation as a curriculum learning problem in two settings:

## Zero-shot
This base method comes from this [paper](https://aclanthology.org/2021.acl-short.61/) on zero-shot fact verification. Claims are generated using a pipeline with the following components:

- Named entity recognition: Extract all of the named entities for a given source sentence
- Question generation: Train a question generation model from [BART](https://arxiv.org/abs/1910.13461) to generate a question given an original sentence and extracted named entity. The model is trained on SQuAD.
- Claim generation: Train a claim generation model from BART to generate a claim given a question and the answer (i.e. the named entity). The model is trained on the [QA2D](https://github.com/kelvinguu/qanli) dataset, which contains questions from SQuAD paired with declarative forms on the questions with their correct answers

## Supervised
In the supervised setting, we train a BART model to directly generate claims from citances using the manually written claims from SciFact (claim generation data can be found [here](https://scifact.s3-us-west-2.amazonaws.com/release/latest/claims_with_citances.jsonl)). We use a very simple input format for claim generation:

`{context} || {citance} || {claim}`

In this case, the context consists of the surounding two sentences from the original citance. During generation, we sample multiple generations from the language model for a given citance. We use half of the number of noun chunks in the original citance as the number of sample generations.

## Curriculum learning
We attempted to improve over the base models by using curriculum learning. We gradually introduce more data to either re-train the NER module (in the zero-shot case) or BART (in the supervised case) by selecting claims which are easily verifiable by an external pre-trained fact checking model. We use the [ParagraphJointModel](https://arxiv.org/abs/2012.14500) pre-trained on FEVER and SciFact to rank claims. Generated claims are paired with the abstracts of the documents which their original citances reference, which gives a set of probabilities $P = {p_0,...,p_N}$ for each generated claim. The score for a given claim is then $s = max_{I}(p_i[SUPPORT] - p_i[CONTRADICT])$. We then rank all claims using this score and select all claims where $s > 0.5$ to re-train the model. For the supervised setting this is straight-forward (pair the claim with the original citance). In the NER case, we take the *entity* from the accepted claims, pair them with their original citances, and re-train an NER model starting from the scispacy `en_core_sci_md` model, which is trained on MedMentions.

# Setup

# Download data

# Dataset creation scripts and tools

# Running claim generation

# Initial results
