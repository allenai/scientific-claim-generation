from transformers import pipeline, FillMaskPipeline, TextClassificationPipeline, PreTrainedTokenizer
from transformers import RobertaTokenizerFast, RobertaForMaskedLM, GPT2LMHeadModel
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import spacy
import scispacy
from scispacy.linking import EntityLinker
from collections import defaultdict
import random
random.seed(1000)
from string import punctuation
from scipy.spatial.distance import cosine
from typing import List, AnyStr


# Load the entity linker
nlp = spacy.load('en_core_sci_md')
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
linker = nlp.get_pipe("scispacy_linker")
tui_to_cui = defaultdict(list)
for cui in linker.kb.cui_to_entity:
    for t in linker.kb.cui_to_entity[cui].types:
        tui_to_cui[t].append(cui)

# Load relation types
cui_to_rel = {}
rel_to_cui = defaultdict(set)
with open('UMLS/MRCONSO.TXT') as f:
    for l in f:
        fields = l.strip().split('|')
        cui_to_rel[fields[0]] = fields[12]
        rel_to_cui[fields[12]].add(fields[0])

# Load concept vectors
cui2vec = {}
with open('cui2vec/cui2vec_pretrained.csv') as f:
    next(f)
    for l in f:
        fields = l.strip().split(',')
        cui2vec[fields[0][1:-1]] = np.array(fields[1:], dtype=np.float32)

def get_perplexity(
        sentences: List[AnyStr],
        lm: GPT2LMHeadModel,
        lm_tokenizer: PreTrainedTokenizer,
        device: torch.device) -> List[float]:
    """
    Get the perplexity for a set of sentences using a given language model
    :param sentences:
    :param lm:
    :param lm_tokenizer:
    :param device:
    :return:
    """
    lm.eval()
    lm.to(device)
    ppl = []
    for sent in sentences:
        inputs = lm_tokenizer.batch_encode_plus([sent])
        with torch.no_grad():
            loss = lm(torch.tensor(inputs['input_ids']).to(device), labels=torch.tensor(inputs['input_ids']).to(device))[0]
            ppl.append(np.exp(loss.cpu().item()))
    lm.to('cpu')
    return ppl


def kbin(
        claims: List[str],
        nli: TextClassificationPipeline,
        language_model: GPT2LMHeadModel,
        lm_tokenizer: PreTrainedTokenizer,
        device: torch.device,
        n_samples: int = 3) -> List[List]:
    """
    Generate negative claims by extracting named entities, linking them to UMLS, gathering all related concepts to the entity, ranking the
    concepts by measuring their cosine distance from the concept for the extracted named entity using cui2vec vectors, selecting
    the text form of those concepts to use by replacing them in the original text and ranking them based on perplexity,
    then sampling from the top concepts and selecting the replacement which has the strongest contradiction with the original claim
    using an external NLI model
    :param claims:
    :param nli:
    :param language_model:
    :param lm_tokenizer:
    :param device:
    :param n_samples:
    :return:
    """
    curr_claims = []
    data = []
    for claim in claims:
        suggs = []
        for ent in nlp(claim).ents:
            if len(ent._.kb_ents) > 0:
                cui = ent._.kb_ents[0][0]
                if cui not in cui2vec:
                    continue

                tui = linker.kb.cui_to_entity[cui].types[0]
                alias_options = []
                cui_options = list(set(tui_to_cui[tui]) - set([cui]))
                if cui in cui_to_rel:
                    cui_options = list(set(cui_options) & rel_to_cui[cui_to_rel[cui]])
                    cui_options = list(set(cui_options) & set(cui2vec.keys()))
                # Calculate distance
                dist = [cosine(cui2vec[cui], cui2vec[opt]) for opt in cui_options]
                cui_options = [cui_options[idx] for idx in np.argsort(dist)]

                j = 0
                while len(alias_options) < n_samples and j < len(cui_options):
                    aliases_curr = [alias.lower() for alias in linker.kb.cui_to_entity[cui_options[j]].aliases + [linker.kb.cui_to_entity[cui_options[j]].canonical_name] if len(alias) < (len(ent.text) * 2) and not any(p in alias for p in punctuation)]

                    if len(aliases_curr) > 0:
                        # Rank by perplexity
                        sents_curr = [[claim.replace(ent.text, alias), (ent.text, alias)] for alias in aliases_curr]
                        ppl = get_perplexity([s[0] for s in sents_curr], language_model, lm_tokenizer, device)
                        alias_options.append(sents_curr[np.argmin(ppl)])
                    j += 1
                suggs.extend(alias_options)

        for sug in suggs:
            if sug[0] != claim.lower():
                score = nli(f"{claim}</s></s>{sug[0]}")[0][0]['score']
                curr_claims.append([claim, sug[1], sug[0], score])
        if len(curr_claims) > 0:
            top_claim = list(sorted(curr_claims, key=lambda x: x[-1], reverse=True))[0]
        else:
            top_claim = None
        data.append(top_claim)

    return data
