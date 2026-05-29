# Infectious-Disease-Adjuvant-LLM-Fine-tuning

This repository contains the code used to build the pipeline for "Evidence-supported Extraction of Vaccine Adjuvants from Infectious Disease Literature Using Fine-tuned Large Language Models"

## Abstract

Vaccine adjuvants are essential for enhancing immune responses, yet their representation in infectious disease literature is heterogeneous and fragmented across synonyms, abbreviations, and experimental formulations, limiting scalable curation of vaccine knowledge. We present an evidence-supported information extraction framework for identifying vaccine adjuvants from titles and abstracts of infectious disease publications. A curated corpus was constructed from the VIOLIN Vaxjo vaccine adjuvant resource by aligning PubMed abstracts with adjuvant–evidence pairs using the Vaccine Ontology-based normalization and lexicon matching. The final dataset comprised 298 abstracts with sentence-level evidence annotations 46 and fixed train, validation, and test splits to ensure reproducibility.

Instruction-tuned language models were fine-tuned to output structured JSON containing adjuvant names and supporting evidence snippets using parameter-efficient instruction tuning. Four instruction-tuned models spanning 4B to 70B parameters (Gemma 3 4B, Qwen 2.5 7B, Mistral 7B v0.3, and Llama 3.3 70B) were evaluated on a held-out test set (29 abstracts with 41 adjuvants) using micro-precision, recall, and F1 metrics (Table 1). The evidence requirement enforced grounding by requiring a supporting snippet from the abstract, constraining predictions to text-supported mentions.

Performance generally increased with model size, with Gemma 3 4B showing limited extraction performance (F1 = 50.0%). Notably, among compact models, Mistral 7B achieved the strongest performance (F1 = 76.9%), outperforming Qwen 2.5 7B (F1 = 64.0%) and slightly exceeding the 70B-parameter Llama 3.3 model (F1 = 73.7%). Analysis of the highest-performing model showed that the most frequently extracted adjuvants, including interferon-γ, cholera toxin and its B subunit, and aluminum-based formulations, closely matched the distribution observed in gold-standard annotations. Extracted evidence snippets predominantly described antibody induction, cytokine responses, mucosal immunity, and protection against challenge, showing strong alignment with curated evidence. Together, these results demonstrate that the framework enables scalable, evidence-grounded curation of adjuvant use and reported immune outcomes across infectious disease literature, and that well-optimized mid-scale language models can match or outperform substantially larger models.

## Corpus construction and normalization
- `1.1. Analyze VIOLIN database_with LEO.ipynb`
- `1.2. Replace References both with Ref ID and Name.ipynb`
- `2.1. violin_schema_selection.ipynb`
- `2.2. violin_adjuvant_kb.ipynb`
- `2.3. Adjuvant NER Lexicon Construction.ipynb`
- `2.4. Download Pubmed Abstracts.ipynb`
- `3.1 dictionary_exact_matching.ipynb`
- `3.2. Visual Validation.ipynb`
- `3.3. Fuzzy matching.ipynb`
- `3.4. Fuzzy Visual Validation.ipynb`
- `4.1 LLM Corpus.ipynb`

## Fine-tuning and sweep execution
- `Hyperparameter_tuning_LLM_V2.py`
- `Hyperparameter_tuning_gemma-3-4b-it_V2.sh`
- `Hyperparameter_tuning_Qwen2.5-7B-Instruct_Instruct_V2.sh`
- `Hyperparameter_tuning_Mistral-7B-Instruct-v0.3_V2.sh`
- `Hyperparameter_tuning_Llama_3.3_70B_Instruct_V2.sh`


## Environment setup

1. Create and activate a Python environment (Python 3.9 recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## External requirements

- PubMed access via Entrez API:
  - `ENTREZ_EMAIL`
  - `ENTREZ_API_KEY`
- GPU environment for model fine-tuning (multi-GPU recommended for larger models).
- Access to base model checkpoints referenced in sweep scripts.

## Expected data layout

The notebooks/scripts expect the following directory structure:

```text
Dataset/VIOLIN_12-10-2025/
  raw/
  interim/
    exact_matches/
    fuzzy_matches/
  final/
```

Key files generated/used across the workflow include:
- `Dataset/VIOLIN_12-10-2025/interim/pubmed_abstracts.jsonl`
- `Dataset/VIOLIN_12-10-2025/final/llm_adjuvant_evidence_corpus.jsonl`
- `Dataset/VIOLIN_12-10-2025/final/llm_adjuvant_evidence_corpus_finetune.jsonl`
- `Dataset/VIOLIN_12-10-2025/final/splits_fixed.json`

## Reproducibility order

Run in this sequence:

1. Corpus construction notebooks (`1.x` -> `4.1`)
2. Fine-tuning sweeps (shell scripts + `Hyperparameter_tuning_LLM_V2.py`)


## Hugging Face model repositories

Fine-tuned merged checkpoints are available on Hugging Face:

- Qwen 2.5 7B: https://huggingface.co/RehanaHasin/qwen2.5-7b-instruct-adjuvant-extractor
- Mistral 7B Instruct v0.3: https://huggingface.co/RehanaHasin/mistral-7b-instruct-v0.3-adjuvant-extractor
- Llama 3.3 70B Instruct: https://huggingface.co/RehanaHasin/llama-3.3-70b-instruct-adjuvant-extractor

Each model card includes:
- exact inference prompt format
- validated inference code
- base model metadata
