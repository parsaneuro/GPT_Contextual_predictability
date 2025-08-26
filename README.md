# GPT-2 Predictability for Contextual Word Processing (MATLAB)

## Overview
This repository contains MATLAB code for computing **word predictability from context** using **GPT-2**.  
Given a sequence of words, the model estimates the probability distribution over the next word and maps this distribution into a lexicon for use in **EEG/MEG language processing models** (e.g., surprisal, entropy, mTRF analyses).

This implementation was adapted from a **group project**. To respect project/IP restrictions, only the **core GPT-2 integration code** is provided here; other lab-specific modules have been removed.

---

## Key Features
- **Context encoding**: builds context strings from prior words and encodes them with the GPT-2 tokenizer.
- **Top-p (nucleus) sampling**: implements a cutoff (`p = 0.95`) where tokens are included **up to and including** the first one that pushes cumulative probability ≥ 95%.
- **Lexicon mapping**: probabilities from GPT-2 tokens are mapped onto a user-defined lexicon (`list_lex`) for word-level priors.
- **Parallel priors**: supports both GPT-2 context priors and frequency-based priors for comparison.
- **Integration ready**: outputs surprisal, entropy, and other measures for both segments and words, ready for use in auditory/cognitive neuroscience pipelines.

---

## Requirements
- MATLAB R2021a+ (tested)  
- Access to a GPT-2 model and tokenizer callable from MATLAB (e.g., via ONNX or custom wrapper)  
- Lexicon variables:
  - `list_lex` – cell array of lowercase word strings
  - `lexicon.segments`, `lexicon.phon`, `lexicon.freq` – downstream model requirements

---
