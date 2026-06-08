# -*- coding: utf-8 -*-
"""GAS: Generative Aspect Sentiment extraction.

T5-based generative model that jointly extracts aspect terms and predicts
sentiment polarity in one decoder pass.  Category is predicted separately
by the BERT multi-task APC model.

Modules
-------
dataset         — data loading & target formatting for T5 GAS training
model           — GasT5Model wrapper with "(aspect, sentiment)" output parsing
trainer         — training loop with GAS-aware evaluation and checkpointing
metrics         — 4-label evaluation (aspect_term / sentiment / category / joint)
train_gas       — CLI entry point for training the GAS model
evaluate_joint  — end-to-end 3-label evaluation combining GAS + BERT APC
"""
