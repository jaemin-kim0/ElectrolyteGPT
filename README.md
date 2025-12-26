# Electrolyte-GPT

**Electrolyte-GPT** is a generative deep learning framework for **molecular design of electrolyte solvents and formulations**, supporting both **unconditioned** and **property-conditioned** generation.

The model is designed to assist research in battery electrolytes by learning chemical representations from curated molecular datasets and generating novel candidate molecules or formulations consistent with desired constraints.

---

## Overview

Electrolyte-GPT enables:

- **Unconditioned molecular generation**
- **Conditional generation** based on target properties or formulation constraints
- Exploration of **novel electrolyte solvent chemistries**
- Scalable training and sampling workflows

The repository contains scripts for training generative models and sampling new molecular structures.

---

## Model Capabilities

- Transformer-based generative architecture  
- SMILES-based molecular representation  
- Conditional control via property embeddings or tokens  
- Compatible with single-molecule and multi-component formulations  

---

## Training

Model training is performed using the provided shell script.
- bash train.sh

All training parameters (dataset paths, batch size, learning rate, number of epochs, conditioning variables, and checkpoint settings) are defined inside train.sh or its associated configuration files.

---

## Generation

Molecule and formulation generation is performed using a trained model checkpoint.
- bash generate.sh

All generated results are written to the outputs/ directory.

Output Contents
- SMILES strings of generated molecules
- Multi-component electrolyte formulations
- Conditioning metadata (e.g., target properties)
- Generation logs and statistics

The outputs can be further filtered, analyzed, or used for downstream property prediction, simulation, or experimental screening.




