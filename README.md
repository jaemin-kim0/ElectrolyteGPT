# Electrolyte-GPT

**Electrolyte-GPT** is a generative deep learning framework for **molecular design of electrolyte solvents and formulations**, supporting both **unconditioned** and **property-conditioned** generation.

The model is designed to assist research in battery electrolytes by learning chemical representations from curated molecular datasets and generating novel candidate molecules or formulations consistent with desired constraints.

---

## Overview

Electrolyte-GPT enables:

- **Unconditioned molecular generation**
- **Conditional generation** based on target properties
- Exploration of **novel electrolyte solvent chemistries**

The repository contains scripts for training generative models and sampling new molecular structures.

---

## Model Capabilities

- Transformer-based generative architecture  
- SMILES-based molecular representation  
- Compatible with single-molecule and multi-component formulations  

---

## Training

Model training is performed using the provided shell script.
- bash train.sh

All training parameters (dataset paths, batch size, number of epochs, conditioning variables) are defined inside train.sh.

---

## Generation

Molecule and formulation generation is performed using a trained model checkpoint.
- bash generate.sh

All generated results are written to the outputs/ directory.

Output Contents
- SMILES strings of generated molecules
- Target properties
- Validity, uniqueness, novelty
- Generation logs and statistics

The outputs can be further filtered or analyzed for downstream property prediction, simulation, or experimental screening.




