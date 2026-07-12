<h1 style="text-align: left;">SADE</h1>

This repository contains the code for reproducing results from the paper:

**Boosting Adversarial Transferability via Similarity-Guided Augmentation and Dynamic Ensembles**

## Requirements

- Python >= 3.8.1
- numpy == 1.19.2
- tensorflow == 2.5.0
- scikit-image == 0.18.0
- opencv-python >= 4.10.0
- scipy == 1.6.3
- pandas == 1.2.4
- imageio >= 2.33.1
- tf-slim == 1.1.0
- tensorflow-addons == 0.14.0

## Quick Start

### Prepare data and checkpoints

Download the [data](https://drive.google.com/drive/folders/1CfobY6i8BfqfWPHL31FKFDipNjqWwAhS) and [checkpoints](https://drive.google.com/drive/folders/10cFNVEhLpCatwECA6SPB-2g0q5zZyfaw), then place them in `dev_data/` and `models/`, respectively.

### Generate adversarial examples

`sade.py` generates adversarial examples on `inception_v3` by default.

If you want to attack another source model, replace the model used in `graph()` and load the corresponding checkpoint in `main()`.

```bash
CUDA_VISIBLE_DEVICES=0 python sade.py
```

### Evaluate transferability

Generated adversarial examples are saved to `./outputs_SADE`. Run `simple_eval.py` to evaluate transfer success on the models used in the paper:

```bash
CUDA_VISIBLE_DEVICES=0 python simple_eval.py
```

## Defense References

The paper also evaluates transferability against five defense-oriented settings. Useful GitHub references:

- HGD: [GitHub search for High-Level Representation Guided Denoiser](https://github.com/search?q=High-Level+Representation+Guided+Denoiser&type=repositories)
- R&P: [cihangxie/NIPS2017_adv_challenge_defense](https://github.com/cihangxie/NIPS2017_adv_challenge_defense)
- NIPS-r3: [GitHub search for NIPS-r3 defense](https://github.com/search?q=NIPS-r3+adversarial+defense&type=repositories)
- NPR / NRP: [Muzammal-Naseer/NRP](https://github.com/Muzammal-Naseer/NRP)
- FD: [GitHub search for Feature Distillation defense](https://github.com/search?q=Feature+Distillation+adversarial+defense&type=repositories)

## Multimodal Extension

SADE is currently implemented for image-based transfer attacks. A natural multimodal extension can be built by following the augmentation-and-guidance framework in SGA:

- SGA paper: [Set-level Guidance Attack: Boosting Adversarial Transferability of Vision-Language Pre-training Models](https://arxiv.org/abs/2307.14061)

In a multimodal setting, we follow SGA's set-level cross-modal guidance idea, but replace its image-side augmentation pipeline with the similarity-guided image augmentation used in SADE. In other words, the text-side cross-modal supervision can remain SGA-style, while the image-side augmentation is swapped to our augmentation strategy.

## Acknowledgments

This codebase is based in part on [SI-NI-FGSM](https://github.com/JHL-HUST/SI-NI-FGSM).
