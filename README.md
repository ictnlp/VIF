# From Attenuation to Attention: Variational Information Flow Manipulation for Fine-Grained Visual Perception.
[![arXiv](https://img.shields.io/badge/arXiv-2604.12508-b31b1b)](https://arxiv.org/abs/2604.12508)


## Overview

Multimodal large language models are strong at general visual understanding, but they often fail on fine-grained perception tasks that require identifying tiny objects or subtle visual relationships. We attribute this limitation to **Visual Attenuation**: sparse visual signals are progressively suppressed by dominant textual tokens during network propagation, so deep layers not only attend less to visual evidence, but also lose spatial focus and drift toward diffuse, text-dominated attention patterns. VIF addresses this problem with a conditional variational formulation that aligns an answer-aware posterior `q(z|I,Q,A)` and a question-conditioned prior `p(z|I,Q)`, decodes the latent representation into a sparse spatial Gaussian mixture, and restores deep-layer visual information flow by injecting the learned visual bias into selected layers. Experiments on general VQA, fine-grained perception, and visual grounding show that VIF improves fine-grained reasoning while preserving general multimodal capability.

![VIF framework](images/framework.png)

## Highlights

- **Visual Attenuation analysis**: VIF is built around the finding that deep layers in MLLMs both reduce visual attention strength and lose spatial focus.
- **Variational visual saliency modeling**: VIF learns an answer-aware posterior and a question-only prior to infer response-relevant visual cues.
- **Spatial GMM decoding**: latent slots are decoded into a sparse visual importance distribution for fine-grained attention restoration.
- **Deep-layer information-flow restoration**: the learned visual bias is injected into selected high layers.

## Quick Start

### Install

```bash
conda create -n llava-vif python=3.10 -y
conda activate llava-vif

pip install -r requirements.txt
```

### Training

Run:

```bash
bash finetune_7b.sh
```

Key VIF knobs:
- Enable: `--use_latent_importance True`
- Learning range: `--latent_learning_start` / `--latent_learning_end`
- Injection range: `--latent_apply_start` / `--latent_apply_end`
- Loss weights: `--latent_kl_weight` / `--latent_sparsity_weight`
- Distribution config: `--latent_num_components`

### Testing / Inference

```bash
python -m llavavif.eval.run_llava \
  --model-path checkpoints/<SAVE_PATH> \
  --image-file /path/to/image.jpg \
  --query "Describe the most important visual details and answer: ..."
```

### Repository Structure

- `llavavif/model/latent_importance.py`: variational prior/posterior, GMM decoder,
  sparsity terms, and bias construction
- `llavavif/model/attention_intervention.py`: selective deep-layer attention
  injection


## Citation

If you find this work useful in your research, please consider cite:
```bibtex
@misc{zhu2026vif,
  title        = {From Attenuation to Attention: Variational Information Flow Manipulation for Fine-Grained Visual Perception},
  author       = {Jilong Zhu and Yang Feng},
  year         = {2026},
  eprint       = {2604.12508},
  archivePrefix= {arXiv},
  primaryClass = {cs.CV},
  doi          = {10.48550/arXiv.2604.12508}
}
```

## License

[Apache License 2.0](LICENSE)

## Acknowledgements

This project is built on top of the open-source implementations of [LLaVA](https://github.com/haotian-liu/llava), [Open-LLaVA-NeXT](https://github.com/xiaoachen98/Open-LLaVA-NeXT) and [lmms-eval](https://github.com/evolvinglmms-lab/lmms-eval). Thanks to the authors and the community.

