## Slogan Generator

Based on dataset and method in the following paper:

> Jin, Yiping, Akshay Bhatia, Dittaya Wanvarie, and Phu TV Le. "Toward Improving Coherence and Diversity of Slogan Generation". Accepted to Natural Language Engineering. Cambridge University Press.

### Three Models Included

1. `scratch_model`: Using pretrained BART tokenizer, and build a decoder-only model from scratch.
2. `scratch_bpe`: Using BPE customized tokenzier, and build a decoder-only model from scratch.
3. `finetune_bart`: Using pretrained BART tokenizer, and finetune a model based on DistilBART (https://huggingface.co/sshleifer/distilbart-cnn-6-6)

### Metrics for Evaluation

Refering to the original paper, the models use ROUGE for evaluating the quality of the slogan generated.
