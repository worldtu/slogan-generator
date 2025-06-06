## Slogan Generator

Based on dataset and method in the following paper:

> Jin, Yiping, Akshay Bhatia, Dittaya Wanvarie, and Phu TV Le. "Toward Improving Coherence and Diversity of Slogan Generation". Accepted to Natural Language Engineering. Cambridge University Press.

### Project Presentation Slide

- Link: [Project Presentation Slide](https://docs.google.com/presentation/d/1G-S-_JpwF3y__-L6hpzThX44N9kdbaSJzu1x7q2syh8/edit?usp=sharing)

### 7 Models Included

1. `scratch_rnn`: Using pretrained BART tokenizer, and build a encoder-decoder RNN model from scratch.
2. `scratch_lstm`: Using pretrained BART tokenizer, and build a encoder-decoder LSTM model from scratch.
3. `scratch_decoder_bpe`: Using BPE customized tokenzier, and build a decoder-only model from scratch.
4. `scratch_decoder`: Using pretrained BART tokenizer, and build a decoder-only transformer model from scratch.
5. `scratch_encoder_decoder`: Using pretrained BART tokenizer, and build a encoder-decoder transformer model from scratch.
6. `finetune_bart_lora`: Using pretrained BART tokenizer, and use LoRA to finetune the DistilBART model (https://huggingface.co/sshleifer/distilbart-cnn-6-6).
7. `finetune_bart_all`: Using pretrained BART tokenizer, and directly finetune all the parameters of DistilBART.

### Metrics for Evaluation

Refering to the original paper, the models are using ROUGE for evaluating the quality of the slogan generated.

### Loss Plot for Fine-tuning

![Fine-tuning loss curve](./Fine-Tuning%20Loss.png)
