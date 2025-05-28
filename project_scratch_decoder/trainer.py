import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
from rouge_score import rouge_scorer
from project_scratch_decoder.infer import SloganGenerator
from transformers import get_linear_schedule_with_warmup
import torch.optim.lr_scheduler as sched
import logging

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Encapsulates training loop for the DecoderOnlyTransformer.
    """
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        dataset: Dataset,
        batch_size: int = 32,
        lr: float = 3e-4,
        device: str = "cpu",
        val_dataset: Dataset = None,
        val_batch_size: int = None,
        model_save_path: str = "./models_scratch/decoder_only_model.pt",
        epochs: int = 5,
        # entropy_weight: float = 0.1  # Added hyperparameter for entropy penalty
        entropy_weight: float = None  # Added hyperparameter for entropy penalty
    ):
        self.device = device
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.epochs = epochs
        self.dataset = dataset
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=dataset.collate_fn
        )
        self.model_save_path = model_save_path # Store the save path
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        
        # Penalize the model for generating common words
        weights = torch.ones(tokenizer.vocab_size, device=self.device)
        common_words = ["<eos>", "and", "the", "of", "to", "a", "in", "for", "on", "with",
                        "is","it","that","at","by","from","this", "be","as","are","or","an","was"]
        for w in common_words:
            tid = tokenizer.convert_tokens_to_ids(w)
            if tid != tokenizer.unk_token_id:
                weights[tid] = 0.001  # adjust this factor as needed
        weights[tokenizer.eos_token_id] = 0.0001
        self.entropy_weight = entropy_weight

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100, weight=weights, 
                                             label_smoothing=0.3)
        
        # Validation dataset and loader
        self.val_dataset = val_dataset
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=val_batch_size or batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=val_dataset.collate_fn
            )
        
        # # RL training dataset and loader
        # self.train_dataset_rl = train_dataset_rl
        # if train_dataset_rl is not None:
        #     self.train_dataloader_rl = DataLoader(
        #         train_dataset_rl,
        #         batch_size=batch_size, # Can be different from supervised batch_size
        #         shuffle=True,
        #         num_workers=0,
        #         collate_fn=train_dataset_rl.collate_fn_rl # Assuming a different collate_fn for RL
        #     )
            
        # Add learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optim, mode='min', factor=0.5, patience=3, min_lr=1e-5)
        
        # Warm-up scheduler (step every batch)
        total_steps  = len(self.loader) * self.epochs
        warmup_steps = int(0.1 * total_steps)
        self.warmup_scheduler = get_linear_schedule_with_warmup(
            self.optim,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

    def validate(self):
        """Run validation and return average loss."""
        if self.val_dataset is None:
            return None
            
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels, prompt_lengths in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                prompt_lengths = prompt_lengths.to(self.device)

                logits = self.model(inputs)
                
                T, B, V = logits.shape
                loss_labels = labels.clone()
                for i in range(B): # Iterate over batch
                    loss_labels[:prompt_lengths[i], i] = self.criterion.ignore_index # Apply masking
                loss = self.criterion(logits.view(T*B, V), loss_labels.view(T*B))
                total_loss += loss.item()
                
        avg_loss = total_loss / len(self.val_loader)
        self.model.train()  # Set back to training mode
        return avg_loss

    def train(self, patience: int = 3, min_delta: float = 0.0):
        """
        Train the model with early stopping.
        
        Args:
            epochs: Maximum number of epochs to train for
            patience: Number of epochs to wait for improvement before stopping
            min_delta: Minimum change in validation loss to qualify as improvement
        """
        self.model.train()
        best_loss = float('inf')
        no_improve_count = 0

        for epoch in range(self.epochs):
            total_loss = 0.0
            progress_bar = tqdm(self.loader, desc=f"Epoch {epoch+1}/{self.epochs}")

            for inputs, labels, prompt_lengths in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                prompt_lengths = prompt_lengths.to(self.device)
                
                # for b in range(2):  # look at first two examples
                #     label_ids = labels[:, b].tolist()
                #     input_ids = inputs[:, b].tolist()
                #     decoded = self.tokenizer.decode(label_ids, skip_special_tokens=False)
                #     print(f"Example {b}:")
                #     print("  Raw IDs:   ", label_ids)
                #     print("  Decoded:   ", decoded.replace(' ', '_'))
                #     print("  Prompt len:", prompt_lengths[b].item())

                self.optim.zero_grad()
                logits = self.model(inputs)

                T, B, V = logits.shape
                loss_labels = labels.clone()
                for i in range(B): # Iterate over batch
                    loss_labels[:prompt_lengths[i], i] = self.criterion.ignore_index
                logits2d = logits.view(T * B, V)
                loss_labels1d = loss_labels.view(T * B)
                # logits2d = logits2d.to(self.device)
                # loss_labels1d = loss_labels1d.to(self.device)

                if self.entropy_weight is not None:
                    ce_loss = self.criterion(logits2d, loss_labels1d)
                    probs = torch.softmax(logits2d, dim=-1)
                    # Add a small epsilon to log to prevent log(0) = NaN
                    entropy_per_token = -(probs * torch.log(probs + 1e-9)) # [T*B, V]
                    entropy = entropy_per_token.sum(dim=-1).mean()        # scalar, mean 
                    loss = ce_loss - self.entropy_weight * entropy
                else:
                    loss = self.criterion(logits2d, loss_labels1d)
                
                loss.backward()
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optim.step()
                
                # Step the warm-up scheduler per batch
                self.warmup_scheduler.step()

                total_loss += loss.item()
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            train_avg_loss = total_loss / len(self.loader)
            # Validate if we have a validation set
            val_loss = self.validate()
            # Use validation loss if available, otherwise use training loss
            current_loss = val_loss if val_loss is not None else train_avg_loss
            log_msg = f"Epoch {epoch+1}/{self.epochs}  train_loss={train_avg_loss:.4f}"
            if val_loss is not None:
                log_msg += f"  val_loss={val_loss:.4f}"
            logger.info(log_msg)
            
            # Update learning rate based on loss
            self.scheduler.step(current_loss)

            # Early stopping
            if current_loss < best_loss - min_delta:
                best_loss = current_loss
                no_improve_count = 0
                # Save model if it's the best so far
                os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
                logger.info(f"Saving model to {self.model_save_path}")
                torch.save(self.model.state_dict(), self.model_save_path)
                logger.info(f"Model saved with loss: {best_loss:.4f}")
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    logger.info(f"Early stopping triggered after {patience} epochs without improvement.")
                    break

        logger.info("Training completed!")
        
    # def train_rl(self, patience: int = 3, min_delta: float = 0.0,
    #              beam_width: int = 5):
    #     """
    #     Fine‐tuning with ROUGE‐L reward.
    #     """
    #     self.model.train()
    #     scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    #     generator = SloganGenerator(self.model, self.tokenizer, self.device)

    #     best_loss = float('inf')
    #     no_improve_count = 0

    #     for epoch in range(epochs):
    #         total_loss = 0.0
    #         progress_bar = tqdm(self.loader, desc=f"Epoch {epoch+1}/{epochs}")

    #         for inputs, labels, prompt_lengths in progress_bar:
    #             inputs, labels = inputs.to(self.device), labels.to(self.device)
    #             prompt_lengths = prompt_lengths.to(self.device)


    #     # Encode all prompts to tensor once
    #     input_ids = [torch.tensor(self.tokenizer.encode(p)) for p in prompts]
    #     input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=False,
    #                 padding_value=self.tokenizer.pad_token_id).to(self.device)

    #     # Forward to get log‐probs
    #     logits = self.model(input_ids)         # [T, B, V]
    #     logp = torch.log_softmax(logits, -1) # same shape

    #     # 1) Generate beam outputs
    #     batch_seqs = generator.generate_beam_batch(
    #         prompts, max_len=50, beam_width=beam_width)  # returns List[List[token_id]]

    #     # 2) Collect log‐probs of generated tokens
    #     #    Align `logp` and `batch_seqs` so you can sum log‐probs
    #     #    (skipping BOS/EOS as needed)
    #     all_logp = []
    #     for b, seq in enumerate(batch_seqs):
    #         # turn seq into tensor of shape [L]
    #         seq_tensor = torch.tensor(seq, device=self.device)
    #         # gather log‐probs step by step:
    #         lp = logp[:, b, :].gather(1, seq_tensor.unsqueeze(1))
    #         all_logp.append(lp.sum())
    #     mean_logp = torch.stack(all_logp).mean()

    #     # 3) Compute average ROUGE‐L reward
    #     rewards = []
    #     for seq, ref in zip(batch_seqs, references):
    #         text = self.tokenizer.decode(seq, skip_special_tokens=True)
    #         reward = scorer.score(ref, text)['rougeL'].fmeasure
    #         rewards.append(reward)
    #     mean_reward = torch.tensor(rewards, device=self.device).mean()

    #     # 4) Policy‐gradient loss: – E[log p] × reward
    #     loss = - mean_logp * mean_reward
    #     self.optim.zero_grad()
    #     loss.backward()
    #     self.optim.step()

    #     logger.info(f"RL step — reward={mean_reward:.4f}  loss={loss.item():.4f}")
