import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import os


class ModelTrainer:
    """
    Encapsulates training loop for Hugging Face sequence-to-sequence models like BART.
    """
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        dataset: Dataset,
        batch_size: int = 8,
        lr: float = 5e-5,
        device: str = "cpu",
        val_dataset: Dataset = None,
        val_batch_size: int = None,
        warmup_ratio: float = 0.1
    ):
        self.device = device
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.dataset = dataset
        collate_function = self.dataset.collate_fn
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_function
        )
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.warmup_ratio = warmup_ratio

        # Validation dataset and loader
        self.val_dataset = val_dataset
        if val_dataset is not None:
            val_collate_function = self.val_dataset.collate_fn
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=val_batch_size or batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=val_collate_function
            )

        # # Add learning rate scheduler
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optim, mode='min', factor=0.5, patience=1
        # )

    def validate(self):
        """Run validation and return average loss."""
        if self.val_dataset is None:
            return None

        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss # Loss is directly provided by the model
                total_loss += loss.item()
                
        avg_loss = total_loss / len(self.val_loader)
        self.model.train()  # Set back to training mode
        return avg_loss

    def train(self, epochs: int = 5, patience: int = 3, min_delta: float = 0.0):
        """
        Train the model with early stopping.
        
        Args:
            epochs: Maximum number of epochs to train for
            patience: Number of epochs to wait for improvement before stopping
            min_delta: Minimum change in validation loss to qualify as improvement
        """
        total_steps = len(self.loader) * epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        self.warmup_scheduler = get_linear_schedule_with_warmup(
            self.optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        self.model.train()
        best_loss = float('inf')
        no_improve_count = 0

        for epoch in range(epochs):
            total_loss = 0.0
            progress_bar = tqdm(self.loader, desc=f"Epoch {epoch+1}/{epochs}")

            for batch in progress_bar:
                self.optim.zero_grad()

                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss # Loss is directly provided by the model
                loss.backward()
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optim.step()
                self.warmup_scheduler.step()

                total_loss += loss.item()
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            train_avg_loss = total_loss / len(self.loader)
            # Validate if we have a validation set
            val_loss = self.validate()
            # Use validation loss if available, otherwise use training loss
            current_loss = val_loss if val_loss is not None else train_avg_loss
            print(f"Epoch {epoch+1}/{epochs}  train_loss={train_avg_loss:.4f}  val_loss={val_loss:.4f}")

            # Early stopping
            if current_loss < best_loss - min_delta:
                best_loss = current_loss
                no_improve_count = 0
                # Ensure the path exists
                os.makedirs(os.path.dirname("./models/distilbart_slogan_model.pt"), exist_ok=True)
                torch.save(self.model.state_dict(), "./models/distilbart_slogan_model.pt")
                print(f"Model saved with loss: {best_loss:.4f}")
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    print(f"Early stopping triggered after {patience} epochs without improvement.")
                    break

        # Save final model regardless of performance
        torch.save(self.model.state_dict(), "./models/distilbart_slogan_model_final.pt")
        print("Training completed!")
        