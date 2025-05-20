import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

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
        val_batch_size: int = None
    ):
        self.device = device
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=dataset.collate_fn
        )
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
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
        
        # Add learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optim, mode='min', factor=0.5, patience=1
        )

    def validate(self):
        """Run validation and return average loss."""
        if self.val_dataset is None:
            return None
            
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                logits = self.model(inputs)
                
                T, B, V = logits.shape
                loss = self.criterion(logits.view(T*B, V), labels.view(T*B))
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
        self.model.train()
        best_loss = float('inf')
        no_improve_count = 0

        for epoch in range(epochs):
            total_loss = 0.0
            progress_bar = tqdm(self.loader, desc=f"Epoch {epoch+1}/{epochs}")

            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optim.zero_grad()
                logits = self.model(inputs)

                T, B, V = logits.shape
                loss = self.criterion(logits.view(T*B, V), labels.view(T*B))
                
                loss.backward()
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optim.step()

                total_loss += loss.item()
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            train_avg_loss = total_loss / len(self.loader)
            # Validate if we have a validation set
            val_loss = self.validate()
            # Use validation loss if available, otherwise use training loss
            current_loss = val_loss if val_loss is not None else train_avg_loss
            print(f"Epoch {epoch+1}/{epochs}  train_loss={train_avg_loss:.4f}  val_loss={val_loss:.4f}")
            
            # Update learning rate based on loss
            self.scheduler.step(current_loss)

            # Early stopping
            if current_loss < best_loss - min_delta:
                best_loss = current_loss
                no_improve_count = 0
                # Save model if it's the best so far
                torch.save(self.model.state_dict(), "./models/decoder_only_model.pt")
                print(f"Model saved with loss: {best_loss:.4f}")
            else:
                # No improvement
                no_improve_count += 1
                print(f"No improvement for {no_improve_count} epochs")
                if no_improve_count >= patience:
                    print(f"Early stopping after {epoch+1} epochs")
                    break
    
        # Save final model regardless of performance
        torch.save(self.model.state_dict(), "./models/decoder_only_model_final.pt")
        print("Training completed!")
        