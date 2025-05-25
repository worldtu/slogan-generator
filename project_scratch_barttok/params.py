import torch
from torchinfo import summary
from model import DecoderOnlyTransformer
from tokenizer import get_tokenizer


# --- Step 1: Instantiate or Load your Model ---
# You MUST have an instance of your model first.
# If you are creating a new model, you need to provide at least vocab_size.
# The other parameters have defaults in your __init__ method, but you can override them.

# model_name = "sshleifer/distilbart-cnn-6-6" # Changed model identifier
# model_path = f'./models_scratch/barttok_decoder_only.pt'
# tokenizer = get_tokenizer(model_name)

# Example: Instantiating a new model (provide your actual vocab_size)
# You would get vocab_size from your tokenizer, e.g., tokenizer.get_vocab_size()
actual_vocab_size = 50265 # Replace with your actual vocabulary size
model = DecoderOnlyTransformer(vocab_size=actual_vocab_size)
print(f"Instantiating DecoderOnlyTransformer with vocab_size: {actual_vocab_size}")

# model.load_state_dict(torch.load(model_path))
# model = DecoderOnlyTransformer(
#     vocab_size=actual_vocab_size,
#     # You can specify other params here if they differ from defaults, e.g.:
#     # d_model=256,
#     # n_head=4,
#     # num_layers=3,
#     # dim_ff=1024,
#     # max_len=512,
#     # dropout=0.1
# )

# OR, if you have a saved model checkpoint, load it:
# model_path = "path/to/your/saved_model.pth"
# model = DecoderOnlyTransformer(vocab_size=actual_vocab_size) # Initialize with necessary params
# model.load_state_dict(torch.load(model_path))
# model.eval() # Set to evaluation mode if you're just inspecting

print("Model instantiated/loaded.")

# --- Step 2: Retrieve parameters from the model instance ---
retrieved_config = {
    'vocab_size': model.token_emb.num_embeddings,
    'd_model': model.token_emb.embedding_dim,
    'n_head': model.decoder.layers[0].self_attn.num_heads,
    'num_layers': len(model.decoder.layers),
    'dim_ff': model.decoder.layers[0].linear1.out_features, # dim_feedforward
    'max_len': model.pos_emb.num_embeddings,
    'dropout': model.dropout.p
}

print("\nRetrieved Model Configuration:")
for key, value in retrieved_config.items():
    print(f"  {key}: {value}")

# --- Step 3: Use torchinfo with an example input shape ---
# For a transformer, a common input shape is (sequence_length, batch_size) 
# as per your model's forward(self, input_ids) where input_ids: (T, B)
example_input_shape = (64, 1) # sequence_length=64, batch_size=1

# Ensure the input tensor matches the expected dtype (usually long for input_ids)
example_input = torch.randint(0, retrieved_config['vocab_size'], example_input_shape, dtype=torch.long)

print(f"\nGenerating model summary with input shape: {example_input_shape} (SeqLen, Batch)")
model_summary = summary(model, input_data=example_input, col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"], verbose=0)
# verbose=0 to prevent torchinfo from printing its own summary before we print ours

print(model_summary)

print(f"\nTotal parameters: {model_summary.total_params}")
print(f"Trainable parameters: {model_summary.trainable_params}")
print(f"Non-trainable parameters: {model_summary.total_params - model_summary.trainable_params}")
