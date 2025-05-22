import torch
from finetune_bart.tokenizer import get_tokenizer
from finetune_bart.data import CausalLMData
from finetune_bart.trainer import ModelTrainer
from finetune_bart.infer import SloganGenerator
import pandas as pd
from sklearn.model_selection import train_test_split
from finetune_bart.evaluation import RougeEvaluator
from transformers import BartForConditionalGeneration

import os
# Set environment variable to avoid tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


if __name__ == "__main__":
    
    # Paths
    model_name = "sshleifer/distilbart-cnn-6-6" # Changed model identifier
    model_path = f'./models/fine_tuned_{model_name.replace("/", "_")}.pt'
    CSV_PATH = "./data/valid.csv"
    train_csv = './data/valid_train.csv'
    test_csv = './data/valid_test.csv'

    # 1. Split dataset into train and test
    print("1. Split dataset into train and test")
    if not (os.path.exists(train_csv) and os.path.exists(test_csv)):
        print("-- Splitting dataset")
        df = pd.read_csv(CSV_PATH)
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=123)
        train_df.to_csv(train_csv, index=False)
        test_df.to_csv(test_csv, index=False)
    else:
        print("-- Train/Test split already exists.")

    # 2. Load pre-trained tokenizer
    print(f"2. Load pre-trained tokenizer: {model_name}")
    tokenizer = get_tokenizer(model_name)

    # 3. Prepare datasets
    print("3. Prepare datasets")
    train_dataset = CausalLMData(train_csv, tokenizer)
    test_dataset  = CausalLMData(test_csv, tokenizer)

    # 4. Load model
    print(f"4. Load pre-trained model: {model_name}")
    model = BartForConditionalGeneration.from_pretrained(model_name)

    # 5. Train Fine-tuned Model
    print("5. Train Fine-tuned Model")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"-- Using device: {device}")
    model.to(device) # Move model to device before checking path or training

    model_path = "./models/distilbart_slogan_model_final.pt"
    if os.path.exists(model_path):
        print(f"-- Loading existing fine-tuned model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("-- Training new model")
        # Note: Batch size might need to be reduced for BART depending on GPU memory
        # e.g., batch_size=8 or 16
        trainer = ModelTrainer(model, tokenizer, train_dataset, device=device, batch_size=8, lr=5e-5,
                                val_dataset=test_dataset,
                                val_batch_size=8) # Pass tokenizer
        trainer.train(epochs=3, patience=3, min_delta=0.001) # Fine-tuning usually requires fewer epochs

    # 6. Example inference
    print("6. Example inference")
    gen = SloganGenerator(model, tokenizer, device=device)
    example = "Funding property projects through peer to peer lending, creating a win-win situation for both investors and property professionals"
    print(f"\nInput description: {example}")
    print("Generated slogan:", gen.generate(example))

    # 7. Evaluate with ROUGE scores
    print("\n7. Evaluating with ROUGE scores")
    evaluator = RougeEvaluator(model, tokenizer, device=device)
    
    print("\nEvaluating on training set (subset):")
    train_results, train_preds = evaluator.evaluate_dataset(train_csv, num_samples=50) # Reduced samples for faster eval
    evaluator.print_results(train_results)
    
    print("\nEvaluating on test set (subset):")
    test_results, test_preds = evaluator.evaluate_dataset(test_csv, num_samples=50) # Reduced samples for faster eval
    evaluator.print_results(test_results)
    
    evaluator.save_results(train_results, test_results, output_file=f"results/rouge_scores_{model_name.replace('/', '_')}.txt")
