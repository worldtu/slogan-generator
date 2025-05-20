import torch
from scratch_model.tokenizer import TokenizerTrainer
from scratch_model.data import CausalLMData
from scratch_model.model import DecoderOnlyTransformer
from scratch_model.trainer import ModelTrainer
from scratch_model.infer import SloganGenerator
import pandas as pd
from sklearn.model_selection import train_test_split
from scratch_model.evaluation import RougeEvaluator

import os
# Set environment variable to avoid tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


if __name__ == "__main__":
    
    # Paths
    model_path = './models/decoder_only_model.pt'
    tokenizer_path = './models/tokenizer.json'
    CSV_PATH = "./data/valid.csv"
    train_csv = './data/valid_train.csv'
    test_csv = './data/valid_test.csv'

    # 1. Split dataset into train and test
    print("1. Split dataset into train and test")
    df = pd.read_csv(CSV_PATH)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=123)
    train_csv = "./data/valid_train.csv"
    test_csv = "./data/valid_test.csv"
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    # 2. Train tokenizer on full dataset
    print("2. Train tokenizer on full dataset")
    tt = TokenizerTrainer()
    tokenizer = tt.train(train_csv)

    # 3. Prepare datasets
    print("3. Prepare datasets")
    train_dataset = CausalLMData(train_csv, tokenizer)
    test_dataset  = CausalLMData(test_csv, tokenizer)

    # 4. Build model
    print("4. Build model")
    model = DecoderOnlyTransformer(vocab_size=tokenizer.vocab_size)

    # 5. Train
    print("5. Train")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Check if model already exists
    if os.path.exists(model_path):
        print("-- Loading existing model")
        model.load_state_dict(torch.load(model_path))
        model.to(device)
    else:
        print("-- Training new model")
        trainer = ModelTrainer(model, tokenizer, train_dataset, device=device,
                                val_dataset=test_dataset,
                                val_batch_size=64)
        trainer.train(epochs=3, patience=3, min_delta=0.001)

    # 6. Example inference
    print("6. Example inference")
    gen = SloganGenerator(model, tokenizer, device)
    example = "Funding property projects through peer to peer lending, creating a win-win situation for both investors and property professionals"
    print("\nGenerated slogan:", gen.generate(example))

    # 7. Evaluate with ROUGE scores
    print("\n7. Evaluating with ROUGE scores")
    evaluator = RougeEvaluator(model, tokenizer, device)
    
    # Evaluate on training set (you can limit the number of samples for faster evaluation)
    print("\nEvaluating on training set:")
    train_results = evaluator.evaluate_dataset(train_csv, num_samples=100)
    evaluator.print_results(train_results)
    
    # Evaluate on test set
    print("\nEvaluating on test set:")
    test_results = evaluator.evaluate_dataset(test_csv, num_samples=100)
    evaluator.print_results(test_results)
    
    # Save the evaluation results
    evaluator.save_results(train_results, test_results)
