import torch
from project_finetune_bart_all.tokenizer import get_tokenizer
from project_finetune_bart_all.data import CausalLMData
from project_finetune_bart_all.trainer import ModelTrainer
from project_finetune_bart_all.infer import SloganGenerator
import pandas as pd
from sklearn.model_selection import train_test_split
from project_finetune_bart_all.evaluation import RougeEvaluator
from transformers import BartForConditionalGeneration

import os
# Set environment variable to avoid tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


if __name__ == "__main__":
    
    # Paths
    model_name = "sshleifer/distilbart-cnn-6-6" # Changed model identifier
    # model_name = "facebook/bart-large-cnn" # Changed model identifier
    model_path = f'./models_ftfull/fine_tuned_{model_name.replace("/", "_")}.pt'
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

    if os.path.exists(model_path):
        print(f"-- Loading existing fine-tuned model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("-- Training new model")
        # Note: Batch size might need to be reduced for BART depending on GPU memory
        # e.g., batch_size=8 or 16
        trainer = ModelTrainer(model, tokenizer, train_dataset, device=device, batch_size=16, lr=5e-5,
                                val_dataset=test_dataset,
                                val_batch_size=16,
                                model_save_path=model_path) # Pass tokenizer
        trainer.train(epochs=30, patience=3, min_delta=0.001) # Fine-tuning usually requires fewer epochs

    # 6. Example inference
    print("6. Example inference")
    gen = SloganGenerator(model, tokenizer, device=device)
    example = "Funding property projects through peer to peer lending, creating a win-win situation for both investors and property professionals"
    print(f"Input description: {example}")
    print("Generated slogan:", gen.generate(example))

    exes = [
        # --- valid.csv
        ['Easily deliver personalized activities that enrich the lives of residents in older adult communities. Save time and increase satisfaction.',
        'Build World-Class Recreation Programs'],
       ['Powerful lead generation software that converts abandoning visitors into subscribers with our dynamic marketing tools and Exit Intent® technology.',
        'Most Powerful Lead Generation Software for Marketers'],
       ["Twine matches companies to the best digital and creative freelancers from a network of over 260,000. It's free to post a job and you only pay when you hire.",
        'Hire quality freelancers for your job'],
       ["Looking for fresh web design & development? Need new marketing materials or a smart campaign to drive business? How about a video or updated photos? Let's talk and tell the world your story.",
        'Ohio Marketing, Web Design & Development'],
        # --- test-curated.csv
        ['Our expert team of Analytical Chemists provide eLiquid analysis & manufacturing services, ensuring full regulatory compliance for the e-cigarette market.',
        'E-Liquid Testing UK'],
       ['From placing entire software engineering teams to integrating easily into your current team, we offer bespoke placements of the very best engineers.',
        'Software Development Consultancy London'],
       ['Turning ideas into visual content since 1999. Content Creation Studio in Ghent. Branded content - corporate video - visuals for events - 360 video',
        'The Image Distillery'],
       ['World market leader for robotic vision systems, inline measurement technology & inspection technology. We are your partner at over 25 locations worldwide.',
        'Leading Machine Vision Systems'],
        # --- other examples
        ['People and projects for sustainable change. Experts in sustainability recruitment, we recruit exceptional people into roles working on sustainability projects or in ethical and responsible organisations.',
         'Change Agents UK']
        ]
    print("====================================")
    for ex in exes:
        print(f"Input description: {ex[0]}")
        print("Generated slogan:", gen.generate(ex[0]))
        print("Actual slogan:", ex[1])
        print("====================================")

    # 7. Evaluate with ROUGE scores
    print("\n7. Evaluating with ROUGE scores")
    evaluator = RougeEvaluator(model, tokenizer, device=device)
    
    print("\nEvaluating on training set (subset):")
    train_results, train_preds = evaluator.evaluate_dataset(train_csv, num_samples=100)
    evaluator.print_results(train_results)
    # train_preds.to_csv(f"results/train_predictions_lora_{model_name.replace('/', '_')}.csv", index=False)
    
    print("\nEvaluating on test set (subset):")
    test_results, test_preds = evaluator.evaluate_dataset(test_csv, num_samples=100)
    evaluator.print_results(test_results)
    # test_preds.to_csv(f"results/test_predictions_lora_{model_name.replace('/', '_')}.csv", index=False)
    
    evaluator.save_results(train_results, test_results, output_file=f"results/rouge_scores_{model_name.replace('/', '_')}.txt")
