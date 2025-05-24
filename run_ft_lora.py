import torch
from project_finetune_bart_lora.tokenizer import get_tokenizer
from project_finetune_bart_lora.data import CausalLMData
from project_finetune_bart_lora.trainer import ModelTrainer
from project_finetune_bart_lora.infer import SloganGenerator
import pandas as pd
from sklearn.model_selection import train_test_split
from project_finetune_bart_lora.evaluation import RougeEvaluator
from transformers import BartForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType, PeftModel # Added PEFT imports


import os
# Set environment variable to avoid tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


if __name__ == "__main__":
    
    # Paths
    model_name = "sshleifer/distilbart-cnn-6-6" # Changed model identifier
    lora_model_dir = f'./models_lora_r4a16/lora_adapters_{model_name.replace("/", "_")}' 
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
    print("5. Apply LoRA or Load Fine-tuned LoRA Model")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"-- Using device: {device}")

    # Check if LoRA adapters already exist
    # A common file in a PEFT saved directory is 'adapter_config.json'
    if os.path.exists(os.path.join(lora_model_dir, "adapter_config.json")):
        print(f"-- Loading existing LoRA adapters from {lora_model_dir}")
        # Load the base model and then apply the saved LoRA adapters
        model = PeftModel.from_pretrained(model, lora_model_dir)
    else:
        print("-- Preparing new model for LoRA fine-tuning")
        # Define LoRA configuration
        lora_config = LoraConfig(
            r=8,  # Rank of the update matrices.
            lora_alpha=16,  # Alpha scaling factor.
            # Target modules for BART. You might need to inspect your specific model's layer names.
            # Common ones are query, key, value, output projections in attention, and fc layers.
            target_modules=["q_proj", "v_proj", "k_proj"],
            lora_dropout=0.05,
            bias="none",  # or "all" or "lora_only"
            task_type="SEQ_2_SEQ_LM"  # Crucial for sequence-to-sequence models like BART
        )
        # Wrap the base model with PEFT LoRA configuration
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters() # Useful to see how many parameters are being trained

        # Move the PEFT model to the device before training
        model.to(device)
        
        print("-- Training new model with LoRA")
        # Note: Batch size might need to be reduced for BART depending on GPU memory
        # e.g., batch_size=8 or 16
        trainer = ModelTrainer(model, tokenizer, train_dataset, device=device, batch_size=16, lr=1e-4,
                                val_dataset=test_dataset,
                                val_batch_size=16,
                                model_save_path=lora_model_dir)
        trainer.train(epochs=30, patience=3, min_delta=0.001) # Fine-tuning usually requires fewer epochs

    # Ensure the final model (base + adapters) is on the correct device for inference
    model.to(device)

    # 6. Example inference
    print("6. Example inference")
    gen = SloganGenerator(model, tokenizer, device=device)
    example = "Funding property projects through peer to peer lending, creating a win-win situation for both investors and property professionals"
    print(f"Input description: {example}")
    print("Generated slogan:", gen.generate(example))

    exes = [['Easily deliver personalized activities that enrich the lives of residents in older adult communities. Save time and increase satisfaction.',
        'Build World-Class Recreation Programs'],
       ['Powerful lead generation software that converts abandoning visitors into subscribers with our dynamic marketing tools and Exit Intent® technology.',
        'Most Powerful Lead Generation Software for Marketers'],
       ["Twine matches companies to the best digital and creative freelancers from a network of over 260,000. It's free to post a job and you only pay when you hire.",
        'Hire quality freelancers for your job'],
       ["Looking for fresh web design & development? Need new marketing materials or a smart campaign to drive business? How about a video or updated photos? Let's talk and tell the world your story.",
        'Ohio Marketing, Web Design & Development']]
    print("====================================")
    for ex in exes:
        print(f"Input description: {ex[0]}")
        print("Generated slogan:", gen.generate(ex[0]))
        print("Actual slogan:", ex[1])
        print("====================================")

    # # 7. Evaluate with ROUGE scores
    # print("\n7. Evaluating with ROUGE scores")
    # evaluator = RougeEvaluator(model, tokenizer, device=device)
    
    # print("\nEvaluating on training set (subset):")
    # train_results, train_preds = evaluator.evaluate_dataset(train_csv, num_samples=100)
    # evaluator.print_results(train_results)
    # # train_preds.to_csv(f"results/train_predictions_lora_{model_name.replace('/', '_')}.csv", index=False)
    
    # print("\nEvaluating on test set (subset):")
    # test_results, test_preds = evaluator.evaluate_dataset(test_csv, num_samples=100)
    # evaluator.print_results(test_results)
    # # test_preds.to_csv(f"results/test_predictions_lora_{model_name.replace('/', '_')}.csv", index=False)
    
    # evaluator.save_results(train_results, test_results, output_file=f"results/rouge_scores_{model_name.replace('/', '_')}.txt")
