import pandas as pd
from rouge_score import rouge_scorer
from tqdm import tqdm

class RougeEvaluator:
    """
    Evaluates slogan generation using ROUGE metrics.
    """
    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def generate_prediction(self, desc: str) -> str:
        self.model.eval()
        inputs = self.tokenizer(
            desc,
            max_length=128,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=64, # Max length for generated slogan
            num_beams=4,
            early_stopping=True
        )
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    def evaluate_dataset(self, csv_path: str, num_samples: int = None):
        df = pd.read_csv(csv_path)
        if num_samples:
            df = df.sample(n=min(num_samples, len(df)), random_state=123)

        all_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Evaluating ROUGE"):
            description = str(row['desc'])
            reference_slogan = str(row['output'])
            
            try:
                predicted_slogan = self.generate_prediction(description)
                
                scores = self.scorer.score(reference_slogan, predicted_slogan)
                all_scores['rouge1'].append(scores['rouge1'].fmeasure)
                all_scores['rouge2'].append(scores['rouge2'].fmeasure)
                all_scores['rougeL'].append(scores['rougeL'].fmeasure)
            except:
                pass
            
        avg_scores = {metric: sum(values)/len(values) for metric, values in all_scores.items()}
        return avg_scores
    
    def print_results(self, results):
        """
        Print the evaluation results in a formatted way.
        """
        print("\n===== ROUGE Evaluation Results =====")
        print(f"ROUGE-1: {results['rouge1']:.4f}")
        print(f"ROUGE-2: {results['rouge2']:.4f}")
        print(f"ROUGE-L: {results['rougeL']:.4f}")
        print("====================================\n")
    
    def save_results(self, train_results, test_results, output_file="./rouge_scores_distilbart.txt"):
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            f.write("Train Set Results:\n")
            for metric, score in train_results.items():
                f.write(f"  {metric}: {score:.4f}\n")
            f.write("\nTest Set Results:\n")
            for metric, score in test_results.items():
                f.write(f"  {metric}: {score:.4f}\n")
        print(f"ROUGE scores saved to {output_file}")
        