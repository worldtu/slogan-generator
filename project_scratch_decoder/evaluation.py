import pandas as pd
import torch
import logging
from rouge_score import rouge_scorer
from tqdm import tqdm
from project_scratch_barttok.infer import SloganGenerator

logger = logging.getLogger(__name__)


class RougeEvaluator:
    """
    Evaluates slogan generation using ROUGE metrics.
    """
    def __init__(self, model, tokenizer, device="cpu"):
        self.generator = SloganGenerator(model, tokenizer, device)
        self.scorer   = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
        self.device = device
        
    def evaluate_dataset(self, csv_path, num_samples=None):
        """
        Evaluate the model on a dataset using ROUGE metrics.
        """
        df = pd.read_csv(csv_path)
        if num_samples is not None:
            df = df.sample(n=min(num_samples, len(df)), random_state=42)
        
        rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
        
        logger.info(f"Evaluating on {len(df)} samples from {csv_path}")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            description = row['desc']
            reference   = row['output']
            
            generated = self.generator.generate_beam(description)
            scores = self.scorer.score(reference, generated)
            
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        results = {
            'rouge1': sum(rouge1_scores) / len(rouge1_scores),
            'rouge2': sum(rouge2_scores) / len(rouge2_scores),
            'rougeL': sum(rougeL_scores) / len(rougeL_scores)
        }
        
        return results
    
    def print_results(self, results):
        """
        Log the evaluation results in a formatted way.
        """
        logger.info("===== ROUGE Evaluation Results =====")
        logger.info(f"ROUGE-1: {results['rouge1']:.4f}")
        logger.info(f"ROUGE-2: {results['rouge2']:.4f}")
        logger.info(f"ROUGE-L: {results['rougeL']:.4f}")
        logger.info("====================================")
    
    def save_results(self, train_results, test_results, output_path="./evaluation_results.csv"):
        """
        Save evaluation results to a CSV file.
        """
        results_df = pd.DataFrame({
            'Dataset': ['Training Set', 'Test Set'],
            'ROUGE-1': [train_results['rouge1'], test_results['rouge1']],
            'ROUGE-2': [train_results['rouge2'], test_results['rouge2']],
            'ROUGE-L': [train_results['rougeL'], test_results['rougeL']]
        })
        
        results_df.to_csv(output_path, index=False)
        logger.info(f"Evaluation results saved to {output_path}")
        