import os
import logging
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

from project_scratch_barttok.tokenizer import get_tokenizer
from project_scratch_barttok.data import CausalLMData
from project_scratch_barttok.model import DecoderOnlyTransformer
from project_scratch_barttok.trainer import ModelTrainer
from project_scratch_barttok.infer import SloganGenerator
from project_scratch_barttok.evaluation import RougeEvaluator 

torch.cuda.empty_cache()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler() # To also print to console
        ])
logger = logging.getLogger(__name__)

# Set environment variable to avoid tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


if __name__ == "__main__":
    
    # Paths
    model_name = "sshleifer/distilbart-cnn-6-6" # Changed model identifier
    model_path = f'./models_scratch/barttok_decoder_only.pt'
    # tokenizer_path = './models/tokenizer.json'
    CSV_PATH = "./data/valid.csv"
    train_csv = './data/valid_train.csv'
    test_csv = './data/valid_test.csv'

    # 1. Split dataset into train and test
    logger.info("1. Split dataset into train and test")
    if not (os.path.exists(train_csv) and os.path.exists(test_csv)):
        logger.info("-- Splitting dataset")
        df = pd.read_csv(CSV_PATH)
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=123)
        train_df.to_csv(train_csv, index=False)
        test_df.to_csv(test_csv, index=False)
    else:
        logger.info("-- Train/Test split already exists.")

    # # 2. Train tokenizer on full dataset
    # print("2. Train tokenizer on full dataset")
    # tt = TokenizerTrainer()
    # tokenizer = tt.train(train_csv)

    # 2. Load pre-trained tokenizer
    logger.info(f"2. Load pre-trained tokenizer: {model_name}")
    tokenizer = get_tokenizer(model_name)

    # 3. Prepare datasets
    logger.info("3. Prepare datasets")
    train_dataset = CausalLMData(train_csv, tokenizer)
    test_dataset  = CausalLMData(test_csv, tokenizer)

    # 4. Build model
    logger.info("4. Build model")
    model = DecoderOnlyTransformer(vocab_size=tokenizer.vocab_size, tokenizer=tokenizer)

    # 5. Train
    logger.info("5. Train")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"-- Using device: {device}")
    model.to(device)
    
    # Check if model already exists
    if os.path.exists(model_path):
        logger.info("-- Loading existing model")
        model.load_state_dict(torch.load(model_path))
    else:
        logger.info("-- Training new model")
        trainer = ModelTrainer(model, tokenizer, train_dataset, device=device,
                                model_save_path=model_path,
                                batch_size=32,  # 128 on virtual machines with 24 GB memory
                                lr=1e-4,
                                val_dataset=test_dataset,
                                val_batch_size=32,
                                epochs=20)
        # trainer.train(epochs=2000, patience=20, min_delta=0.001)  # only on virtual machines
        trainer.train(patience=2, min_delta=0.001)

    # 6. Example inference
    logger.info("6. Example inference")
    gen = SloganGenerator(model, tokenizer, device)
    example = "Funding property projects through peer to peer lending, creating a win-win situation for both investors and property professionals"
    logger.info("Generated slogan:", gen.generate_beam(example))

    examples = [
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
    logger.info("====================================")
    for inp, actual in examples:
        slogan = gen.generate_beam(inp)
        logger.info(f"Input description: {inp}")
        logger.info(f"Generated slogan: {slogan}")
        logger.info(f"Actual slogan:    {actual}")
        logger.info("====================================")

    # 7. Evaluate with ROUGE scores
    logger.info("7. Evaluating with ROUGE scores")
    evaluator = RougeEvaluator(model, tokenizer, device)
    
    # Evaluate on training set (you can limit samples for speed)
    logger.info("Evaluating on training set:")
    train_results = evaluator.evaluate_dataset(train_csv, num_samples=100)
    evaluator.print_results(train_results)
    
    # Evaluate on test set
    logger.info("Evaluating on test set:")
    test_results = evaluator.evaluate_dataset(test_csv, num_samples=100)
    evaluator.print_results(test_results)
    
    # Save the evaluation results
    evaluator.save_results(train_results, test_results)
    logger.info("ROUGE evaluation complete and results saved.")
