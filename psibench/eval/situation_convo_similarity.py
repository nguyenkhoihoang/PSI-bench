"""Analyze similarity between conversations and their situations."""

import json
import os
from pathlib import Path
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.util import ngrams
from rouge_score import rouge_scorer
from bert_score import score
import torch

from data_loader.esc import (
    load_esc_data,
    load_esc_data_with_indices,
    load_esc_original_data,
)
from data_loader.main_loader import load_synthetic_data_to_df, get_synthetic_indices
#TODO: If use this similarity in final benchmark, make it generalize to other datasets besides ESC
## Situation vs Conversation: Real Convo
def concat_client_messages(messages_list):
    assistant_messages = []
    for message in messages_list:
        if message.get('role') == 'assistant':
            assistant_messages.append(message.get('content', ''))
    return " ".join(assistant_messages)

def clean_text(text):
    """
    Cleans the input text by lowercasing, removing punctuation, and removing stopwords.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return " ".join(words)

def calculate_similarity_metrics(cleaned_assistant_messages, cleaned_situation_text):
    """
    Calculates various similarity metrics between two cleaned text strings.

    Args:
        cleaned_assistant_messages: The cleaned string of assistant messages.
        cleaned_situation_text: The cleaned string of the situation text.

    Returns:
        A dictionary containing the calculated similarity metrics.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1'])
    rouge_scores = scorer.score(cleaned_situation_text, cleaned_assistant_messages)

    word_overlap_percentage = calculate_word_overlap(cleaned_assistant_messages, cleaned_situation_text)
    _, R, _ = score([cleaned_assistant_messages], [cleaned_situation_text], lang="en", verbose=False, device="cuda")
    bert_score_recall = R.mean().item()

    similarity_metrics = {
        'bert_score_recall': bert_score_recall,
        'rouge1_recall': rouge_scores['rouge1'].recall,
        'word_overlap_percentage': word_overlap_percentage
    }
    return similarity_metrics

def calculate_word_overlap(text1, text2):
    """Calculate percentage of word overlap between two texts."""
    words1 = set(text1.split())
    words2 = set(text2.split())
    overlap = len(words1.intersection(words2))
    total_words = len(words1.union(words2))
    return (overlap / total_words) * 100 if total_words > 0 else 0

def calculate_ngram_overlap(text1, text2, n):
    """Calculate percentage of n-gram overlap between two texts."""
    words1 = text1.split()
    words2 = text2.split()
    ngrams1 = set(ngrams(words1, n))
    ngrams2 = set(ngrams(words2, n))
    overlap = len(ngrams1.intersection(ngrams2))
    total_ngrams = len(ngrams1.union(ngrams2))
    return (overlap / total_ngrams) * 100 if total_ngrams > 0 else 0



def analyze_dataset(df: pd.DataFrame, output_dir: str = 'output', k: int = None, is_real: bool = False):
    """
    Analyze similarity metrics for a dataset and save results.
    
    Args:
        df: DataFrame containing ESC samples
        output_dir: Directory to save results
        k: Number of samples to analyze. If None, analyzes all samples.
        is_real: Whether analyzing real conversations (need to match id_source)
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load original ESC data if analyzing real conversations
    esc_data = load_esc_original_data() if is_real else {}
    
    similarities_results = []    
    if k is not None:
        df = df.sample(n=min(k, len(df)), random_state=42)
    
    for _, row in df.iterrows():
        messages_list = row['messages']
        
        if is_real:
            # For real conversations, get situation from original ESConv using id_source
            id_source = row["id_source"]
            situation = esc_data[int(id_source)].get("situation", "")
            if not situation:
                print(f"[WARNING] No situation found for id_source: {id_source}")
                continue
        else:
            # For synthetic data, get from profile
            profile = row['profile']
            if isinstance(profile, str):
                profile = json.loads(profile)
            situation = profile.get('situation of the client', '')
        
        if not situation:
            # Skip rows without situation data
            continue

        cleaned_assistant_messages = clean_text(concat_client_messages(messages_list))
        cleaned_situation_text = clean_text(situation)

        sample_similarity = calculate_similarity_metrics(cleaned_assistant_messages, cleaned_situation_text)
        similarities_results.append(sample_similarity)

    # Save raw results
    with open(output_dir / 'similarities_results.json', 'w') as f:
        json.dump(similarities_results, f, indent=2)

    results_df = pd.DataFrame(similarities_results)    
    analysis = results_df.describe()
    selected_analysis = analysis.loc[['mean', 'min', 'max', 'std',
                                    #   '25%', '75%'
                                      ], 
                                   ['bert_score_recall', 'rouge1_recall', 'word_overlap_percentage']]
    
    selected_analysis = selected_analysis.round(5)    
    with open(output_dir / 'similarity_analysis.txt', 'w') as f:
        f.write(selected_analysis.to_string(float_format='{:.5f}'.format))
    print(f"Data Results: Real={is_real}")
    print(selected_analysis.to_string(float_format='{:.5f}'.format))
    return selected_analysis


def compare_real_and_synthetic(data_dir: str, output_dir: str = 'output', k: int = None):
    """
    Compare similarity metrics between real and synthetic data.
    Ensures they're from the same situations (matching indices).
    
    Args:
        data_dir: Directory containing synthetic data
        output_dir: Base output directory
        k: Number of samples to analyze
    """
    from datetime import datetime
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path(output_dir) / timestamp
    
    synthetic_indices = get_synthetic_indices(data_dir)    
    if k:
        synthetic_indices = synthetic_indices[:k]
        
    real_df = load_esc_data_with_indices(synthetic_indices)
    print(f"[INFO] Loaded {len(real_df)} real conversations from eeyore_profile")
    synthetic_df = load_synthetic_data_to_df(data_dir)
    print(f"[INFO] Loaded {len(synthetic_df)} synthetic conversations from {data_dir}")
    
    # Analyze both datasets with automatically named directories
    real_output = base_output_dir / "real_analysis"
    synthetic_output = base_output_dir / "synthetic_analysis"
    
    analyze_dataset(df=real_df, output_dir=real_output, k=None, is_real=True)
    analyze_dataset(df=synthetic_df, output_dir=synthetic_output, k=None, is_real=False)


def main():
    """Run similarity analysis on ESC dataset or synthetic data."""
    import argparse
    from datetime import datetime
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Analyze similarity between assistant messages and situations.')
    parser.add_argument('-k', '--num-samples', type=int, 
                       help='Number of samples to analyze. If not specified, analyzes all samples.')
    parser.add_argument('--mode', type=str, choices=['analyze', 'compare'], default='analyze', 
                       help='Mode of operation: analyze single dataset or compare real vs synthetic')
    parser.add_argument('--data-type', type=str, choices=['real', 'synthetic'], default='real',
                       help='Type of data to analyze (real ESC or synthetic)')
    parser.add_argument('--data-dir', type=str, 
                       help='Directory containing synthetic session data')
    parser.add_argument('-o', '--output-dir', type=str, default='output',
                       help='Base output directory')
    args = parser.parse_args()

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path(args.output_dir) / timestamp
    
    if args.mode == 'compare':
        if not args.data_dir:
            print("[ERROR] --data-dir is required for comparison mode")
            return
        compare_real_and_synthetic(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            k=args.num_samples
        )
    else:
        # Single dataset analysis mode
        if args.data_type == 'real':
            print(f"[INFO] Loading real ESC data...")
            df = load_esc_data()
            output_subdir = base_output_dir / "real_analysis"
        else:
            if not args.data_dir:
                print("[ERROR] --data-dir is required when using --data-type=synthetic")
                return
            print(f"[INFO] Loading synthetic data from {args.data_dir}...")
            df = load_synthetic_data_to_df(args.data_dir)
            output_subdir = base_output_dir / "synthetic_analysis"
        
        if df.empty:
            print("[ERROR] No data loaded")
            return
        
        print(f"[INFO] Loaded {len(df)} samples")
        print(f"\nAnalyzing {'all' if args.num_samples is None else args.num_samples} assistant (patient) messages for similarity...")
        analysis = analyze_dataset(df, output_dir=output_subdir, k=args.num_samples, is_real=(args.data_type == 'real'))
        print("\nSimilarity Analysis Results (Assistant/Patient Messages):")
        print(analysis.to_string(float_format='{:.5f}'.format))

if __name__ == "__main__":
    main()
    
