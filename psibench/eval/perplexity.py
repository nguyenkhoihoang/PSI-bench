import json
import torch
import os

import argparse
import logging
import wandb
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

from data_loader.main_loader import load_eeyore_dataset
load_dotenv()
os.environ['HF_HOME'] = os.getenv('HF_HOME')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser(description="Calculate perplexity of conversations")
    parser.add_argument(
        "--model",
        type=str,
        default="liusiyang/eeyore_sft_llama3.1_8B_epoch2",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="esc"
    )
    parser.add_argument(
        "--max_conversations",
        type=int,
        default=None,
        help="Limit the number of conversations to process (for debugging)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for model inference"
    )
    args = parser.parse_args()
    return args
    
def calculate_perplexity(
    args,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, Any]:

    run_name = f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project="perplexity-analysis",
        name=run_name,
        entity=None,
        config={
            "model": args.model,
            "dataset_type": args.dataset_type,
            "device": device
        }
    )
    
    logger.info(f"Loading model: {args.model}")
    logger.info(f"HF_HOME: {os.environ.get('HF_HOME', 'not set (using default)')}")
    wandb.log({"status": "loading_model"})

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=os.getenv('HF_HOME'))
    model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir=os.getenv('HF_HOME')).to(device)
    model.eval()
    
    conversations = load_eeyore_dataset(args.dataset_type)
    logger.info(f"Loaded {len(conversations)} conversations")
    wandb.log({"num_conversations_loaded": len(conversations)})

    # Limit number of conversations for debugging
    if args.max_conversations is not None:
        conversations = conversations.iloc[:args.max_conversations]
        logger.info(f"Limiting to first {args.max_conversations} conversations for debugging.")

    all_turn_perplexities = []  # All individual patient turn perplexities
    conversation_perplexities_list = []  # List of turn perplexities per conversation
    all_losses = []

    logger.info("Calculating turn-level perplexity for patient turns given full conversation history...")
    with torch.no_grad():
        for idx, row in tqdm(conversations.iterrows(), total=len(conversations), desc="Processing conversations"):
            conversation = row['messages']
            profile = row["profile"]

            conversation_turns_perplexities = []
            patient_turn_count = 0
            for turn_idx, turn_text in enumerate(conversation):
                if not turn_text['role'] == 'assistant':
                    continue
                if patient_turn_count >= 20:
                    break
                try:    
                    # Build context: all previous turns (therapist + patient) up to this turn
                    context = "\n".join([m['content'] for m in conversation[:turn_idx]])
                    response = turn_text['content']
                    full_input = profile + "\n" + context + "\n" + response if context else response

                    # Tokenize full input
                    encodings = tokenizer(full_input, return_tensors="pt", truncation=True)
                    input_ids = encodings["input_ids"].to(device)

                    # Tokenize context to find its length in tokens
                    if context:
                        context_ids = tokenizer(context, return_tensors="pt")["input_ids"]
                        context_len = context_ids.shape[1]
                    else:
                        context_len = 0

                    # Prepare labels: mask context tokens with -100, only score response tokens
                    labels = input_ids.clone()
                    if context_len > 0:
                        labels[:, :context_len] = -100

                    outputs = model(input_ids, labels=labels)
                    loss = outputs.loss
                    perplexity = torch.exp(loss)
                    breakpoint()

                    all_losses.append(loss.item())
                    all_turn_perplexities.append(perplexity.item())
                    conversation_turns_perplexities.append(perplexity.item())
                    patient_turn_count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing conversation {idx}, turn {turn_idx}: {e}")
                    continue

            # Store all turn perplexities for this conversation
            if conversation_turns_perplexities:
                conversation_perplexities_list.append(conversation_turns_perplexities)
    
    # Calculate statistics for turn-level perplexity
    if all_turn_perplexities:
        avg_turn_perplexity = sum(all_turn_perplexities) / len(all_turn_perplexities)
        min_turn_perplexity = min(all_turn_perplexities)
        max_turn_perplexity = max(all_turn_perplexities)
    else:
        avg_turn_perplexity = min_turn_perplexity = max_turn_perplexity = 0

    # Calculate statistics for conversation-level perplexity (average of turns within each conversation)
    conversation_avg_perplexities = [
        sum(turns) / len(turns) for turns in conversation_perplexities_list if turns
    ]

    if conversation_avg_perplexities:
        avg_conv_perplexity = sum(conversation_avg_perplexities) / len(conversation_avg_perplexities)
        min_conv_perplexity = min(conversation_avg_perplexities)
        max_conv_perplexity = max(conversation_avg_perplexities)
    else:
        avg_conv_perplexity = min_conv_perplexity = max_conv_perplexity = 0

    avg_loss = sum(all_losses) / len(all_losses) if all_losses else 0

    results = {
        "model": args.model,
        "dataset_level": {
            "num_conversations": len(conversations),
            "num_conversations_processed": len(conversation_avg_perplexities),
            "num_seeker_turns": len(all_turn_perplexities),
            "perplexity": {
                "average": avg_conv_perplexity,
                "min": min_conv_perplexity,
                "max": max_conv_perplexity
            }
        },
        "turn_level": {
            "perplexity": {
                "average": avg_turn_perplexity,
                "min": min_turn_perplexity,
                "max": max_turn_perplexity
            }
        },
        "average_loss": avg_loss,
        "detailed_data": {
            "all_turn_perplexities": all_turn_perplexities,
            "conversation_perplexities_by_conversation": conversation_perplexities_list,
            "conversation_averages": conversation_avg_perplexities,
            "losses": all_losses
        }
    }
    
    # Log metrics to wandb (only log available statistics)
    wandb.log({
        "dataset_level/num_conversations": results["dataset_level"]["num_conversations"],
        "dataset_level/num_seeker_turns": results["dataset_level"]["num_seeker_turns"],
        "dataset_level/perplexity_avg": results["dataset_level"]["perplexity"]["average"],
        "dataset_level/perplexity_min": results["dataset_level"]["perplexity"]["min"],
        "dataset_level/perplexity_max": results["dataset_level"]["perplexity"]["max"],
        "turn_level/perplexity_avg": results["turn_level"]["perplexity"]["average"],
        "turn_level/perplexity_min": results["turn_level"]["perplexity"]["min"],
        "turn_level/perplexity_max": results["turn_level"]["perplexity"]["max"],
        "average_loss": results["average_loss"]
    })
    
    return results


def save_perplexity_data(conversation_perplexities_list: List[List[float]], output_path: str) -> None:
    """
    Save perplexity data in a structured format: [num_conversations][num_turns_per_conversation][turn_perplexity]
    
    Args:
        conversation_perplexities_list: List of lists where each inner list contains perplexities for each turn in a conversation
        output_path: Path to save the data (as JSON)
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON with nested structure
    data = {
        "structure": "[num_conversations][num_turns_per_conversation]",
        "num_conversations": len(conversation_perplexities_list),
        "perplexities": conversation_perplexities_list
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved detailed perplexity data to: {output_path}")
    
    # Also save as CSV for easier analysis
    csv_path = output_path.replace(".json", ".csv")
    try:
        # Create flat structure for CSV
        rows = []
        for conv_idx, turns in enumerate(conversation_perplexities_list):
            for turn_idx, ppl in enumerate(turns):
                rows.append({
                    "conversation_id": conv_idx,
                    "turn_id": turn_idx,
                    "perplexity": ppl
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved perplexity data as CSV to: {csv_path}")
    except ImportError:
        logger.warning("pandas not available, skipping CSV export")
    except Exception as e:
        logger.error(f"Failed to save CSV: {e}")


def plot_perplexity(conversation_perplexities_list: List[List[float]], all_turn_perplexities: List[float], output_path_trend: str = "output/perplexity_trend.png", output_path_dist: str = "output/perplexity_distribution.png") -> None:
    """
    Plot two graphs:
    1. Trend of average perplexity across first 20 turns (averaged across all conversations)
    2. Distribution histogram of all turn perplexities
    
    Args:
        conversation_perplexities_list: List of lists where each inner list contains perplexities for each turn
        all_turn_perplexities: List of all individual turn perplexities
        output_path_trend: Path to save the trend plot
        output_path_dist: Path to save the distribution plot
    """
    try:
        # ===== Plot 1: Trend across first 20 turns =====
        max_turns_to_plot = 20
        
        # Calculate average perplexity for each turn position across all conversations
        turn_averages = []
        for turn_idx in range(max_turns_to_plot):
            turn_perplexities = []
            for conv_turns in conversation_perplexities_list:
                # Get perplexity at this turn index if it exists in this conversation
                if turn_idx < len(conv_turns):
                    turn_perplexities.append(conv_turns[turn_idx])
            
            if turn_perplexities:
                avg_ppl = sum(turn_perplexities) / len(turn_perplexities)
                turn_averages.append(avg_ppl)
            else:
                break  # No more turns available in conversations
        
        if turn_averages:
            plt.figure(figsize=(14, 6))
            turn_indices = list(range(1, len(turn_averages) + 1))
            plt.plot(turn_indices, turn_averages, marker='o', linestyle='-', linewidth=2.5, markersize=8, alpha=0.7, color='steelblue')
            plt.xlabel('Turn Position', fontsize=12)
            plt.ylabel('Average Perplexity', fontsize=12)
            plt.title('Average Perplexity Trend Across First 20 Turns (Averaged Across All Conversations)', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.xticks(turn_indices)
            
            # Add statistics to plot
            avg = sum(turn_averages) / len(turn_averages)
            min_ppl = min(turn_averages)
            max_ppl = max(turn_averages)
            plt.axhline(y=avg, color='r', linestyle='--', linewidth=2, label=f'Overall Average: {avg:.4f}')
            plt.legend(fontsize=10)
            
            # Save plot
            output_dir = Path(output_path_trend).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(output_path_trend, dpi=300, bbox_inches='tight')
            logger.info(f"Trend plot saved to: {output_path_trend}")
            logger.info(f"Turn averages: {[f'{ppl:.4f}' for ppl in turn_averages]}")
            
            # Log plot to wandb
            wandb.log({"perplexity_trend_plot": wandb.Image(output_path_trend)})
            plt.close()
        
        # ===== Plot 2: Distribution of all turn perplexities =====
        if all_turn_perplexities:
            import numpy as np
            plt.figure(figsize=(14, 6))
            # Exclude outliers using tighter percentile clipping (1st to 95th percentile)
            lower = np.percentile(all_turn_perplexities, 1)
            upper = np.percentile(all_turn_perplexities, 95)
            clipped = [p for p in all_turn_perplexities if lower <= p <= upper and p > 0]
            plt.hist(clipped, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
            plt.xscale('log')
            plt.xlabel('Perplexity (log scale, 1st-95th percentile)', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.title('Distribution of All Turn Perplexities (Log Scale, Outliers Excluded)', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3, axis='y')

            # Add statistics lines to plot
            avg = sum(clipped) / len(clipped)
            min_ppl = min(clipped)
            max_ppl = max(clipped)
            plt.axvline(x=avg, color='r', linestyle='--', linewidth=2, label=f'Average: {avg:.4f}')
            plt.axvline(x=min_ppl, color='g', linestyle=':', linewidth=2, label=f'Min: {min_ppl:.4f}')
            plt.axvline(x=max_ppl, color='orange', linestyle=':', linewidth=2, label=f'Max: {max_ppl:.4f}')
            plt.legend(fontsize=10)

            # Save plot
            output_dir = Path(output_path_dist).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(output_path_dist, dpi=300, bbox_inches='tight')
            logger.info(f"Distribution plot saved to: {output_path_dist}")

            # Log plot to wandb
            wandb.log({"perplexity_distribution": wandb.Image(output_path_dist)})
            plt.close()
        
    except Exception as e:
        logger.error(f"Failed to create plots: {e}")


def print_results(results: Dict[str, Any]) -> None:
    """
    Print perplexity results in a readable format showing turn, conversation, and dataset levels.
    
    Args:
        results: Results dictionary from calculate_perplexity
    """
    output = "\n" + "="*80 + "\n"
    output += "SEEKER TURN PERPLEXITY EVALUATION RESULTS\n"
    output += "="*80 + "\n"
    output += f"Model: {results['model']}\n"
    
    output += "\n" + "-"*80 + "\n"
    output += "DATASET-LEVEL SUMMARY (Overall average across all conversations):\n"
    output += "-"*80 + "\n"
    ds_level = results['dataset_level']
    output += f"  Total conversations: {ds_level['num_conversations']}\n"
    output += f"  Conversations processed: {ds_level['num_conversations_processed']}\n"
    output += f"  Total seeker turns: {ds_level['num_seeker_turns']}\n"
    output += f"\n  Average Conversation Perplexity: {ds_level['perplexity']['average']:.4f}\n"
    output += f"  Min:     {ds_level['perplexity']['min']:.4f}\n"
    output += f"  Max:     {ds_level['perplexity']['max']:.4f}\n"
    
    output += "\n" + "-"*80 + "\n"
    output += "TURN-LEVEL STATISTICS (Per individual seeker turn):\n"
    output += "-"*80 + "\n"
    turn_level = results['turn_level']
    output += f"  Average Perplexity: {turn_level['perplexity']['average']:.4f}\n"
    output += f"  Min:     {turn_level['perplexity']['min']:.4f}\n"
    output += f"  Max:     {turn_level['perplexity']['max']:.4f}\n"
    
    output += f"\nAverage Loss: {results['average_loss']:.4f}\n"
    output += "="*80 + "\n"
    
    logger.info(output)


if __name__ == "__main__":
    args = get_args()
    
    results = calculate_perplexity(
        args=args
    )
    
    print_results(results)
    
    # Save detailed perplexity data
    output_path = "output/perplexity_detailed.json"
    
    # Save perplexity data in structured format
    save_perplexity_data(results["detailed_data"]["conversation_perplexities_by_conversation"], output_path)
    
    # Save results summary
    summary_path = "output/perplexity_summary.json"
    
    summary_data = {
        "model": results["model"],
        "dataset_level": results["dataset_level"],
        "turn_level": results["turn_level"],
        "average_loss": results["average_loss"]
    }
    
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"Summary saved to: {summary_path}")
    
    # Log summary file to wandb
    artifact = wandb.Artifact(Path(summary_path).name, type="summary")
    artifact.add_file(str(summary_path))
    wandb.log_artifact(artifact)
    
    # Plot perplexity - both trend and distribution
    plot_perplexity(
        results["detailed_data"]["conversation_perplexities_by_conversation"],
        results["detailed_data"]["all_turn_perplexities"]
    )
    
    wandb.finish()
    logger.info("wandb run finished")
