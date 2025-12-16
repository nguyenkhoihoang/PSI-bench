"""
Compare two classification methods: by-conversation vs by-turn.

This script tests whether classify_conversations_batch (full conversation context)
and classify_turns_batch (limited history window) produce consistent labels for
the same conversations.
"""

import yaml
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict
from psibench.eval.ptc.ptc_classification import PTCClassifier
from psibench.data_loader.main_loader import load_eeyore_dataset

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def compute_agreement_metrics(labels1, labels2):
    """Compute agreement metrics between two sets of labels.
    
    Args:
        labels1: List of labels from method 1
        labels2: List of labels from method 2
        
    Returns:
        dict with agreement metrics
    """
    from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
    
    # Ensure same length
    min_len = min(len(labels1), len(labels2))
    labels1 = labels1[:min_len]
    labels2 = labels2[:min_len]
    
    # Overall agreement
    agreement = sum(1 for a, b in zip(labels1, labels2) if a == b)
    total = len(labels1)
    accuracy = agreement / total if total > 0 else 0
    
    # Cohen's Kappa
    kappa = cohen_kappa_score(labels1, labels2)
    
    # Confusion matrix
    categories = ['P', 'T', 'C', 'F']
    cm = confusion_matrix(labels1, labels2, labels=categories)
    
    # Per-class agreement
    class_agreement = {}
    for cat in categories:
        cat_indices = [i for i, label in enumerate(labels1) if label == cat]
        if cat_indices:
            cat_agree = sum(1 for i in cat_indices if labels1[i] == labels2[i])
            class_agreement[cat] = cat_agree / len(cat_indices)
        else:
            class_agreement[cat] = None
    
    # Classification report
    report = classification_report(labels1, labels2, labels=categories, 
                                   target_names=categories, zero_division=0,
                                   output_dict=True)
    
    return {
        'accuracy': accuracy,
        'kappa': kappa,
        'confusion_matrix': cm,
        'class_agreement': class_agreement,
        'classification_report': report,
        'n_samples': total,
        'n_agreements': agreement,
        'n_disagreements': total - agreement
    }


def plot_confusion_matrix(cm, categories, output_path):
    """Plot confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=categories, yticklabels=categories,
                ax=ax, cbar_kws={'label': 'Count'})
    
    ax.set_xlabel('By-Turn Classification', fontsize=12, fontweight='bold')
    ax.set_ylabel('By-Conversation Classification', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix: Method Comparison', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {output_path}")


def plot_progression_comparison(conv_results, output_path, max_turns=15):
    """Plot side-by-side progression for each conversation."""
    n_convs = len(conv_results)
    fig, axes = plt.subplots(n_convs, 1, figsize=(14, 4 * n_convs))
    
    if n_convs == 1:
        axes = [axes]
    
    ptc_map = {'P': 1, 'T': 2, 'C': 3, 'F': 0}
    
    for idx, (conv_id, data) in enumerate(conv_results.items()):
        ax = axes[idx]
        
        # Prepare data
        turns = list(range(len(data['by_conv'])))[:max_turns]
        by_conv_values = [ptc_map.get(data['by_conv'][i], 0) for i in turns]
        by_turn_values = [ptc_map.get(data['by_turn'][i], 0) for i in turns]
        
        # Plot
        ax.plot(turns, by_conv_values, 'o-', label='By-Conversation', 
               linewidth=2, markersize=8, alpha=0.7, color='steelblue')
        ax.plot(turns, by_turn_values, 's--', label='By-Turn', 
               linewidth=2, markersize=8, alpha=0.7, color='coral')
        
        # Highlight disagreements
        for i in turns:
            if data['by_conv'][i] != data['by_turn'][i]:
                ax.axvspan(i-0.3, i+0.3, alpha=0.2, color='red')
        
        ax.set_xlabel('Turn Index', fontsize=11)
        ax.set_ylabel('Classification', fontsize=11)
        ax.set_title(f'Conversation {conv_id} - Agreement: {data["agreement"]:.1%}', 
                    fontsize=12, fontweight='bold')
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(['F', 'P', 'T', 'C'])
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved progression comparison to {output_path}")


def plot_agreement_by_position(conv_results, output_path):
    """Plot agreement rate by turn position."""
    # Collect data by position
    position_data = defaultdict(lambda: {'agree': 0, 'total': 0})
    
    for conv_id, data in conv_results.items():
        for i, (label1, label2) in enumerate(zip(data['by_conv'], data['by_turn'])):
            position_data[i]['total'] += 1
            if label1 == label2:
                position_data[i]['agree'] += 1
    
    # Calculate agreement rates
    positions = sorted(position_data.keys())
    agreement_rates = [position_data[p]['agree'] / position_data[p]['total'] 
                       for p in positions]
    totals = [position_data[p]['total'] for p in positions]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Agreement rate
    ax1.plot(positions, agreement_rates, 'o-', linewidth=2, markersize=6, color='purple')
    ax1.axhline(y=np.mean(agreement_rates), color='red', linestyle='--', 
                label=f'Mean: {np.mean(agreement_rates):.2%}')
    ax1.set_xlabel('Turn Position', fontsize=12)
    ax1.set_ylabel('Agreement Rate', fontsize=12)
    ax1.set_title('Classification Agreement by Turn Position', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1.05])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Sample count
    ax2.bar(positions, totals, alpha=0.6, color='gray')
    ax2.set_xlabel('Turn Position', fontsize=12)
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_title('Sample Count per Position', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved agreement by position to {output_path}")


def plot_label_distribution(conv_results, output_path):
    """Compare label distributions between methods."""
    # Collect all labels
    by_conv_all = []
    by_turn_all = []
    
    for data in conv_results.values():
        by_conv_all.extend(data['by_conv'])
        by_turn_all.extend(data['by_turn'])
    
    # Count labels
    categories = ['P', 'T', 'C', 'F']
    by_conv_counts = [by_conv_all.count(cat) for cat in categories]
    by_turn_counts = [by_turn_all.count(cat) for cat in categories]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, by_conv_counts, width, label='By-Conversation', alpha=0.8)
    bars2 = ax.bar(x + width/2, by_turn_counts, width, label='By-Turn', alpha=0.8)
    
    ax.set_xlabel('Classification Label', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Label Distribution Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved label distribution to {output_path}")


def analyze_disagreements(conv_results, output_path, top_n=20):
    """Analyze and save examples of disagreements."""
    disagreements = []
    
    for conv_id, data in conv_results.items():
        for i, (label1, label2) in enumerate(zip(data['by_conv'], data['by_turn'])):
            if label1 != label2:
                disagreements.append({
                    'conversation_id': conv_id,
                    'turn_index': i,
                    'by_conversation': label1,
                    'by_turn': label2,
                    'content': data['contents'][i] if i < len(data['contents']) else 'N/A'
                })
    
    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_disagreements': len(disagreements),
            'examples': disagreements[:top_n]
        }, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(disagreements)} disagreements to {output_path}")
    return disagreements


def main():
    """Main comparison test."""
    # Configuration
    N_CONVERSATIONS = 100
    NUM_MESSAGES_HISTORY = 6  # For by-turn classification
    
    print("=" * 70)
    print("CLASSIFICATION METHOD COMPARISON TEST")
    print("=" * 70)
    print(f"\nSettings:")
    print(f"  Number of conversations: {N_CONVERSATIONS}")
    print(f"  History window (by-turn): {NUM_MESSAGES_HISTORY}")
    
    # Load configuration
    config_path = Path("configs/default.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize classifier
    print("\nInitializing classifier...")
    judge = PTCClassifier(config, debug=False)
    print(f"  Model: {judge.model_name}")
    print(f"  Temperature: {judge.temperature}")
    
    # Load conversations with exactly 12 patient turns
    print(f"\nLoading conversations from ESC dataset...")
    df = load_eeyore_dataset(dataset_type="esc", indices=list(range(N_CONVERSATIONS)))
    
    # Filter to conversations with exactly 12 patient turns
    conversations = []
    conversation_indices = []
    for idx in df.index:
        messages = df.loc[idx]["messages"]
        patient_turns = sum(1 for msg in messages if msg.get('role') == 'assistant' and msg.get('content', '').strip())
        if patient_turns == 12:
            conversations.append(messages)
            conversation_indices.append(idx)
    
    print(f"Loaded {len(conversations)} conversations with exactly 12 patient turns")
    print(f"Conversation indices: {conversation_indices}")
    
    # Method 1: By-conversation classification
    print("\n" + "=" * 70)
    print("METHOD 1: By-Conversation Classification")
    print("=" * 70)
    by_conv_results = judge.classify_conversations_batch(conversations)
    print(f"Classified {len(by_conv_results)} conversations")
    
    # Method 2: By-turn classification
    print("\n" + "=" * 70)
    print("METHOD 2: By-Turn Classification")
    print("=" * 70)
    by_turn_results = judge.classify_turns_batch(conversations, num_messages=NUM_MESSAGES_HISTORY)
    print(f"Classified {len(by_turn_results)} conversations")
    
    # Organize results by conversation
    print("\n" + "=" * 70)
    print("COMPARING RESULTS")
    print("=" * 70)
    
    conv_results = {}
    all_by_conv = []
    all_by_turn = []
    
    for i in range(len(conversations)):
        # Extract labels
        by_conv_labels = [turn['classification'] for turn in by_conv_results[i]]
        by_turn_labels = [turn['classification'] for turn in by_turn_results[i]]
        
        # Get full content from original messages
        patient_messages = [msg for msg in conversations[i] 
                           if msg.get('role') == 'assistant' and msg.get('content', '').strip()]
        contents = [msg.get('content', 'N/A') for msg in patient_messages]
        
        # Compute agreement for this conversation
        min_len = min(len(by_conv_labels), len(by_turn_labels))
        by_conv_labels = by_conv_labels[:min_len]
        by_turn_labels = by_turn_labels[:min_len]
        
        agreement = sum(1 for a, b in zip(by_conv_labels, by_turn_labels) if a == b)
        agreement_rate = agreement / min_len if min_len > 0 else 0
        
        conv_results[i] = {
            'by_conv': by_conv_labels,
            'by_turn': by_turn_labels,
            'contents': contents,
            'agreement': agreement_rate,
            'n_turns': min_len
        }
        
        all_by_conv.extend(by_conv_labels)
        all_by_turn.extend(by_turn_labels)
        
        print(f"\nConversation {i}:")
        print(f"  Turns: {min_len}")
        print(f"  Agreement: {agreement}/{min_len} ({agreement_rate:.1%})")
    
    # Overall agreement metrics
    print("\n" + "=" * 70)
    print("OVERALL METRICS")
    print("=" * 70)
    
    metrics = compute_agreement_metrics(all_by_conv, all_by_turn)
    
    print(f"\nTotal samples: {metrics['n_samples']}")
    print(f"Agreements: {metrics['n_agreements']} ({metrics['accuracy']:.1%})")
    print(f"Disagreements: {metrics['n_disagreements']} ({1-metrics['accuracy']:.1%})")
    print(f"Cohen's Kappa: {metrics['kappa']:.3f}")
    
    print("\nPer-class agreement:")
    for cat in ['P', 'T', 'C', 'F']:
        rate = metrics['class_agreement'][cat]
        if rate is not None:
            print(f"  {cat}: {rate:.1%}")
        else:
            print(f"  {cat}: N/A (no samples)")
    
    print("\nConfusion Matrix (By-Conv vs By-Turn):")
    categories = ['P', 'T', 'C', 'F']
    print(f"{'':>8} " + " ".join(f"{cat:>6}" for cat in categories))
    for i, cat in enumerate(categories):
        print(f"{cat:>8} " + " ".join(f"{metrics['confusion_matrix'][i][j]:>6}" 
                                       for j in range(len(categories))))
    
    # Create output directory
    output_dir = Path("output/classification_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations and analysis
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    plot_confusion_matrix(metrics['confusion_matrix'], categories, 
                         output_dir / 'confusion_matrix.png')
    
    plot_progression_comparison(conv_results, 
                                output_dir / 'progression_comparison.png')
    
    plot_agreement_by_position(conv_results, 
                               output_dir / 'agreement_by_position.png')
    
    plot_label_distribution(conv_results, 
                           output_dir / 'label_distribution.png')
    
    disagreements = analyze_disagreements(conv_results, 
                                         output_dir / 'disagreements.json')
    
    # Save summary report
    print("\nSaving summary report...")
    with open(output_dir / 'comparison_summary.txt', 'w') as f:
        f.write("CLASSIFICATION METHOD COMPARISON REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Settings:\n")
        f.write(f"  Number of conversations: {N_CONVERSATIONS}\n")
        f.write(f"  History window (by-turn): {NUM_MESSAGES_HISTORY}\n")
        f.write(f"  Model: {judge.model_name}\n")
        f.write(f"  Temperature: {judge.temperature}\n\n")
        
        f.write("Overall Metrics:\n")
        f.write(f"  Total samples: {metrics['n_samples']}\n")
        f.write(f"  Accuracy: {metrics['accuracy']:.1%}\n")
        f.write(f"  Cohen's Kappa: {metrics['kappa']:.3f}\n")
        f.write(f"  Disagreements: {metrics['n_disagreements']}\n\n")
        
        f.write("Per-class Agreement:\n")
        for cat in ['P', 'T', 'C', 'F']:
            rate = metrics['class_agreement'][cat]
            if rate is not None:
                f.write(f"  {cat}: {rate:.1%}\n")
            else:
                f.write(f"  {cat}: N/A\n")
        
        f.write("\nPer-conversation Agreement:\n")
        for conv_id, data in conv_results.items():
            f.write(f"  Conversation {conv_id}: {data['agreement']:.1%} ({data['n_turns']} turns)\n")
    
    print(f"\n{'='*70}")
    print(f"RESULTS SAVED TO: {output_dir}")
    print(f"{'='*70}")
    print(f"\nGenerated files:")
    print(f"  - comparison_summary.txt: Overall statistics and metrics")
    print(f"  - confusion_matrix.png: Heatmap of label confusions")
    print(f"  - progression_comparison.png: Side-by-side progressions")
    print(f"  - agreement_by_position.png: Agreement rate by turn position")
    print(f"  - label_distribution.png: Distribution comparison")
    print(f"  - disagreements.json: Examples of disagreements")


if __name__ == "__main__":
    main()
