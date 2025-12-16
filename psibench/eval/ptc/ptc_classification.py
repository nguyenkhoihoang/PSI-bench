"""
PTC Classification: Classify patient turns as Problem, Transition, or Change.

This module uses an LLM-judge to classify each patient turn in therapy conversations
according to the Problem-Transition-Change framework.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
import yaml
from litellm import batch_completion
from json_repair import repair_json

from psibench.prompts.judge_prompt import create_ptc_judge_conversation_prompt
from psibench.data_loader.main_loader import load_eeyore_dataset

load_dotenv()

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class PTCClassifier:
    """Judge for Problem-Transition-Change framework classification."""
    
    def __init__(self, config: Dict[str, Any], debug: bool = False):
        """Initialize the PTC judge.
        
        Args:
            config: Configuration dictionary containing eval.ptc_classifier settings
            debug: Enable debug logging (default: False)
        """
        # Store model settings for batch completion
        self.debug = debug
        judge_config = config.get("eval", {}).get("ptc_classifier", {})
        self.model_name = judge_config.get("model")
        self.temperature = judge_config.get("temperature", 0.3)
        self.history_limit = judge_config.get("history_limit", 20)
        
        # Get API settings
        if judge_config.get("api_base"):
            self.api_base = judge_config.get("api_base")
            self.api_key = "sk-no-key-required"
        else:
            self.api_base = os.getenv("OPENAI_BASE_URL")
            self.api_key = os.getenv("OPENAI_API_KEY")
            
    def _format_history(self, history: list[Dict[str, str]]) -> str:
        """Format conversation history for the prompt.
        Switch to Therapist/Assistant labels, makes life easier for judges
        
        Args:
            history: List of conversation messages
            
        Returns:
            Formatted conversation history string
        """
        formatted = []
        for msg in history:
            # Skip empty messages
            if not msg.get("content", "").strip():
                continue
            role = "THERAPIST" if msg["role"] == "user" else "PATIENT"
            content = msg["content"]
            formatted.append(f"{role}: {content}")
            if len(formatted) == self.history_limit:
                break
            
        return "\n".join(formatted)

    def classify_conversations_batch(self, conversations: List[List[Dict[str, str]]]) -> List[List[Dict[str, Any]]]:
        """Batch classify multiple conversations using litellm.batch_completion.
        
        Args:
            conversations: List of conversations, each being a list of messages
            
        Returns:
            List of classification results for each conversation
        """
        # Get the prompt template
        conversation_prompt = create_ptc_judge_conversation_prompt()
        
        # Build messages for each conversation
        all_messages = []
        for messages in conversations:
            conversation_str = self._format_history(messages)
            # Debug: print approximate context size
            if self.debug:
                approx_tokens = max(1, len(conversation_str) // 4)
                print(f"Judge prompt context: ~{approx_tokens} tokens, {len(conversation_str)} chars")
            
            # Format prompt messages
            formatted_prompt = conversation_prompt.format_messages(conversation=conversation_str)
            # Convert to litellm format - map 'human' to 'user' for OpenAI compatibility
            litellm_messages = []
            for i, msg in enumerate(formatted_prompt):
                if hasattr(msg, 'type'):
                    role = msg.type
                    # Fix: Map 'human' to 'user' for OpenAI API
                    if role == 'human':
                        role = 'user'
                    elif role not in ['system', 'assistant', 'user', 'function', 'tool', 'developer']:
                        role = 'system' if i == 0 else 'user'
                else:
                    role = 'system' if i == 0 else 'user'
                
                litellm_messages.append({
                    "role": role,
                    "content": msg.content
                })
            all_messages.append(litellm_messages)
        if self.debug:
            print(f"Prepared {len(all_messages)} judge calls; first message roles: {[m['role'] for m in all_messages[0]] if all_messages else []}")
        
        # Batch completion call with error handling
        try:
            responses = batch_completion(
                model=self.model_name,
                messages=all_messages,
                temperature=self.temperature,
                api_key=self.api_key,
                api_base=self.api_base,
            )
        except Exception as e:
            print(f"Error in batch_completion call: {e}")
            return [[] for _ in conversations]
        
        # Parse all responses
        results = []
        for i, response in enumerate(responses):
            try:
                # Handle different response formats
                if isinstance(response, str):
                    response_text = response.strip()
                elif hasattr(response, 'choices') and len(getattr(response, 'choices', [])) > 0:
                    content = response.choices[0].message.content
                    if content is None:
                        if self.debug:
                            print(f"Batch {i}: choices[0].message.content is None. Full response: {getattr(response, 'dict', lambda: response)() if hasattr(response, 'dict') else response}")
                        results.append([])
                        continue
                    response_text = content.strip()
                elif hasattr(response, 'message') and hasattr(response.message, 'content'):
                    content = response.message.content
                    if content is None:
                        if self.debug:
                            print(f"Batch {i}: message.content is None. Full response: {getattr(response, 'dict', lambda: response)() if hasattr(response, 'dict') else response}")
                        results.append([])
                        continue
                    response_text = content.strip()
                elif hasattr(response, 'content'):
                    content = response.content
                    if content is None:
                        if self.debug:
                            print(f"Batch {i}: content is None. Full response: {getattr(response, 'dict', lambda: response)() if hasattr(response, 'dict') else response}")
                        results.append([])
                        continue
                    response_text = content.strip()
                else:
                    raise ValueError(f"Unexpected response format: {type(response)}")
                
                classifications = repair_json(response_text, return_objects=True)
                # Normalize common shapes: dict with key, list of dicts, list of strings
                if isinstance(classifications, dict):
                    # Try common keys
                    for key in ('classifications', 'turns', 'labels'):
                        if key in classifications and isinstance(classifications[key], list):
                            classifications = classifications[key]
                            break
                if isinstance(classifications, list):
                    # If list of strings, convert to dicts
                    if all(isinstance(x, str) for x in classifications):
                        classifications = [{'classification': x} for x in classifications]
                    # If list-of-lists, unwrap first
                    elif len(classifications) > 0 and all(isinstance(x, list) for x in classifications):
                        classifications = classifications[0]
                
                # Filter invalid items and log
                valid_items = []
                for idx, item in enumerate(classifications if isinstance(classifications, list) else []):
                    if isinstance(item, dict) and 'classification' in item and isinstance(item['classification'], str):
                        valid_items.append(item)
                    else:
                        if self.debug:
                            print(f"Batch {i}: Dropping invalid item at pos {idx}: {item}")
                classifications = valid_items
                results.append(classifications)
            except Exception as e:
                print(f"Error parsing batch response {i}: {e}")
                print(f"Response type: {type(response)}")
                results.append([])  # Empty list on error
        
        return results

    def classify_turns_batch(self, conversations: List[List[Dict[str, str]]], num_messages: int = 6) -> List[List[Dict[str, Any]]]:
        """Batch classify individual patient turns with limited history context.
        
        Args:
            conversations: List of conversations, each being a list of messages
            num_messages: Number of previous messages to include as history (default: 6)
            
        Returns:
            List of classification results for each conversation, where each result is a list of 
            {"content": str, "classification": str, "turn_index": int} for each patient turn
        """
        from psibench.prompts.judge_prompt import create_ptc_judge_single_turn_prompt
        
        # Get the prompt template
        single_turn_prompt = create_ptc_judge_single_turn_prompt()
        
        # Build all classification tasks across all conversations
        all_messages = []
        task_metadata = []  # Track which conversation and turn each task belongs to
        
        for conv_idx, messages in enumerate(conversations):
            # Find all patient turns (role == 'assistant')
            patient_indices = [i for i, msg in enumerate(messages) 
                             if msg.get('role') == 'assistant' and msg.get('content', '').strip()]
            
            for patient_idx in patient_indices:
                # Get history: up to num_messages before this patient turn
                history_start = max(0, patient_idx - num_messages)
                history_messages = messages[history_start:patient_idx]
                history_str = self._format_history(history_messages) if history_messages else "(No previous history)"
                
                # Current patient message
                current_message = messages[patient_idx]['content']
                
                # Format prompt
                formatted_prompt = single_turn_prompt.format_messages(
                    history=history_str,
                    current_message=current_message
                )
                
                # Convert to litellm format
                litellm_messages = []
                for i, msg in enumerate(formatted_prompt):
                    if hasattr(msg, 'type'):
                        role = msg.type
                        if role == 'human':
                            role = 'user'
                        elif role not in ['system', 'assistant', 'user', 'function', 'tool', 'developer']:
                            role = 'system' if i == 0 else 'user'
                    else:
                        role = 'system' if i == 0 else 'user'
                    
                    litellm_messages.append({
                        "role": role,
                        "content": msg.content
                    })
                
                all_messages.append(litellm_messages)
                task_metadata.append({
                    'conv_idx': conv_idx,
                    'turn_idx': patient_idx,
                    'content': current_message
                })
        
        if self.debug:
            print(f"Prepared {len(all_messages)} single-turn classification tasks across {len(conversations)} conversations")
            # Log first few request structures for debugging
            for task_idx in range(min(2, len(all_messages))):
                print(f"\n--- Sample request {task_idx} ---")
                print(f"Number of messages: {len(all_messages[task_idx])}")
                for msg_idx, msg in enumerate(all_messages[task_idx]):
                    content_preview = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                    print(f"  Message {msg_idx} (role={msg['role']}): {content_preview}")
        
        # Batch completion call
        try:
            if self.debug:
                print(f"\n--- Sending batch completion request ---")
                print(f"Model: {self.model_name}")
                print(f"Temperature: {self.temperature}")
                print(f"API base: {self.api_base}")
            
            responses = batch_completion(
                model=self.model_name,
                messages=all_messages,
                temperature=self.temperature,
                api_key=self.api_key,
                api_base=self.api_base,
            )
            
            if self.debug:
                print(f"Batch completion succeeded. Received {len(responses)} responses")
        except Exception as e:
            print(f"Error in batch_completion call: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
            return [[] for _ in conversations]
        
        # Parse responses and organize by conversation
        conversation_results = [[] for _ in conversations]
        
        for i, (response, metadata) in enumerate(zip(responses, task_metadata)):
            try:
                # Extract response text
                if isinstance(response, str):
                    response_text = response.strip()
                elif hasattr(response, 'choices') and len(getattr(response, 'choices', [])) > 0:
                    content = response.choices[0].message.content
                    if content is None:
                        if self.debug:
                            print(f"Task {i}: choices[0].message.content is None; skipping")
                        continue
                    else:
                        response_text = content.strip()
                elif hasattr(response, 'message') and hasattr(response.message, 'content'):
                    content = response.message.content
                    if content is None:
                        if self.debug:
                            print(f"Task {i}: message.content is None; skipping")
                        continue
                    else:
                        response_text = content.strip()
                elif hasattr(response, 'content'):
                    content = response.content
                    if content is None:
                        if self.debug:
                            print(f"Task {i}: content is None; skipping")
                        continue
                    else:
                        response_text = content.strip()
                else:
                    raise ValueError(f"Unexpected response format: {type(response)}")
                
                # Parse classification (should be single letter)
                if 'response_text' in locals():
                    classification = response_text.upper().strip()
                    # Extract just the classification letter if there's extra text
                    for letter in ['P', 'T', 'C', 'F']:
                        if letter in classification:
                            classification = letter
                            break
                    else:
                        if self.debug:
                            print(f"Task {i}: Could not find valid classification in '{response_text}'; skipping")
                        continue
                
                # Add to appropriate conversation's results
                conv_idx = metadata['conv_idx']
                conversation_results[conv_idx].append({
                    'content': metadata['content'],
                    'classification': classification,
                    'turn_index': metadata['turn_idx']
                })
                
            except Exception as e:
                if self.debug:
                    print(f"Error parsing task {i} response: {e}; skipping")
                continue
        
        return conversation_results


def analyze_conversations(
    judge: PTCClassifier,
    output_dir: Path,
    batch_size: int = 1,
    dataset: str = None,
    indices: List[int] = None,
    data_dir: Path = None,
    session_files: List[Path] = None,
    single_turn: bool = False,
    num_messages: int = 6,
    exact_turns: int = None,
) -> pd.DataFrame:
    """Analyze conversations from either real dataset or synthetic files.
    
    Args:
        judge: PTCClassifier instance
        output_dir: Directory to save results
        batch_size: Number of conversations to process in parallel
        dataset: Dataset type for real conversations (e.g., 'esc', 'hope'). If provided, loads from eeyore dataset.
        indices: List of conversation indices (required if dataset is provided)
        data_dir: Directory containing synthetic conversation JSON files (alternative to dataset/indices)
        single_turn: If True, use single-turn classification with limited history (default: False)
        num_messages: Number of previous messages to include as history for single-turn mode (default: 6)
        exact_turns: If specified, only include conversations with exactly this many patient turns (default: None)
        
    Returns:
        DataFrame with analysis results
    """
    results = []
    all_conversations = []
    conversation_ids = []  # Store identifiers (indices or filenames)
    mismatch_count = 0  # Track classification mismatches
    
    # Load conversations based on source
    if dataset is not None and indices is not None:
        # Load from real dataset
        eeyore_df = load_eeyore_dataset(dataset_type=dataset, indices=indices)
        print(f"Loaded {len(eeyore_df)} real conversations from {dataset} dataset")
        
        for idx, row in eeyore_df.iterrows():
            messages = row["messages"]
            # Filter by exact patient turn count if specified
            if exact_turns is not None:
                patient_turn_count = sum(1 for msg in messages 
                                       if msg.get('role') == 'assistant' and msg.get('content', '').strip())
                if patient_turn_count != exact_turns:
                    continue
            all_conversations.append(messages)
            conversation_ids.append(('real', idx))
        
        print(f"Analyzing {len(all_conversations)} conversations (filtered to {exact_turns} patient turns)" if exact_turns else f"Analyzing {len(all_conversations)} conversations")
            
    elif data_dir is not None:
        # Load from synthetic files
        # If a specific list of session_files is provided, use that; otherwise glob from data_dir
        if session_files is None:
            session_files = sorted(data_dir.glob('session_*.json'))
        print(f"Loaded {len(session_files)} conversations from {data_dir}")
        
        for session_file in session_files:
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                messages = data.get('messages', [])
                # Filter by exact patient turn count if specified
                if exact_turns is not None:
                    patient_turn_count = sum(1 for msg in messages 
                                           if msg.get('role') == 'assistant' and msg.get('content', '').strip())
                    if patient_turn_count != exact_turns:
                        continue
                all_conversations.append(messages)
                conversation_ids.append(('synthetic', session_file))
            except Exception as e:
                print(f"Error loading {session_file}: {e}")
                continue
        
        print(f"Analyzing {len(all_conversations)} conversations (filtered to {exact_turns} patient turns)" if exact_turns else f"Analyzing {len(all_conversations)} conversations")
    else:
        raise ValueError("Must provide either (dataset and indices) or data_dir")
    
    print(f"Batch size: {batch_size}")
    
    # Process in batches
    for batch_start in tqdm(range(0, len(all_conversations), batch_size), desc="Classifying conversations"):
        batch_end = min(batch_start + batch_size, len(all_conversations))
        batch_conversations = all_conversations[batch_start:batch_end]
        batch_ids = conversation_ids[batch_start:batch_end]
        
        try:
            if single_turn:
                classifications_list = judge.classify_turns_batch(batch_conversations, num_messages=num_messages)
            else:
                classifications_list = judge.classify_conversations_batch(batch_conversations)
            
            # Process results
            for conv_id, classifications, messages in zip(batch_ids, classifications_list, batch_conversations):
                # Count expected non-empty patient turns
                expected_patient_turns = min(judge.history_limit // 2 ,sum(1 for msg in messages 
                                              if msg.get('role') == 'assistant' and msg.get('content', '').strip()))
                
                # Validate classification count
                actual_classifications = len(classifications)
                conv_identifier = f"conversation {conv_id[1]}" if conv_id[0] == 'real' else conv_id[1].name
                has_mismatch = actual_classifications != expected_patient_turns
                if has_mismatch:
                    mismatch_count += 1
                    print(f"WARNING: Classification mismatch for {conv_identifier}: expected {expected_patient_turns} patient turns, got {actual_classifications} classifications")
                    # Save debug output
                    debug_filename = f"debug_{conv_id[0]}_{Path(str(conv_id[1])).stem if conv_id[0] == 'synthetic' else conv_id[1]}.json"
                    debug_path = output_dir / "debug" / debug_filename
                    debug_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    debug_output = {
                        "conversation_id": conv_identifier,
                        "expected_patient_turns": expected_patient_turns,
                        "actual_classifications": actual_classifications,
                        "input_messages": messages,
                        "output_classifications": classifications
                    }
                    
                    with open(debug_path, 'w', encoding='utf-8') as f:
                        json.dump(debug_output, f, indent=2, ensure_ascii=False)
                    
                    print(f"Debug output saved to {debug_path}")
                
                ptc_counts = Counter([c['classification'] for c in classifications])
                total = len(classifications)
                non_filler_total = sum([ptc_counts.get(cat, 0) for cat in ['P', 'T', 'C']])
                
                # Build result dict based on source type
                result = { 
                        'total_patient_turns': total,
                        'non_filler_turns': non_filler_total,
                        'P_count': ptc_counts.get('P', 0),
                        'T_count': ptc_counts.get('T', 0),
                        'C_count': ptc_counts.get('C', 0),
                        'F_count': ptc_counts.get('F', 0),
                        'P_ratio': ptc_counts.get('P', 0) / non_filler_total if non_filler_total > 0 else 0,
                        'T_ratio': ptc_counts.get('T', 0) / non_filler_total if non_filler_total > 0 else 0,
                        'C_ratio': ptc_counts.get('C', 0) / non_filler_total if non_filler_total > 0 else 0,
                        'F_ratio': ptc_counts.get('F', 0) / total if total > 0 else 0,
                        'classifications': classifications,
                        'has_mismatch': has_mismatch
                        }
                detail_filename = f"session_{conv_id[1]}_ptc.json"
                
                results.append(result)
                
                # Save detailed classification for this conversation
                detail_path = output_dir / detail_filename
                detail_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(detail_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                    
        except Exception as e:
            print(f"Error processing batch starting at {batch_start}: {e}")
            continue
    
    total_conversations = len(results)
    print(f"\n{'='*60}")
    print(f"Classification Summary: {mismatch_count}/{total_conversations} conversations had mismatches")
    print(f"{'='*60}")
    # Write summary to text file
    summary_path = output_dir / "classification_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"Total conversations: {total_conversations}\n")
        f.write(f"Conversations with mismatches: {mismatch_count}\n")
        f.write(f"Mismatch rate: {mismatch_count/total_conversations*100:.1f}%\n")
    print(f"Summary saved to {summary_path}")
    return pd.DataFrame(results)


def compare_distributions(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, output_dir: Path, judge: PTCClassifier = None):
    """Compare PTC distributions between real and synthetic data.
    
    Args:
        real_df: DataFrame with real conversation analysis
        synthetic_df: DataFrame with synthetic conversation analysis
        output_dir: Directory to save comparison results
        judge: PTCClassifier instance for model settings
    """
    # Calculate aggregate statistics (P, T, C averaged over non-filler turns only)
    real_stats = {
        'P_mean': real_df['P_ratio'].mean(),  # Already calculated over non-filler turns
        'T_mean': real_df['T_ratio'].mean(),
        'C_mean': real_df['C_ratio'].mean(),
        'F_mean': real_df['F_ratio'].mean(),  # Calculated over all turns
        'P_std': real_df['P_ratio'].std(),
        'T_std': real_df['T_ratio'].std(),
        'C_std': real_df['C_ratio'].std(),
        'F_std': real_df['F_ratio'].std(),
        'total_turns': real_df['total_patient_turns'].sum(),
        'non_filler_turns': real_df['non_filler_turns'].sum()
    }
    
    synthetic_stats = {
        'P_mean': synthetic_df['P_ratio'].mean(),  # Already calculated over non-filler turns
        'T_mean': synthetic_df['T_ratio'].mean(),
        'C_mean': synthetic_df['C_ratio'].mean(),
        'F_mean': synthetic_df['F_ratio'].mean(),  # Calculated over all turns
        'P_std': synthetic_df['P_ratio'].std(),
        'T_std': synthetic_df['T_ratio'].std(),
        'C_std': synthetic_df['C_ratio'].std(),
        'F_std': synthetic_df['F_ratio'].std(),
        'total_turns': synthetic_df['total_patient_turns'].sum(),
        'non_filler_turns': synthetic_df['non_filler_turns'].sum()
    }
    
    # Create comparison DataFrame (P, T, C are proportions of non-filler turns)
    comparison = pd.DataFrame({
        'Real_Mean': [real_stats['P_mean'], real_stats['T_mean'], real_stats['C_mean']],
        'Real_Std': [real_stats['P_std'], real_stats['T_std'], real_stats['C_std']],
        'Synthetic_Mean': [synthetic_stats['P_mean'], synthetic_stats['T_mean'], synthetic_stats['C_mean']],
        'Synthetic_Std': [synthetic_stats['P_std'], synthetic_stats['T_std'], synthetic_stats['C_std']],
        'Difference': [
            synthetic_stats['P_mean'] - real_stats['P_mean'],
            synthetic_stats['T_mean'] - real_stats['T_mean'],
            synthetic_stats['C_mean'] - real_stats['C_mean']
        ]
    }, index=['Problem (P)', 'Transition (T)', 'Change (C)'])
    
    with open(output_dir / 'ptc_comparison.txt', 'w') as f:
        f.write("PTC Distribution Comparison: Real vs Synthetic\n")
        f.write("=" * 70 + "\n\n")
        
        # Write model settings if judge is provided
        if judge:
            f.write("--- Model Settings ---\n")
            f.write(f"Model: {judge.model_name}\n")
            f.write(f"Temperature: {judge.temperature}\n")
            f.write("\n")
        
        f.write("Note: P, T, C ratios are calculated over non-filler turns only\n")
        f.write("=" * 70 + "\n\n")
        f.write(comparison.to_string())
        f.write(f"\n\n--- Turn Statistics ---")
        f.write(f"\nTotal Real Turns: {real_stats['total_turns']}")
        f.write(f"\nNon-Filler Real Turns: {real_stats['non_filler_turns']}")
        f.write(f"\nReal Filler Ratio: {real_stats['F_mean']:.3f}")
        f.write(f"\n\nTotal Synthetic Turns: {synthetic_stats['total_turns']}")
        f.write(f"\nNon-Filler Synthetic Turns: {synthetic_stats['non_filler_turns']}")
        f.write(f"\nSynthetic Filler Ratio: {synthetic_stats['F_mean']:.3f}\n")
    
    print("\nPTC Distribution Comparison:")
    print(comparison)
    
    return comparison


def get_turn_classifications(df: pd.DataFrame) -> pd.DataFrame:
    """Extract turn-by-turn classifications from analysis results.
    
    Args:
        df: DataFrame with conversation analysis including classifications
        
    Returns:
        DataFrame with columns: conversation_id, turn_index, classification
    """
    turn_data = []
    
    for idx, row in df.iterrows():
        classifications = row['classifications']
        for turn_idx, turn in enumerate(classifications):
            turn_data.append({
                'conversation_id': idx,
                'turn_index': turn_idx,
                'classification': turn['classification']
            })
    
    return pd.DataFrame(turn_data)


def visualize_distributions(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, 
                           output_dir: Path, turn_threshold: int = 12, exact_turns: int = None):
    """Create visualizations comparing PTC distributions.
    
    Args:
        real_df: DataFrame with real conversation analysis
        synthetic_df: DataFrame with synthetic conversation analysis
        output_dir: Directory to save plots
        exact_turns: If provided, label plots to reflect exact patient-turn filtering
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate average counts as ratios of total turns (including fillers)
    real_p_ratio = real_df['P_count'].sum() / real_df['total_patient_turns'].sum()
    real_t_ratio = real_df['T_count'].sum() / real_df['total_patient_turns'].sum()
    real_c_ratio = real_df['C_count'].sum() / real_df['total_patient_turns'].sum()
    real_f_ratio = real_df['F_count'].sum() / real_df['total_patient_turns'].sum()
    
    synthetic_p_ratio = synthetic_df['P_count'].sum() / synthetic_df['total_patient_turns'].sum()
    synthetic_t_ratio = synthetic_df['T_count'].sum() / synthetic_df['total_patient_turns'].sum()
    synthetic_c_ratio = synthetic_df['C_count'].sum() / synthetic_df['total_patient_turns'].sum()
    synthetic_f_ratio = synthetic_df['F_count'].sum() / synthetic_df['total_patient_turns'].sum()
    
    real_means = [real_p_ratio, real_t_ratio, real_c_ratio]
    synthetic_means = [synthetic_p_ratio, synthetic_t_ratio, synthetic_c_ratio]
    real_filler_mean = real_f_ratio
    synthetic_filler_mean = synthetic_f_ratio
    
    # 1. Stacked bar chart of average P, T, C, F ratios
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    colors = ['#e74c3c', '#f39c12', '#27ae60', '#95a5a6']
    labels = ['Problem', 'Transition', 'Change', 'Filler']
    
    # Real stacked bar
    bottom = 0
    real_values = real_means + [real_filler_mean]
    for i, (value, color, label) in enumerate(zip(real_values, colors, labels)):
        axes[0].bar(0, value, bottom=bottom, label=label, alpha=0.8, color=color, width=0.5)
        # Add value label
        if value > 0.02:  # Only show label if segment is large enough
            axes[0].text(0, bottom + value/2, f'{value:.2f}', 
                    ha='center', va='center', fontsize=12, fontweight='bold')
        bottom += value
    
    axes[0].set_xlim([-0.5, 0.5])
    axes[0].set_ylim([0, 1])
    axes[0].set_ylabel('Ratio', fontsize=12)
    axes[0].set_title('Real Conversations', fontsize=14, fontweight='bold')
    axes[0].set_xticks([])
    axes[0].legend(loc='upper right')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Synthetic stacked bar
    bottom = 0
    synthetic_values = synthetic_means + [synthetic_filler_mean]
    for i, (value, color, label) in enumerate(zip(synthetic_values, colors, labels)):
        axes[1].bar(0, value, bottom=bottom, label=label, alpha=0.8, color=color, width=0.5)
        # Add value label
        if value > 0.02:  # Only show label if segment is large enough
            axes[1].text(0, bottom + value/2, f'{value:.2f}', 
                        ha='center', va='center', fontsize=12, fontweight='bold')
        bottom += value
    
    axes[1].set_xlim([-0.5, 0.5])
    axes[1].set_ylim([0, 1])
    axes[1].set_ylabel('Ratio', fontsize=12)
    axes[1].set_title('Synthetic Conversations', fontsize=14, fontweight='bold')
    axes[1].set_xticks([])
    axes[1].legend(loc='upper right')
    axes[1].grid(axis='y', alpha=0.3)
    
    subtitle_suffix = f" (exactly {exact_turns} patient turns)" if exact_turns else ""
    plt.suptitle(f'Average PTCF Distribution{subtitle_suffix}', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / 'ptc_stacked_average.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Step line plot showing PTC progression through conversation turns
    # Filter to conversations with at least turn_threshold patient messages
    real_df_filtered = real_df[real_df['total_patient_turns'] >= turn_threshold].copy()
    synthetic_df_filtered = synthetic_df[synthetic_df['total_patient_turns'] >= turn_threshold].copy()
    
    if exact_turns:
        print(f"\nFiltered to {len(real_df_filtered)} real and {len(synthetic_df_filtered)} synthetic conversations with exactly {exact_turns} patient turns")
    else:
        print(f"\nFiltered to {len(real_df_filtered)} real and {len(synthetic_df_filtered)} synthetic conversations with ≥turn_threshold turns")
    
    # Get turn-by-turn data
    real_turns = get_turn_classifications(real_df_filtered)
    synthetic_turns = get_turn_classifications(synthetic_df_filtered)
    
    # Filter to first 10 turns only
    real_turns = real_turns[real_turns['turn_index'] < turn_threshold].copy()
    synthetic_turns = synthetic_turns[synthetic_turns['turn_index'] < turn_threshold].copy()
    
    # Map classifications to numeric values: P=1, T=2, C=3, F=NaN
    ptc_map = {'P': 1, 'T': 2, 'C': 3, 'F': np.nan}
    real_turns['ptc_numeric'] = real_turns['classification'].map(ptc_map)
    synthetic_turns['ptc_numeric'] = synthetic_turns['classification'].map(ptc_map)
    
    # Average PTC value per turn index across all conversations (F will be NaN and excluded)
    real_avg = real_turns.groupby('turn_index')['ptc_numeric'].mean().reset_index()
    synthetic_avg = synthetic_turns.groupby('turn_index')['ptc_numeric'].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot step lines for non-filler turns only
    ax.step(real_avg['turn_index'], real_avg['ptc_numeric'], 
            where='mid', label='Real', linewidth=2.5, alpha=0.8, color='steelblue')
    ax.step(synthetic_avg['turn_index'], synthetic_avg['ptc_numeric'], 
            where='mid', label='Synthetic', linewidth=2.5, alpha=0.8, color='coral')
    
    ax.set_xlabel('Turn Index', fontsize=12)
    ax.set_ylabel('PTC Classification', fontsize=12)
    if exact_turns:
        title_text = f'Average PTC Progression (First {turn_threshold} Turns, Conversations with exactly {exact_turns} patient turns)'
    else:
        title_text = f'Average PTC Progression (First {turn_threshold} Turns, Conversations with ≥{turn_threshold} Turns)'
    ax.set_title(title_text, fontsize=14, fontweight='bold')
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['Problem (P)', 'Transition (T)', 'Change (C)'])
    ax.set_ylim([0.5, 3.5])
    ax.set_xlim([-0.5, turn_threshold - 0.5])
    
    # Force x-axis to show only integer values
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ptc_progression_stepline.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization plots saved to {output_dir}/")
    print("  - ptc_stacked_average.png: Stacked bar chart of average P/T/C/F ratios (all conversations)")
    print(f"  - ptc_progression_stepline.png: Step line plot showing PTC progression (first {turn_threshold} turns, conversations ≥{turn_threshold} turns)")


def main():
    """Main function to run PTC classification analysis."""
    import argparse
    from datetime import datetime
    import time
    
    parser = argparse.ArgumentParser(description='Classify conversations using PTC framework')
    parser.add_argument('--dataset', type=str, default='esc', 
                       help='Dataset type for real conversations (default: esc)')
    parser.add_argument('--synthetic-dir', type=str, help='Directory with synthetic conversations')
    parser.add_argument('--output-dir', type=str, default='output/ptc_analysis',
                       help='Output directory')
    parser.add_argument('--compare', action='store_true',
                       help='Compare real vs synthetic')
    parser.add_argument('--N', type=int, default=None,
                       help='Number of conversations to classify (default: all available samples)')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file (default: configs/default.yaml)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Number of parallel tasks to run (default: 1)')
    parser.add_argument('--turn-threshold', type=int, default=12,
                       help='Minimum number of turns for progression analysis (default: 12)')
    parser.add_argument('--exact-turns', type=int, default=None,
                       help='Only analyze conversations with exactly this many patient turns (default: None, analyze all)')
    parser.add_argument('--single-turn', action='store_true',
                       help='Use single-turn classification with limited history instead of full conversation')
    parser.add_argument('--num-messages', type=int, default=6,
                       help='Number of previous messages to include as history for single-turn mode (default: 6)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Clean string arguments
    args.dataset = args.dataset.strip().lower() if args.dataset else args.dataset
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize PTC judge
    judge = PTCClassifier(config, debug=args.debug)
    
    start_time = time.time()
    
    if args.compare and args.synthetic_dir:
        # Compare mode - need to get indices from synthetic data
        synthetic_dir = Path(args.synthetic_dir)
        
        # Extract indices from synthetic session files
        session_files = sorted(synthetic_dir.glob('session_*.json'))
        indices = [int(f.stem.split('_')[1]) for f in session_files]
        
        # Limit number of conversations if specified
        if args.N:
            indices = indices[:args.N]
            session_files = session_files[:args.N]
        
        print(f"Found {len(indices)} conversations to analyze")
        print(f"Batch size: {args.batch_size}")
        print(f"Classification mode: {'Single-turn' if args.single_turn else 'Full conversation'}")
        if args.single_turn:
            print(f"History window: {args.num_messages} messages")
        if args.exact_turns:
            print(f"Filtering to conversations with exactly {args.exact_turns} patient turns")
        print(f"Analyzing corresponding real conversations from {args.dataset} dataset...")
        real_df = analyze_conversations(judge, output_dir / 'real', batch_size=args.batch_size, 
                                        dataset=args.dataset, indices=indices,
                                        single_turn=args.single_turn, num_messages=args.num_messages,
                                        exact_turns=args.exact_turns)
        
        print("\nAnalyzing synthetic conversations...")
        synthetic_df = analyze_conversations(
            judge,
            output_dir / 'synthetic',
            batch_size=args.batch_size,
            data_dir=synthetic_dir,
            session_files=session_files,
            single_turn=args.single_turn,
            num_messages=args.num_messages,
            exact_turns=args.exact_turns,
        )
        
        print("\nComparing distributions...")
        compare_distributions(real_df, synthetic_df, output_dir, judge)
        
        print("\nGenerating visualizations...")
        visualize_distributions(real_df, synthetic_df, output_dir, turn_threshold=args.turn_threshold, exact_turns=args.exact_turns)
        
    elif args.synthetic_dir:
        # Analyze synthetic only
        print("Analyzing synthetic conversations...")
        print(f"Classification mode: {'Single-turn' if args.single_turn else 'Full conversation'}")
        if args.single_turn:
            print(f"History window: {args.num_messages} messages")
        if args.exact_turns:
            print(f"Filtering to conversations with exactly {args.exact_turns} patient turns")
        synthetic_dir = Path(args.synthetic_dir)
        # If N specified, pick the first N session files
        selected_files = None
        if args.N:
            all_files = sorted(synthetic_dir.glob('session_*.json'))
            selected_files = all_files[:args.N]
        df = analyze_conversations(
            judge,
            output_dir,
            batch_size=args.batch_size,
            data_dir=synthetic_dir,
            session_files=selected_files,
            single_turn=args.single_turn,
            num_messages=args.num_messages,
            exact_turns=args.exact_turns,
        )
    else:
        print("Error: Please provide --real-dir and/or --synthetic-dir")
        return
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\nResults saved to {output_dir}")
    print(f"Total time taken: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()