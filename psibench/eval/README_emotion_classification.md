# Emotion Classification

This module classifies patient turns in therapy conversations using **Plutchik's 8 basic emotions** plus a neutral category:

- anger
- disgust
- fear
- joy
- sadness
- surprise
- anticipation
- trust
- neutral

## Features

- **Context-aware classification**: Uses up to 4 previous messages as context
- **Parallel processing**: Classifies multiple conversations and turns in parallel for efficiency
- **Persistent storage**: Saves all classifications to CSV and JSON for later analysis
- **Comprehensive visualizations**:
  - Line graphs showing emotion percentages across turns (all emotions in one figure)
  - Horizontal stacked bar charts comparing emotion distributions across datasets

## Usage

### Analyze all HuggingFace PSI-backend pairs against real data

```bash
python -m psibench.eval.emotion_classification \
  --hf \
  --batch-size 32 \
  --config configs/default.yaml
```

### Analyze with exact turn filtering

```bash
python -m psibench.eval.emotion_classification \
  --hf \
  --batch-size 500 \
  --exact-turns 12 \
  --config configs/default.yaml
```

### Regenerate visualizations from saved results

```bash
python -m psibench.eval.emotion_classification \
  --csv-file output/emotion_analysis
```

## Command-line Arguments

- `--hf`: Load all PSI-backend pairs from HuggingFace dataset
- `--csv-file DIR`: Regenerate visualizations from saved CSV/JSON files
- `--output-dir DIR`: Output directory (default: output/emotion_analysis)
- `--batch-size N`: Number of conversations to process in parallel (default: 1)
- `--num-messages N`: Number of previous messages for context (default: 4)
- `--exact-turns N`: Only analyze conversations with exactly N patient turns
- `--turn-threshold N`: Maximum turn index for line plots (default: 12)
- `--config PATH`: Path to config file (default: configs/default.yaml)
- `--debug`: Enable debug logging

## Output Files

The analysis produces the following files:

### CSV Files
- `real_emotion_summary.csv`: Summary statistics for real conversations
- `*_emotion_summary.csv`: Summary statistics for each synthetic dataset
- Contains columns:
  - `conversation_id`: Unique identifier
  - `total_patient_turns`: Number of patient turns
  - `anger_count`, `disgust_count`, ...: Counts for each emotion
  - `anger_pct`, `disgust_pct`, ...: Percentages for each emotion

### JSON Files
- `real_emotion_detailed.json`: Complete classifications for real conversations
- Contains full turn-by-turn emotion classifications with content

### Visualizations
- `emotion_percentages_by_turn.png`: Line graphs showing how each emotion's percentage changes across turns (all emotions in subplots)
- `emotion_distribution_all_pairs.png`: Horizontal stacked bar chart comparing overall emotion distributions across all datasets

## Architecture

The module follows the same pattern as `ptc_classification.py`:

1. **EmotionClassifier**: Core classifier that uses LLM judge to classify emotions
   - Uses the prompt from `judge_prompt.py`
   - Processes turns in parallel using `litellm.batch_completion`
   - Includes up to 4 previous messages as context

2. **analyze_conversations()**: Processes conversations and saves results
   - Works with both real and synthetic data
   - Filters by exact turn count if specified
   - Saves both summary CSV and detailed JSON

3. **compare_all_hf_pairs()**: Analyzes all PSI-backend combinations
   - Loads real data from eeyore dataset (esc, hope, annomi)
   - Loads synthetic data from HuggingFace
   - Generates comprehensive visualizations

4. **Visualization functions**:
   - `visualize_emotion_percentages_by_turn()`: Creates multi-panel line plots
   - `visualize_emotion_distributions()`: Creates stacked bar charts

## Example Workflow

```bash
# 1. Run emotion classification on all HF pairs with 12-turn conversations
python -m psibench.eval.emotion_classification \
  --hf \
  --batch-size 32 \
  --exact-turns 12 \
  --config configs/default.yaml

# Output will be saved to: output/emotion_analysis/exact_turns_12_TIMESTAMP/

# 2. Later, regenerate visualizations with different parameters
python -m psibench.eval.emotion_classification \
  --csv-file output/emotion_analysis/exact_turns_12_20260301_120000 \
  --turn-threshold 10
```

## Integration with Existing Code

The emotion classification system integrates seamlessly with the existing PSI-bench evaluation framework:

- Uses the same data loaders (`load_eeyore_dataset`, `load_synthetic_hf_to_df`)
- Follows the same configuration structure
- Uses the same utility functions for naming and sorting
- Follows the same CSV/JSON output pattern for reproducibility
