# Lexical Diversity Analysis Methodology

This document outlines the rigorous methodology used to compare the lexical diversity of Real Patient transcripts vs. LLM-generated (Synthetic) responses.

## 1. Overview
Lexical diversity measures the richness of vocabulary used by a speaker. Comparing LLMs to humans is challenging due to confounding factors like **response length** and **verbosity**. This implementation (`lexical_diversity_strict.py`) uses a multi-strategy approach to ensure fair and robust comparisons.

## 2. Preprocessing
Before analysis, all text undergoes strict preprocessing:
*   **Lowercase**: All text is converted to lowercase.
*   **Punctuation Removal**: All punctuation is stripped.
*   **Disfluency Removal**: Common speech disfluencies (*um, uh, hm, ah, er, mm, mhm, huh*) are removed to focus on content words.

## 3. Metrics
We use one primary metric, calculated using the `lexicalrichness` library:

### A. Bidirectional MTLD (Measure of Textual Lexical Diversity)
*   **Definition**: Calculates the average length of sequential word strings that maintain a Type-Token Ratio (TTR) above a threshold (0.72).
*   **Bidirectional**: To avoid bias from the start/end of the text, we calculate MTLD forward, then reverse the text and calculate it backward, averaging the two scores.
*   **Use Case**: Best for **Session Level** analysis (long texts).



## 4. Analysis Levels & Strategies

### Level 1: Session Level 
Aggregates all turns from a speaker into one large text block per session.

*   **Strategy A: Raw Aggregated**
    *   **Method**: Concatenate all turns. Calculate metrics on the full text.
    *   **Purpose**: Use for **Statistical Modeling** (e.g., `MTLD ~ Speaker + TokenCount`). Captures the full "volume" of diversity.
*   **Strategy B: Matched Truncated**
    *   **Method**: Find the shorter speaker in the pair (Real vs. Synth). Truncate the longer speaker's text to match that exact token count.
    *   **Purpose**: Use for **Direct Comparison**. Removes length as a confounding variable.
*   **Strategy C: Cumulative Progression**
    *   **Method**: Calculate MTLD after Turn 1, then Turn 1+2, then Turn 1+2+3, etc.
    *   **Purpose**: Visualizes *when* diversity plateaus. Does the LLM start diverse but become repetitive?

### Level 2: Turn Level (The "Dynamic View")
Analyzes each response individually to capture moment-to-moment variance.

*   **Strategy D: Turn-Level Analysis**
    *   **Method**: Calculate **MTLD** for each individual turn.
    *   **Constraint**: **Low Threshold**. We calculate scores for any turn with **>= 1 token**.
    *   **Flagging**: Turns with **< 100 tokens** are flagged in visualizations (colored red) to indicate potentially unstable scores, but the score is still shown.
    *   **Purpose**: Detects local repetition or "canned" responses that might be hidden by session aggregation.

## 5. Output & Visualization
Results are saved to `output/lexical_diversity_strict/{psi}/{dataset}/`.

### A. Boxplot Comparison (`boxplot_mtld_comparison.png`)
This plot summarizes the distribution of MTLD scores across all sessions.
*   **X-Axis**: Strategies (Raw vs. Matched).
*   **Y-Axis**: MTLD Score (Higher = More Diverse).
*   **Box**: Shows the median (middle line), 25th percentile (bottom edge), and 75th percentile (top edge).
*   **Interpretation**:
    *   If the "Synthetic" box is lower than "Real", the model is less diverse.
    *   Comparing "Raw" vs "Matched" shows if the difference is due to the model simply talking more (Raw) or a genuine vocabulary difference (Matched).

### B. Session Plots (`session_plots/session_{id}/`)
For each session, we generate two detailed plots:

1.  **Cumulative MTLD (`cumulative_mtld.png`)**:
    *   **X-Axis**: Turn Index (progression of the conversation).
    *   **Y-Axis**: MTLD of all text *up to that turn*.
    *   **Points**: Each dot represents the score at that step.
        *   **Blue/Orange**: Turn length >= 100 tokens.
        *   **Red**: Turn length < 100 tokens (warning: score might be volatile).
    *   **Annotations**: The number next to each dot is the **token count** of the text added in that turn.

2.  **Turn-Level MTLD (`turn_mtld.png`)**:
    *   **X-Axis**: Turn Index.
    *   **Y-Axis**: MTLD of *just that specific turn*.
    *   **Interpretation**: Shows the variance in quality. Does the model have "bad turns" (low drops) or is it consistent?
    *   **Annotations**: Same as above (Token count + Red warning color).

## 6. References & Justification

We utilize a minimum threshold of **100 tokens** for flagging reliable MTLD scores based on established research indicating that MTLD stabilizes and becomes minimally related to text length at this point.

*   **Koizumi, R. (2012).** Relationships Between Text Length and Lexical Diversity Measures: Can We Use Short Texts of Less than 100 Tokens? *Vocabulary Learning and Instruction*, 1(1), 60-69.
    > "MTLD was minimally related to length, especially in samples that were longer than 100 tokens."

*   **Fergadiotis, G., Wright, H. H., & West, T. M. (2015).** Psychometric Evaluation of Lexical Diversity Indices: Assessing Length Effects. *Journal of Speech, Language, and Hearing Research*, 58(3), 840–852.
    > "MTLD did not correlate strongly with TTR (r = .32), a measure known to be influenced by length... Koizumi and In’nami (2012) reported additional supporting evidence... They found that MTLD was minimally related to length, especially in samples that were longer than 100 tokens."

## 7. Usage Example

To run the analysis, use the following command:

```bash
python -m psibench.eval.lexical_diversity_strict --data-dir data/synthetic --psi eeyore --dataset esc --output-dir output/lexical_diversity_stict
```
