# Patient-Ψ Generation

The task was creating CCD profiles from real convos and creating Patient-Ψ prompts from them.

## To run do the following:

1. Switch to my branch
2. `cd psibench/models`
3. rerun either `pip install -e . (I added langchain-core==0.3.79)`  
   or `pip install langchain-core==0.3.79`
4. `python3 patient_psi.py --transcript-file ESConv.json --conv-number 6`

`--conv-number` can be between 0-1299.

## Outputs

Outputs are saved in `PSI-bench/output/gen_psi_prof/`:

- `Patient{n}_CCD.json` = extracted CCD
- `Patient{n}_prompt.txt` = generated Patient-Ψ prompt

Let me know if y’all run into any issues.
