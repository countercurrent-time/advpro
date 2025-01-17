import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import ast
import numpy as np

# Load the model and tokenizer
MODEL_NAME = "Salesforce/codegen-2b-mono"
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
model.train()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Function to load data from the dataset
def load_sample(base_dir, cve_id, commit_id):
    sample_dir = os.path.join(base_dir, cve_id, commit_id)
    
    json_file = [f for f in os.listdir(sample_dir) if f.endswith('.json')]
    py_file = [f for f in os.listdir(sample_dir) if f.endswith('.py')]

    if not json_file or not py_file:
        raise FileNotFoundError("JSON or Python file is missing in the specified directory.")

    json_path = os.path.join(sample_dir, json_file[0])
    py_path = os.path.join(sample_dir, py_file[0])

    with open(json_path, 'r') as f:
        json_data = json.load(f)

    with open(py_path, 'r') as f:
        code_prompt = f.read()
    
    return json_data, code_prompt

# Compute gradients to identify important tokens
def compute_gradients(model, tokenizer, code_prompt, safe_label, unsafe_label):
    inputs = tokenizer(code_prompt, return_tensors="pt", padding=True).to(model.device)

    safe_token = tokenizer(safe_label, return_tensors="pt").input_ids[0]
    unsafe_token = tokenizer(unsafe_label, return_tensors="pt").input_ids[0]

    safe_token = safe_token.to(model.device)
    unsafe_token = unsafe_token.to(model.device)

    outputs = model(**inputs, labels=inputs["input_ids"])
    logits = outputs.logits
    
    # Probabilities for the safe and unsafe labels
    probs = torch.softmax(logits, dim=-1)
    safe_prob = torch.gather(probs, 2, safe_token.unsqueeze(0).unsqueeze(0)).mean()
    unsafe_prob = torch.gather(probs, 2, unsafe_token.unsqueeze(0).unsqueeze(0)).mean()

    # Compute the gradient of the probability difference
    prob_diff = unsafe_prob - safe_prob
    prob_diff.backward()

    gradients = inputs["input_ids"].grad
    importance_scores = gradients.norm(dim=-1).squeeze().tolist()
    
    return importance_scores

# Parse and aggregate token importance using AST
def aggregate_importance_with_ast(code_prompt, importance_scores):
    tree = ast.parse(code_prompt)
    token_importance = {}

    # Assign scores to AST nodes based on their token positions
    for node in ast.walk(tree):
        if hasattr(node, 'lineno'):
            token_importance[(node.lineno, node.col_offset)] = 0

    # Aggregate importance scores for tokens into AST nodes
    for idx, score in enumerate(importance_scores):
        token_pos = tokenizer.convert_ids_to_tokens([inputs["input_ids"][0][idx]])
        token_importance[token_pos] += score
    
    return token_importance

# Perform semantic-preserving mutations
def apply_mutation(code_prompt, mutation_type):
    # Placeholder for mutation logic (e.g., identifier replacement)
    if mutation_type == "replace_identifier":
        return code_prompt.replace("identifier", "new_identifier")
    elif mutation_type == "add_alias":
        return "from module import func as alias\n" + code_prompt
    return code_prompt

# Main workflow
if __name__ == "__main__":
    dataset_dir = "dataset_py"
    cve_id = "CVE-2009-5145"
    commit_id = "2abdf14620f146857dc8e3ffd2b6a754884c331d"

    # Load the sample
    sample_data, code_prompt = load_sample(dataset_dir, cve_id, commit_id)

    # Compute gradients
    importance_scores = compute_gradients(
        model, tokenizer, code_prompt, sample_data["safe_label"], sample_data["unsafe_label"]
    )

    # Aggregate importance scores
    token_importance = aggregate_importance_with_ast(code_prompt, importance_scores)

    # Apply mutations
    mutated_code = apply_mutation(code_prompt, mutation_type="replace_identifier")

    # Save or test mutated code
    print(mutated_code)
