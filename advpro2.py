import os
import json
import ast
import astunparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader

from mutation import get_mutation_strategies, apply_strategies

class AdvPro:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def _parse_code(self, code):
        return ast.parse(code)

    def _get_importance_scores(self, code):
        input_ids = self.tokenizer(code, return_tensors='pt').input_ids
        attention_mask = self.tokenizer(code, return_tensors='pt').attention_mask

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Forward pass
        outputs = self.model(input_ids, attention_mask)
        logits = outputs.logits

        # Assume secure and vulnerable targets are token indices
        secure_target = 1  # Replace with actual target token index
        vulnerable_target = 2  # Replace with actual target token index

        # Compute probabilities
        secure_prob = logits[0, -1, secure_target]
        vulnerable_prob = logits[0, -1, vulnerable_target]

        # Calculate probability difference
        prob_diff = vulnerable_prob - secure_prob

        # Compute gradient
        prob_diff.backward()

        # Get embeddings gradient
        embeddings = self.model.get_input_embeddings().weight
        gradient = embeddings.grad

        # Compute importance scores
        importance_scores = torch.norm(gradient, dim=1)

        return importance_scores.cpu().detach().numpy()

    def _apply_mutations(self, ast_root, importance_scores):
        strategies = get_mutation_strategies()
        return apply_strategies(ast_root, strategies, importance_scores)

    def attack(self, code_prompt):
        ast_root = self._parse_code(code_prompt)
        importance_scores = self._get_importance_scores(code_prompt)
        mutated_ast = self._apply_mutations(ast_root, importance_scores)
        mutated_code = astunparse.unparse(mutated_ast)

        input_ids = self.tokenizer(mutated_code, return_tensors='pt').input_ids.to(self.device)
        outputs = self.model.generate(input_ids, max_length=512, num_return_sequences=1)
        generated_code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return generated_code

def process_sample(json_path, py_path):
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    with open(py_path, 'r') as f:
        py_content = f.read()

    advpro = AdvPro('Salesforce/codegen-2B-mono')
    generated_code = advpro.attack(py_content)
    return generated_code

def traverse_directory(root_dir):
    cnt = 0
    tot = 0

    for root, dirs, files in os.walk(root_dir):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            for sub_dir_name in os.listdir(dir_path):
                sub_dir_path = os.path.join(dir_path, sub_dir_name)
                if os.path.isdir(sub_dir_path):
                    json_file = None
                    py_file = None
                    for file_name in os.listdir(sub_dir_path):
                        if file_name.endswith(".json"):
                            json_file = os.path.join(sub_dir_path, file_name)
                        elif file_name.endswith(".py"):
                            py_file = os.path.join(sub_dir_path, file_name)
                    
                    if json_file and py_file:
                        print(f"Processing: {json_file}, {py_file}")
                        process_sample(json_file, py_file)
                        tot += 1
                        return

if __name__ == '__main__':
    dataset_dir = '../advpro-dataset/dataset_py'
    traverse_directory(dataset_dir)