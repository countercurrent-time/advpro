import os
import json
import torch
import torch.nn.functional as F
import random
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------------
# 全局模型加载（仅加载一次）
# -------------------------------
MODEL_NAME = "Salesforce/codegen-2B-mono"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

global_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
global_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
global_model.to(DEVICE)
global_model.eval()

# -------------------------------
# AdvPro 类，不再重复加载模型
# -------------------------------
class AdvPro:
    def __init__(self):
        self.device = DEVICE
        self.tokenizer = global_tokenizer
        self.model = global_model
    
    def compute_importance(self, prompt, unsafe_keyword, safe_keyword):
        print(f"prompt: {prompt}")
        print(f"unsafe_keyword: {unsafe_keyword}")
        print(f"safe_keyword: {safe_keyword}")
        if unsafe_keyword is None or safe_keyword is None:
            raise ValueError("必须提供 unsafe_keyword 和 safe_keyword")
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]

        # 计算词嵌入，并确保其可以计算梯度
        embeddings = self.model.transformer.wte(input_ids)
        embeddings = embeddings.detach().clone().requires_grad_(True)
        
        # 构造 position_ids，与输入长度一致
        seq_length = input_ids.size(1)
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=self.device).unsqueeze(0)
        
        # 前向计算，传入 inputs_embeds 与 position_ids
        outputs = self.model(inputs_embeds=embeddings, position_ids=position_ids)
        logits = outputs.logits  # [1, seq_len, vocab_size]
        
        # 取最后一个 token 位置的 logits，并计算 log softmax
        next_token_logits = logits[0, -1]
        log_probs = F.log_softmax(next_token_logits, dim=-1)
        
        # 对 unsafe_keyword 与 safe_keyword 编码，取首个 token
        unsafe_ids = self.tokenizer.encode(unsafe_keyword, add_special_tokens=False)
        safe_ids = self.tokenizer.encode(safe_keyword, add_special_tokens=False)
        if not unsafe_ids or not safe_ids:
            raise ValueError("unsafe_keyword 和 safe_keyword 不应为空")
        unsafe_token_id = unsafe_ids[0]
        safe_token_id = safe_ids[0]
        
        if unsafe_token_id < 0 or unsafe_token_id >= log_probs.size(0):
            raise ValueError("unsafe_token_id out of bounds")
        if safe_token_id < 0 or safe_token_id >= log_probs.size(0):
            raise ValueError("safe_token_id out of bounds")
        
        unsafe_log_prob = log_probs[unsafe_token_id]
        safe_log_prob = log_probs[safe_token_id]
        score = unsafe_log_prob - safe_log_prob
        
        self.model.zero_grad()
        score.backward()
        
        grads = embeddings.grad  # [1, seq_len, embed_dim]
        importance_scores = torch.norm(grads[0], dim=-1)
        return input_ids[0], importance_scores.detach()
    
    def mutate_prompt(self, prompt, token_ids, importance_scores, mutation_rate=0.2):
        max_idx = torch.argmax(importance_scores).item()
        if random.random() < mutation_rate:
            vocab = self.tokenizer.get_vocab()
            random_token_id = random.choice(list(vocab.values()))
            new_token = self.tokenizer.decode([random_token_id])
            mutated_ids = token_ids.clone()
            mutated_ids[max_idx] = random_token_id
            mutated_prompt = self.tokenizer.decode(mutated_ids)
            return mutated_prompt, max_idx, new_token
        else:
            return prompt, None, None
        
    def attack(self, prompt, unsafe_keyword, safe_keyword, max_iterations=20):
        current_prompt = prompt
        for i in range(max_iterations):
            print(f"\n=== Iteration {i+1} ===")
            token_ids, importance_scores = self.compute_importance(current_prompt, unsafe_keyword, safe_keyword)
            tokens = self.tokenizer.decode(token_ids).split()
            print("Current Prompt Tokens:", tokens)
            print("Importance Scores:", importance_scores.cpu().numpy())
            
            mutated_prompt, mutated_idx, new_token = self.mutate_prompt(current_prompt, token_ids, importance_scores)
            if mutated_idx is not None:
                print(f"Mutated token at position {mutated_idx} to '{new_token}'")
                current_prompt = mutated_prompt
            else:
                print("No mutation applied in this iteration.")
            
            inputs = self.tokenizer(current_prompt, return_tensors="pt").to(self.device)
            generated_ids = self.model.generate(**inputs, max_new_tokens=50)
            generated_code = self.tokenizer.decode(generated_ids[0])
            print("Generated Code:\n", generated_code)
            
        return generated_code

def process_sample(json_path, py_path):
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    with open(py_path, 'r') as f:
        py_content = f.read()

    unsafe_keyword = json_data.get("unsafe_keyword")
    safe_keyword = json_data.get("safe_keyword")
    if unsafe_keyword is None or safe_keyword is None:
        raise ValueError(f"文件 {json_path} 中未包含 unsafe_keyword 或 safe_keyword")
    
    advpro = AdvPro()
    generated_code = advpro.attack(py_content, unsafe_keyword=unsafe_keyword, safe_keyword=safe_keyword)
    return generated_code

def traverse_directory(root_dir):
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
                        print(f"\nProcessing: {json_file}, {py_file}")
                        generated_code = process_sample(json_file, py_file)
                        print("\nFinal Generated Code:\n", generated_code)
                        return  # 示例仅处理一个样本

if __name__ == '__main__':
    dataset_dir = '../advpro-dataset/dataset_py'
    traverse_directory(dataset_dir)
