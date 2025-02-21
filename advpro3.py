import os
import json
import torch
import torch.nn.functional as F
import random
from transformers import AutoTokenizer, AutoModelForCausalLM

class AdvPro:
    def __init__(self, model_name, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
    
    def compute_importance(self, prompt, unsafe_keyword, safe_keyword):
        """
        对 prompt 进行前向计算，利用下一个 token 的 logits 计算目标标量：
            score = log_prob(unsafe_token) - log_prob(safe_token)
        然后对输入嵌入求梯度，计算每个 token 的梯度 L2 范数作为重要性分数。
        为简化起见，此处仅取 unsafe_keyword 和 safe_keyword 的首个 token。
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        
        # 获取输入对应的 embeddings（取模型内部的词嵌入）
        embeddings = self.model.transformer.wte(input_ids)
        embeddings.requires_grad_(True)
        
        # 前向计算：采用模型生成下一个 token 的 logits
        outputs = self.model(inputs_embeds=embeddings)
        logits = outputs.logits  # [1, seq_len, vocab_size]
        
        # 将 prompt 编码为 token id 序列
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        # 对 unsafe_keyword 和 safe_keyword 同样编码，取首个 token
        unsafe_ids = self.tokenizer.encode(unsafe_keyword, add_special_tokens=False)
        safe_ids = self.tokenizer.encode(safe_keyword, add_special_tokens=False)
        if not unsafe_ids or not safe_ids:
            raise ValueError("请确保 unsafe_keyword 和 safe_keyword 非空")
        unsafe_token_id = unsafe_ids[0]
        safe_token_id = safe_ids[0]
        
        # 为简化，取序列最后位置的 logits作为下一个 token 的预测
        next_token_logits = logits[0, -1]
        log_probs = F.log_softmax(next_token_logits, dim=-1)
        unsafe_log_prob = log_probs[unsafe_token_id]
        safe_log_prob = log_probs[safe_token_id]
        
        # 目标标量：希望 unsafe_log_prob > safe_log_prob，从而引导生成漏洞代码
        score = unsafe_log_prob - safe_log_prob
        
        # 反向传播计算梯度
        self.model.zero_grad()
        score.backward()
        grads = embeddings.grad[0]  # [seq_len, embed_dim]
        
        # 计算每个 token 的梯度 L2 范数作为重要性分数
        importance_scores = torch.norm(grads, dim=-1)
        return input_ids[0], importance_scores.detach()
    
    def mutate_prompt(self, prompt, token_ids, importance_scores, mutation_rate=0.2):
        """
        采用简单策略：随机概率下，将重要性最高的 token 替换为词汇表中随机的 token。
        """
        max_idx = torch.argmax(importance_scores).item()
        if random.random() < mutation_rate:
            vocab = self.tokenizer.get_vocab()
            random_token_id = random.choice(list(vocab.values()))
            new_token = self.tokenizer.decode([random_token_id])
            # 替换 token_ids 中对应位置
            mutated_ids = token_ids.clone()
            mutated_ids[max_idx] = random_token_id
            mutated_prompt = self.tokenizer.decode(mutated_ids)
            return mutated_prompt, max_idx, new_token
        else:
            return prompt, None, None
        
    def attack(self, prompt, unsafe_keyword=None, safe_keyword=None, max_iterations=20):
        """
        对输入 prompt 进行多轮迭代攻击：
          1. 计算归因得分，定位关键 token；
          2. 对关键 token 进行简单变异；
          3. 利用变异后的 prompt 生成代码补全；
          4. 重复上述过程。
        若未提供关键词，则采用默认值（与示例 JSON 中一致）。
        """
        if unsafe_keyword is None:
            unsafe_keyword = "if not self.memcache_servers:\n"
        if safe_keyword is None:
            safe_keyword = "if not self.memcache_servers or serialization_format is None:\n"
        
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
            
            # 生成代码补全（示例中设置 max_new_tokens 为 50）
            inputs = self.tokenizer(current_prompt, return_tensors="pt").to(self.device)
            generated_ids = self.model.generate(**inputs, max_new_tokens=50)
            generated_code = self.tokenizer.decode(generated_ids[0])
            print("Generated Code:\n", generated_code)
            
            # 这里可以根据生成的代码是否更接近 unsafe_label 判断是否继续攻击，
            # 本示例只做迭代展示，实际可结合 BLEU、字符串匹配等度量进行判断。
        return generated_code

# 以下是结合你提供的代码片段，实现对单个样本的处理和目录遍历

def process_sample(json_path, py_path):
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    with open(py_path, 'r') as f:
        py_content = f.read()

    # 初始化 AdvPro，并指定模型为 Salesforce/codegen-2B-mono
    advpro = AdvPro('Salesforce/codegen-2B-mono')
    # 从 JSON 中获取关键词（注意：这里 safe_label/unsafe_label 实际上通常为一整行代码，但我们采用其中的关键词进行指导）
    unsafe_keyword = json_data.get("unsafe_keyword", "if not self.memcache_servers:\n")
    safe_keyword = json_data.get("safe_keyword", "if not self.memcache_servers or serialization_format is None:\n")
    
    # 执行攻击，生成 adversarial code prompt
    generated_code = advpro.attack(py_content, unsafe_keyword=unsafe_keyword, safe_keyword=safe_keyword)
    return generated_code

def traverse_directory(root_dir):
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
                        print(f"\nProcessing: {json_file}, {py_file}")
                        generated_code = process_sample(json_file, py_file)
                        print("\nFinal Generated Code:\n", generated_code)
                        tot += 1
                        return  # 本示例只处理一个样本

if __name__ == '__main__':
    dataset_dir = '../advpro-dataset/dataset_py'
    traverse_directory(dataset_dir)
