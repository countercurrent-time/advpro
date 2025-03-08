# for Salesforce/codegen2-1B
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# model_name = "Salesforce/codegen2-1B"
# device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型和 tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, revision="main")
# model.to(device)
# model.eval()

model_name = "Salesforce/codegen-2B-mono"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型和 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)

# 定义一个简单的代码提示
prompt = "def add(a, b):\n    return a + b\n"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# 生成补全结果，设定生成50个 token
output_ids = model.generate(**inputs, labels=inputs["input_ids"], max_new_tokens=50)
output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("Generated Output:")
print(output)