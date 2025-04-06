import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import gc
import os

# Memory optimization settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
torch.backends.cuda.max_memory_split_size = 512 * 1024 * 1024  # 512MB

# Load transcript data
df = pd.read_csv("outputs/transcripts.csv")

# Load tokenizer
print("⏳ Loading tokenizer...")
model_name = "deepseek-ai/deepseek-coder-1.3b-base"  # Using smaller 1.3B model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to free up memory
def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# Load model with memory optimizations
print("⏳ Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    trust_remote_code=True,
    offload_folder="offload"
)

# Build prompt for grammar scoring
def build_prompt(text):
    return f"""<|system|>You are an English grammar evaluator. Rate the grammar quality of the following transcript from 0 (worst) to 5 (perfect grammar).<|end|>
<|user|>Transcript: {text}<|end|>
<|assistant|>Score:"""

# Get grammar score using model
def get_grammar_score(text):
    try:
        prompt = build_prompt(text)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                num_beams=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        try:
            score_str = result.split("Score:")[-1].strip().split()[0]
            score = float("".join(c for c in score_str if c.isdigit() or c == "."))
            return min(max(score, 0), 5)
        except:
            return None
    except Exception as e:
        print(f"Error processing text: {str(e)}")
        return None
    finally:
        clear_memory()

# Process transcripts in small batches
print("⏳ Scoring transcripts...")
batch_size = 5
scores = []

for i in tqdm(range(0, len(df), batch_size)):
    batch = df["transcript"].iloc[i:i+batch_size]
    batch_scores = []
    
    for text in batch:
        score = get_grammar_score(text)
        batch_scores.append(score)
    
    scores.extend(batch_scores)
    
    # Save intermediate results every 50 samples
    if i % 50 == 0 and i > 0:
        temp_df = df.copy()
        temp_df["grammar_score"] = scores + [None] * (len(df) - len(scores))
        temp_df.to_csv("outputs/grammar_scores_temp.csv", index=False)

# Save final results
df["grammar_score"] = scores
df.to_csv("outputs/grammar_scores.csv", index=False)
print("✅ Saved grammar scores to outputs/grammar_scores.csv")
