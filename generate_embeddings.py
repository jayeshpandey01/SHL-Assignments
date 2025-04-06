import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
from tqdm import tqdm
import gc
import os
import numpy as np

# Memory optimization settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# Check if a GPU is available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and preprocess data
print("Loading and preprocessing data...")
df = pd.read_csv("outputs/grammar_scores.csv")
# Convert transcript column to string type and handle NaN values
df['transcript'] = df['transcript'].fillna('').astype(str)

# Load model and tokenizer
print("Loading tokenizer and model...")
model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Switch to a model designed for embeddings
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Move model to device
model = model.to(device)
model.eval()  # Set to evaluation mode

EMBEDDING_SIZE = 384  # Hidden size for MiniLM model

# Function to clear memory
def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def get_embedding(text, retry_count=0):
    max_retries = 3
    try:
        # Ensure text is string
        if not isinstance(text, str):
            text = str(text)
        
        # Skip empty texts
        if not text.strip():
            print("Warning: Empty text, returning zero embedding")
            return np.zeros(EMBEDDING_SIZE)

        # Tokenize with smaller max length
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512  # Increased max_length for better context
        )
        
        # Check for empty inputs
        if inputs["input_ids"].numel() == 0:
            print("Warning: Empty tokenization")
            return np.zeros(EMBEDDING_SIZE)

        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            # Get model outputs
            outputs = model(**inputs)
            
            # Get the embeddings from the last hidden state
            last_hidden_state = outputs.last_hidden_state
            
            # Use attention mask to get mean of only valid tokens
            attention_mask = inputs['attention_mask']
            
            # Calculate mean embedding (ignoring padding tokens)
            sum_embeddings = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1), dim=1)
            sum_mask = torch.clamp(torch.sum(attention_mask, dim=1).unsqueeze(-1), min=1e-9)
            embedding = (sum_embeddings / sum_mask).squeeze()
            
            # Convert to numpy and verify shape
            embedding = embedding.cpu().numpy()
            
            if embedding.size != EMBEDDING_SIZE:
                print(f"Warning: Got embedding of size {embedding.size}, expected {EMBEDDING_SIZE}")
                return np.zeros(EMBEDDING_SIZE)
                
            # Verify no NaN values
            if np.isnan(embedding).any():
                print("Warning: NaN values in embedding")
                return np.zeros(EMBEDDING_SIZE)
                
            return embedding

    except RuntimeError as e:
        if 'out of memory' in str(e) and retry_count < max_retries:
            print(f"Out of memory error, clearing cache and retrying... (attempt {retry_count + 1}/{max_retries})")
            clear_memory()
            return get_embedding(text, retry_count + 1)
        else:
            print(f"Error processing text: {str(e)}")
            return np.zeros(EMBEDDING_SIZE)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return np.zeros(EMBEDDING_SIZE)
    finally:
        clear_memory()

# Process in batches
print("Generating embeddings...")
batch_size = 8  # Increased batch size since we're using a smaller model
embeddings = []
num_batches = (len(df) + batch_size - 1) // batch_size

for i in tqdm(range(0, len(df), batch_size)):
    batch = df["transcript"].iloc[i:i+batch_size]
    batch_embeddings = []
    
    for text in batch:
        emb = get_embedding(text)
        if not np.all(emb == 0):  # Only append if not all zeros
            batch_embeddings.append(emb)
        else:
            print("Warning: Got zero embedding, using random small values")
            batch_embeddings.append(np.random.normal(0, 0.01, EMBEDDING_SIZE))
    
    embeddings.extend(batch_embeddings)
    
    # Save intermediate results
    if i > 0 and i % 50 == 0:
        temp_df = pd.DataFrame(
            embeddings + [np.zeros(EMBEDDING_SIZE)] * (len(df) - len(embeddings)),
            columns=[f"emb_{i}" for i in range(EMBEDDING_SIZE)]
        )
        temp_df = pd.concat([df.iloc[:len(embeddings)].reset_index(drop=True), temp_df], axis=1)
        temp_df.to_csv("outputs/train_features_temp.csv", index=False)
        print(f"\nSaved intermediate results ({len(embeddings)}/{len(df)} samples)")

# Verify we have embeddings for all samples
assert len(embeddings) == len(df), f"Got {len(embeddings)} embeddings for {len(df)} samples"

# Convert to DataFrame
print("\nSaving final results...")
emb_df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(EMBEDDING_SIZE)])
final_df = pd.concat([df.reset_index(drop=True), emb_df], axis=1)
final_df.to_csv("outputs/train_features_full.csv", index=False)
print("✅ Embeddings saved successfully.")
