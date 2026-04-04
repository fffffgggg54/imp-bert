import math
import torch
import numpy as np
import mteb
from mteb.models.model_meta import ModelMeta 
from mteb.models.models_protocols import EncoderProtocol
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling
)

class MTEB_Wrapper(EncoderProtocol):
    """
    MTEB wrapper
    """
    def __init__(self, model, tokenizer):
        super().__init__("test", 0.1)
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    def encode(self, sentences, task_metadata, hf_split, hf_subset, batch_size=32, **kwargs):
        all_embeddings = []
        self.model.eval()
        
        with torch.inference_mode():
            for i, batch_sentences in enumerate(sentences):
                batch_sentences = batch_sentences['text']
                
                # Standard padding
                inputs = self.tokenizer(
                    batch_sentences, 
                    padding=True, 
                    truncation=True, 
                    return_tensors="pt", 
                    max_length=512
                )
                
                # non_blocking=True for async PCIe transfer
                inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
                
                outputs = self.model(**inputs, output_hidden_states=True)
                
                # Extract last hidden state: (batch_size, sequence_length, hidden_size)
                last_hidden = outputs.hidden_states[-1]
                
                # Mean Pooling (respecting standard padding masks)
                attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).to(torch.bfloat16)
                sum_embeddings = torch.sum(last_hidden * attention_mask, dim=1)
                sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                
                embeddings = sum_embeddings / sum_mask
                
                # Normalize and cast to fp32 for standard MTEB CPU distance calculations
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().to(torch.float32).numpy())
        print(np.vstack(all_embeddings))
        return np.vstack(all_embeddings)

    def similarity(self, embeddings1, embeddings2):
        """
        Computes the all-to-all similarity matrix between two collections of embeddings.
        Returns: [num_embeddings_1, num_embeddings_2]-shaped torch tensor.
        """
        # Ensure inputs are torch tensors
        if isinstance(embeddings1, np.ndarray):
            embeddings1 = torch.from_numpy(embeddings1)
        if isinstance(embeddings2, np.ndarray):
            embeddings2 = torch.from_numpy(embeddings2)
            
        # Optional safeguard: re-normalize in case external unnormalized embeddings are passed
        #embeddings1 = torch.nn.functional.normalize(embeddings1, p=2, dim=-1)
        #embeddings2 = torch.nn.functional.normalize(embeddings2, p=2, dim=-1)
        
        # Dot product of normalized vectors equals cosine similarity
        return torch.matmul(embeddings1, embeddings2.transpose(0, 1))

    def similarity_pairwise(self, embeddings1, embeddings2):
        """
        Computes the one-to-one similarity between corresponding pairs of embeddings.
        Returns: [num_embeddings]-shaped torch tensor.
        """
        # Ensure inputs are torch tensors
        if isinstance(embeddings1, np.ndarray):
            embeddings1 = torch.from_numpy(embeddings1)
        if isinstance(embeddings2, np.ndarray):
            embeddings2 = torch.from_numpy(embeddings2)
            
        # Use built-in pairwise cosine similarity 
        return torch.nn.functional.cosine_similarity(embeddings1, embeddings2, dim=-1)

def evaluate_text_modeling(model, tokenizer, dataset_path, dataset_name=None, split="test", text_column="text", batch_size=512):
    """
    Standard Language Modeling loop using standard 15% random token masking.
    """
    print(f"\n--- Evaluating Text Modeling on {dataset_path} ({split} split) ---")
    
    dataset = load_dataset(dataset_path, dataset_name, split=split) if dataset_name else load_dataset(dataset_path, split=split)

    def tokenize_function(examples):
        # Standard tokenization without packing
        return tokenizer(examples[text_column], return_special_tokens_mask=True, truncation=True, max_length=512)

    tokenized_datasets = dataset.map(
        tokenize_function, batched=True, num_proc=8, 
        remove_columns=[col for col in dataset.column_names if col != "input_ids"],
        desc=f"Tokenizing {dataset_path}"
    )
    tokenized_datasets = tokenized_datasets.filter(lambda x: len(x["input_ids"]) > 0, num_proc=8)

    # Standard Random Token Masking (No Packing)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    
    dataloader = DataLoader(
        tokenized_datasets, 
        batch_size=batch_size, 
        collate_fn=data_collator,
        num_workers=8,
        pin_memory=True
    )

    model.eval()
    total_loss, total_steps = 0.0, 0

    print(f"Running forward passes (Batch Size: {batch_size})...")
    
    # Context manager strictly forces Flash Attention execution
    with torch.inference_mode():
        for batch in dataloader:
            batch = {k: v.to(model.device, non_blocking=True) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            total_steps += 1

    avg_loss = total_loss / total_steps
    pseudo_perplexity = math.exp(avg_loss)
    
    print(f"[{dataset_path.upper()}] Cross-Entropy Loss: {avg_loss:.4f} | Pseudo-Perplexity: {pseudo_perplexity:.4f}")
    return avg_loss, pseudo_perplexity

def evaluate_zero_shot_glue(model, tokenizer, batch_size=512):
    """
    GLUE NLU capabilities using batched zero-shot cloze tasking.
    """
    print("\n--- Evaluating Zero-Shot GLUE (NLU) ---")
    model.eval()
    
    def get_token_id(word):
        return tokenizer.encode(word, add_special_tokens=False)[0]

    # --- TASK 1: SST-2 ---
    print("Evaluating GLUE: SST-2 (Sentiment)...")
    sst2 = load_dataset("glue", "sst2", split="validation")
    
    sst2_pos_token = get_token_id("great")
    sst2_neg_token = get_token_id("terrible")
    correct_sst2 = 0

    prompts = [f"{ex['sentence']} It was {tokenizer.mask_token}." for ex in sst2]
    labels = [ex["label"] for ex in sst2]

    with torch.inference_mode():
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            # Standard padded sequence arrays
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device, non_blocking=True)
            outputs = model(**inputs)
            
            mask_positions = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)
            mask_logits = outputs.logits[mask_positions[0], mask_positions[1], :]
            
            pos_scores = mask_logits[:, sst2_pos_token]
            neg_scores = mask_logits[:, sst2_neg_token]
            
            predictions = (pos_scores > neg_scores).long().cpu().numpy()
            
            for pred, label in zip(predictions, batch_labels):
                if pred == label:
                    correct_sst2 += 1

    sst2_accuracy = correct_sst2 / len(sst2)
    print(f"[GLUE - SST-2] Batched Zero-Shot Accuracy: {sst2_accuracy:.4f}")


RARB_tasks = [ 
    "ARCChallenge",
    "AlphaNLI",
    "HellaSwag",
    "WinoGrande",
    "PIQA",
    "SIQA",
    "Quail",
    "SpartQA",
    "TempReasonL1",
    "TempReasonL2Pure",
    "TempReasonL2Fact",
    "TempReasonL2Context",
    "TempReasonL3Pure",
    "TempReasonL3Fact",
    "TempReasonL3Context",
    "RARbCode",
    "RARbMath",
]
    
def run_mteb_benchmark(model, tokenizer, batch_size=32):
    """
    Evaluates the model on ALL MTEB English tasks with extreme batch sizes.
    """
    print("\n--- Starting Massive Text Embedding Benchmark (MTEB) ---")

    mteb_model = MTEB_Wrapper(model, tokenizer)
    all_english_tasks = mteb.get_tasks(languages=["eng"])
    
    print(f"Loaded {len(all_english_tasks)} English tasks for evaluation.")
    
    # FIX 2: Define a ModelMeta configuration with a lambda loader
    '''
    custom_model_meta = ModelMeta(
        name="ModernBERT-base-custom",
        revision="1.0.0",
        languages=["eng-Latn"],
        loader=lambda **kwargs: mteb_model  # Returns your pre-loaded PyTorch model wrapper
    )
    '''
    
    # Retrieve the task (use get_tasks for v2 compatibility)
    tasks = mteb.get_tasks(tasks=RARB_tasks)
    
    # Pass the custom_model_meta into evaluate, not the wrapper directly
    results = mteb.evaluate(mteb_model, tasks=tasks, encode_kwargs={"batch_size": 32})

    
    print("\n--- Summary of MTEB Evaluation Results ---")
    for task_results in results:
        task_name = task_results.task_name
        try:
            if 'accuracy' in task_results.scores.get('test', [{}])[0]:
                metric, score = "Accuracy", task_results.scores['test'][0]['accuracy']
            elif 'cosine_spearman' in task_results.scores.get('test', [{}])[0]:
                metric, score = "Cosine Spearman", task_results.scores['test'][0]['cosine_spearman']
            elif 'ndcg_at_10' in task_results.scores.get('test', [{}])[0]:
                metric, score = "NDCG@10", task_results.scores['test'][0]['ndcg_at_10']
            else:
                metric = list(task_results.scores['test'][0].keys())[0]
                score = task_results.scores['test'][0][metric]
                
            print(f"[{task_name}] {metric}: {score:.4f}")
        except Exception:
            pass 

if __name__ == "__main__":
    # Older BERT architecture constraint
    MODEL_NAME = "google-bert/bert-base-uncased" 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Batch Sizing
    # Set to 512 for large BERT. If you have all 141GB free, you can safely push this to 1024.
    BATCH_SIZE = 32
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"Hardware: Initializing on {DEVICE.upper()}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Loading Model: {MODEL_NAME}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    model = AutoModelForMaskedLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa"
    )
    model.to(DEVICE)
    
    evaluate_text_modeling(model, tokenizer, "wikitext", "wikitext-2-raw-v1", batch_size=BATCH_SIZE)
    #evaluate_text_modeling(model, tokenizer, "ptb_text_only", "penn_treebank", text_column="sentence", batch_size=BATCH_SIZE)
    
    evaluate_zero_shot_glue(model, tokenizer, batch_size=BATCH_SIZE)
    
    #run_mteb_benchmark(model, tokenizer, batch_size=BATCH_SIZE)
    
    print("\nAll benchmarking completed successfully.")