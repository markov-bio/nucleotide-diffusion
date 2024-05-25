from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def calculate_nll(model, tokenizer, text):
    """Calculate the negative log likelihood of a text given a language model."""
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    outputs = model(input_ids, labels=input_ids)
    log_likelihood = outputs.loss * input_ids.shape[1]  # Multiply loss by length to get sum of log likelihoods
    return -log_likelihood.item()  # Return negative log likelihood

def average_nll_from_file(file_path, model_name="gpt2"):
    """Compute the average negative log likelihood for texts in a file."""
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    
    total_nll = 0
    count = 0

    with torch.no_grad(), open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:  # Ensure the line is not empty
                nll = calculate_nll(model, tokenizer, line)
                total_nll += nll
                count += 1

    average_nll = total_nll / count if count > 0 else float('inf')
    return average_nll

# Example usage
file_path = 'checkpoints/138.62M_ep0ep_ba2000ba.txt'  # Replace with your actual file path
print("Average NLL:", average_nll_from_file(file_path))
