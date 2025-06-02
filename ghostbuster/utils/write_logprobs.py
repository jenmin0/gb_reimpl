from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "TinyLlama/TinyLlama_v1.1" #"meta-llama/Llama-2-7b-hf"
llama_tokenizer = AutoTokenizer.from_pretrained(model_name)
llama_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16
    ).to(device)


def write_llama_logprobs(text, out, model):
    """
    Run text under model and write logprobs to file, separated by newline.
    """
    with torch.no_grad():
        encodings = llama_tokenizer(text, return_tensors="pt", max_length=2048, truncation=True)
        input_ids = encodings["input_ids"].to(device)
        output = model(input_ids)
        log_probs = F.log_softmax(output.logits, dim=-1)

        tokens = encodings["input_ids"]
        indices = torch.tensor([[[i] for i in tokens[0]]])[:, 1:, :].to(device)

        # 1 to N tokens
        target_ids = input_ids[0, 1:]
        logprob_pro_token = log_probs[0, :-1, :].gather(1, target_ids.unsqueeze(1)).squeeze(1).cpu().numpy()
        tokens = llama_tokenizer.convert_ids_to_tokens(target_ids.cpu())

    to_write = ""
    for _, (w, p) in enumerate(zip(tokens, logprob_pro_token)):
        to_write += f"{w} {-p}\n"

    with open(out, "w") as f:
        f.write(to_write)


def is_txt_file(file):
    return file.endswith(".txt")

def process_folder(folder, llama_model):
    if folder.endswith("prompts"):
        return
    # print(f"in Folder {folder}")
    # Check if folder logprob exists
    logprobs_dir = os.path.join(folder, "logprobs")
    if not os.path.exists(logprobs_dir):
        os.makedirs(logprobs_dir)
    # read all txt files
    for fname in os.listdir(folder):
        if is_txt_file(fname):
            full_path = os.path.join(folder, fname)
            print(f"Processing {full_path}")
            with open(full_path, "r") as f:
                text = f.read().strip()

        # output path
        #out_llama = os.path.join(logprobs_dir, fname.replace(".txt", "-llama.txt"))
        out_tinyllama = os.path.join(logprobs_dir, fname.replace(".txt", "-tinyllama.txt"))
        # out_ds = os.path.join(logprobs_dir, fname.replace(".txt", "-ds.txt"))

        write_llama_logprobs(text, out_tinyllama, llama_model)
        # write_ds_logprobs(text, out_ds)

def traverse_and_generate(data_root, llama_model):
    for root, dirs, files in os.walk(data_root):

        # Skip folders which do not need generate logprobs
        if os.path.basename(root) in ("prompt", "logprobs", "headlines", "logprobs_babbage","prompts"):
            continue

        # Check if txt files contained
        if any(is_txt_file(f) for f in files):
            process_folder(root, llama_model)

if __name__ == "__main__":
    # Generate Logprobs
    traverse_and_generate("data/essay", llama_model)
