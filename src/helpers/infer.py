import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def infer(run_path, eval_prompt: str):

    base_model_id = "microsoft/phi-2"
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,  # Phi2, same as before
        device_map="auto",
        trust_remote_code=True,
        load_in_8bit=True,
        torch_dtype=torch.float16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
        use_fast=False, # needed for now, should be fixed soon
    )
    tokenizer.pad_token = tokenizer.eos_token

    eval_tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True, use_fast=False)
    eval_tokenizer.pad_token = tokenizer.eos_token   
    

    ft_model = PeftModel.from_pretrained(base_model, run_path)    
    
    model_input = eval_tokenizer(eval_prompt, return_tensors="pt").to("cuda")

    ft_model.eval()
    with torch.no_grad():
        return(eval_tokenizer.decode(ft_model.generate(**model_input, 
                                                      max_new_tokens=100, 
                                                      repetition_penalty=1.11)[0], skip_special_tokens=True))