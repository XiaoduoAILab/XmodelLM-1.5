import os
import argparse
import transformers
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM,  StoppingCriteriaList, pipeline, set_seed

from models.modeling_xmodel import XModelForCausalLM
from models.configuration_xmodel import XModelConfig

def custom_stopping_criteria(input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
    if input_ids[0][-1] == 68:  # <|im_end|>
        return True
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="xl", help="")
    parser.add_argument("--model_path", type=str, default="", help="")
    parser.add_argument("--prompt", type=str, default="The capital of China is", help="")
    parser.add_argument("--max_new_tokens", type=int, default=10, help="")
    parser.add_argument("--device", type=str, default="cuda:0", help="")
    args = parser.parse_args()

    model_name, model_path, prompt, max_new_tokens, device = args.model_name, args.model_path, args.prompt, args.max_new_tokens, args.device
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    
    ckp_path = f'{args.model_path}/pytorch_model.bin'
    if not os.path.isfile(ckp_path):
        # instruct model 
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
        messages = [{"role":'user','content':prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # pretrain model
        config = XModelConfig.from_name(model_name)
        model = XModelForCausalLM(config)
        model.load_state_dict(torch.load('%s/pytorch_model.bin'%model_path))
        model.eval()
        model.to(device)

    print('model loaded!')

    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    num_tokens = inputs.shape[-1]
    outputs = model.generate(inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id,stopping_criteria=StoppingCriteriaList([custom_stopping_criteria]),)
    text = tokenizer.decode(outputs[0][num_tokens:]).strip()

    print('prompt: \n', prompt)
    print('response: \n', text)