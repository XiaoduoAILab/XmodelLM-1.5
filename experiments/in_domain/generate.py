import argparse

import torch
import transformers
from transformers import StoppingCriteriaList


def custom_stopping_criteria(input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
    if input_ids[0][-1] == NEW_LINE_TOKEN_ID:
        return True

    return False


@torch.no_grad()
def generate(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    num_tokens = inputs.shape[-1]
    outputs = model.generate(inputs, max_new_tokens=args.seq_len, pad_token_id=tokenizer.eos_token_id,
                             stopping_criteria=StoppingCriteriaList([custom_stopping_criteria]))
    text = tokenizer.decode(outputs[0][num_tokens:]).strip()
    return text


if __name__ == "__main__":
    NEW_LINE_TOKEN_ID = 97

    parser = argparse.ArgumentParser(description='')
    # general
    parser.add_argument('--model_path', type=str,
                        default='/data1/liuyang/llm_pretrain_xmodel/out_xdata_f_line/data_mix_0_to_1/iter-0064000/')
    parser.add_argument('--prompt_path', type=str, default='experiments/in_domain/input.txt', help='')
    parser.add_argument('--seq_len', type=int, default=20, help='')
    args = parser.parse_args()
    print(args)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)

    prompt = open(args.prompt_path, 'r').read()
    print('prompt: ' + str(prompt))

    text = generate(prompt)
    print('text: ' + str(text))


