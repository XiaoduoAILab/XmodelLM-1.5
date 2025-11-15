import argparse
import json
import os
import random

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_gen_visualizer import visulize_text


@torch.no_grad()
def generate_one_completion(prompt):
    encoding = tokenizer.encode(prompt, return_tensors="pt")
    # print('encoding: ' + str(encoding))
    num_tokens = encoding.shape[-1]
    print('num_tokens: ' + str(num_tokens))

    result = [dict(token=tokenizer.decode([encoding[0][0]]), prob=0)]
    for i in tqdm(range(1, num_tokens)):
        inputs = encoding[:, :i].to(model.device)
        # print('inputs: ' + str(inputs))
        # print('inputs.shape: ' + str(inputs.shape))
        outputs = model.generate(inputs, max_new_tokens=1, return_dict_in_generate=True, output_scores=True,
                                 pad_token_id=tokenizer.eos_token_id)
        logits = outputs.scores[0][0]
        logits = torch.nn.functional.softmax(logits, dim=0, dtype=torch.float32)
        token_id = encoding[0][i]
        # print('token_id: ' + str(token_id))
        token = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        # print('token: ' + str(token))
        prob = logits[token_id].detach().cpu().item()
        # print('prob: ' + str(prob))
        result.append(dict(token=token, prob=prob))

    print(result)

    return result


if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="google/flan-t5-small")
    # parser.add_argument("--sent", type=str, default="The prime minister of the United Kingdom is David Cameron.")
    parser.add_argument("--prompt_file", type=str, default="input.txt")
    args = parser.parse_args()

    print(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              skip_special_tokens=True,
                                              add_bos_token=False,
                                              add_eos_token=False,
                                              use_fast=False,
                                              trust_remote_code=True)

    print(f"Loading model from {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model,
                                                 torch_dtype=torch.float16,
                                                 trust_remote_code=True,
                                                 device_map='auto')
    model.eval()

    print(model)
    print(tokenizer)

    # sent = 'The prime minister of the United Kingdom is David Cameron.'
    sent = open(args.prompt_file, "r").read()
    result = generate_one_completion(sent)

    with open('output.json', 'w') as fp:
        json.dump(result, fp)

    visulize_text(sent, result)
