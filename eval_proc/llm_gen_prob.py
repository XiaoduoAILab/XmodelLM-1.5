import argparse
import json
import os
import random

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_gen_visualizer import visulize_text
from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = 'cuda:7'

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

    model_path = "/data/wangyichuan/llm_pretrain_xmodel/gpt/gpt2"
    tokenizer_path = "/data/wangyichuan/llm_pretrain_xmodel/gpt/gpt2"

    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
    model.eval()

    print(model)
    print(tokenizer)

    sent = "Once upon a time, in a barn near a farm house, there lived a little white kitten named Cotton. Cotton lived high up in a nice warm place above the barn where all of the farmer's horses slept. But Cotton wasn't alone in her little home above the barn, oh no. She shared her hay bed with her mommy and 5 other sisters. All of her sisters were cute and fluffy, like Cotton. But she was the only white one in the bunch. The rest of her sisters were all orange with beautiful white tiger stripes like Cotton's mommy. Being different made Cotton quite sad. She often wished she looked like the rest of her family. So one day, when Cotton found a can of the old farmer's orange paint, she used it to paint herself like them. When her mommy and sisters found her they started laughing. \n\n\"What are you doing, Cotton?!\" \n\n\"I only wanted to be more like you\". \n\nCotton's mommy rubbed her face on Cotton's and said \"Oh Cotton, but your fur is so pretty and special, like you. We would never want you to be any other way\". And with that, Cotton's mommy picked her up and dropped her into a big bucket of water. When Cotton came out she was herself again. Her sisters licked her face until Cotton's fur was all all dry. \n\n\"Don't ever do that again, Cotton!\" they all cried. \"Next time you might mess up that pretty white fur of yours and we wouldn't want that!\" \n\nThen Cotton thought, \"I change my mind. I like being special\".\n\n"
    sent += "Question: What color was Cotton?\n\nAnswer: white\n\n"
    sent += "Question: Where did she live?\n\nAnswer: in a barn\n\n"
    sent += "Question: Did she live alone?\n\nAnswer: no\n\n"
    sent += "Question: Who did she live with?\n\nAnswer: with her mommy and 5 sisters\n\n"
    sent += "Question: What color were her sisters?\n\nAnswer: orange and white\n\n"
    result = generate_one_completion(sent)

    with open('output.json', 'w') as fp:
        json.dump(result, fp)

    visulize_text(sent, result)
