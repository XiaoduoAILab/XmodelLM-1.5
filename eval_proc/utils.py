import glob
import json
import os
import shutil

import numpy as np
import torch
from tqdm import tqdm

wd = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
import torch.nn.functional as F


def eval_ppl(dataset, model, tokenizer, device='cuda'):
    encodings = tokenizer("\n\n".join(dataset), return_tensors="pt")

    # max_length = model.config.n_positions
    # max_length = 1024
    max_length = 512
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over input tokens.
            # Multiply it with trg_len to get the summation instead of average.
            # We will take average over all the tokens to get the true average
            # in the last step of this example.
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    nlls = [num.item() for num in nlls]
    # print(nlls)
    avg_loss = sum(nlls) / end_loc
    # print('avg_loss: ' + str(avg_loss))
    ppl = np.exp(avg_loss)

    # ppl: tensor(25.1704, device='cuda:0')
    return ppl


def eval_ppl_sent(text, model, tokenizer, device='cuda:6'):
    encodings = tokenizer(text, return_tensors="pt")

    input_ids = encodings.input_ids.to(device)
    target_ids = input_ids.clone()

    with torch.no_grad():
        model.eval()
        outputs = model(input_ids, labels=target_ids)
    nll = outputs.loss
    nll = nll.item()
    ppl = np.exp(nll)
    return ppl


def eval_acc(dataset, model, tokenizer, device='cuda'):
    encodings = tokenizer("\n\n".join(dataset), return_tensors="pt")

    max_length = stride = 512
    seq_len = encodings.input_ids.size(1)

    accs, tol = 0, 0
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        target_ids = target_ids[:, 1:]

        with torch.no_grad():
            model.eval()
            logits = model(input_ids).logits
            probabilities = F.softmax(logits, dim=-1)
            predicted_ids = torch.argmax(probabilities, dim=-1)[:, :-1]
            correct = (predicted_ids == target_ids).float()

        accs += correct.sum().cpu().numpy()
        tol += correct.size()[1]

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    return accs / tol


def deploy_model(model_dir, eval_name, tokenizer_path='tokenizers/llama'):
    folder = 'working_dir_%s' % eval_name
    if not os.path.exists(folder):
        os.mkdir(folder)

    files = []
    for file in glob.glob(model_dir + '/*'):
        files.append(file)

    for file in glob.glob(f'{tokenizer_path}/*'):
        files.append(file)

    files.append('models/configuration_xmodel.py')
    files.append('models/modeling_xmodel.py')

    # print(files)

    for file in files:
        # if not file.endswith('model.safetensors'):
        # print(file)
        shutil.copy(file, folder)

    # print(f'changing attributes in {folder}/config.json')
    # with open(os.path.join(folder, 'config.json'), 'r') as fp:
    #     data = json.load(fp)

    # data['auto_map'] = dict(AutoConfig="configuration_xmodel.XModelConfig",
    #                         AutoModelForCausalLM="modeling_xmodel.XModelForCausalLM")
    # data['torch_dtype'] = 'bfloat16'

    # with open(os.path.join(folder, 'config.json'), 'w') as fp:
    #     json.dump(data, fp, indent=2, ensure_ascii=False)
    # print('done')


def deploy_code(ckp_folder):
    files = ['models/configuration_xmodel.py', 'models/modeling_xmodel.py']

    for file in files:
        shutil.copy(file, ckp_folder)

    with open(os.path.join(ckp_folder, 'config.json'), 'r') as fp:
        data = json.load(fp)

    data['auto_map'] = dict(AutoConfig="configuration_xmodel.XModelConfig",
                            AutoModelForCausalLM="modeling_xmodel.XModelForCausalLM")

    with open(os.path.join(ckp_folder, 'config.json'), 'w') as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)


def extract_embeddings(model, tokenizer, texts, batch_size=8):
    model.eval()

    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    all_sent_embedding = []
    for batch in batches:
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            last_hidden_state = model(**inputs, output_hidden_states=True).hidden_states[-1]

        weights_for_non_padding = inputs.attention_mask * torch.arange(start=1,
                                                                       end=last_hidden_state.shape[1] + 1).unsqueeze(
            0).to(model.device)

        sum_embeddings = torch.sum(last_hidden_state * weights_for_non_padding.unsqueeze(-1), dim=1)
        num_of_none_padding_tokens = torch.sum(weights_for_non_padding, dim=-1).unsqueeze(-1)
        sentence_embeddings = sum_embeddings / num_of_none_padding_tokens
        all_sent_embedding.append(sentence_embeddings.to('cpu'))

    return torch.cat(all_sent_embedding, axis=0)
