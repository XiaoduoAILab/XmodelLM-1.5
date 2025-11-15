import argparse
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_argument_parser():
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='/data1/liuyang/checkpoints/e_line/xl/iter-0596000/',
                        help="model path")
    parser.add_argument("--tokenizer_version", type=str, default='v8', help="tokenizer version")
    parser.add_argument("--vocab_size", type=int, default=64000, help="vocab size")
    parser.add_argument("--new_tokenizer_path", type=str, default=None, help="new tokenizer path")
    parser.add_argument("--output_path", type=str, default='extend_vocab_output', help="output path")
    return parser


if __name__ == "__main__":
    arg_parser = get_argument_parser()
    args = arg_parser.parse_args()

    model_path = args.model_path
    tokenizer_version = args.tokenizer_version
    vocab_size = args.vocab_size
    output_path = args.output_path

    Path(output_path).mkdir(parents=True, exist_ok=True)

    original_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    original_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    new_tokenizer_path = args.new_tokenizer_path
    if new_tokenizer_path is None:
        new_tokenizer_path = f'tokenizers/xmodel/{tokenizer_version}/xmodel_{vocab_size}'
    print("new_tokenizer_path: " + str(new_tokenizer_path))
    new_tokenizer = AutoTokenizer.from_pretrained(new_tokenizer_path, trust_remote_code=True, use_cache=False)
    original_vocab = original_tokenizer.get_vocab()
    new_vocab = new_tokenizer.get_vocab()

    # 创建新的模型，调整词嵌入层大小
    new_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    new_model.resize_token_embeddings(vocab_size)

    # 复制原始模型的词嵌入到新的词嵌入层（假设有重叠的部分）
    original_embed_tokens = original_model.model.embed_tokens.weight.data
    original_lm_head = original_model.lm_head.weight.data
    new_embed_tokens = new_model.model.embed_tokens.weight.data
    new_lm_head = new_model.lm_head.weight.data

    # 映射重叠部分的词嵌入
    num_overlap = 0
    for word, idx in tqdm(new_vocab.items()):
        if word in original_vocab:
            original_idx = original_vocab[word]
            new_embed_tokens[idx] = original_embed_tokens[original_idx]
            new_lm_head[idx] = original_lm_head[original_idx]
            num_overlap += 1

    print('num_overlap: ' + str(num_overlap))

    # 保存模型，供重新预训练和微调
    new_model.save_pretrained(output_path, safe_serialization=False)
    new_tokenizer.save_pretrained(output_path)
