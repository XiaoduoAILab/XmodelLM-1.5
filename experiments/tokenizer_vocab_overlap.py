from transformers import AutoTokenizer


if __name__ == "__main__":
    tokenizer_llama = AutoTokenizer.from_pretrained('tokenizers/llama')
    vocab_llama = set(tokenizer_llama.get_vocab().keys())

    tokenizer_baichuan1 = AutoTokenizer.from_pretrained('tokenizers/baichuan1', trust_remote_code=True)
    vocab_baichuan1 = set(tokenizer_baichuan1.get_vocab().keys())

    tokenizer_baichuan2 = AutoTokenizer.from_pretrained('tokenizers/baichuan2', trust_remote_code=True)
    print(tokenizer_baichuan2.vocab_size)
    vocab_baichuan2 = set(tokenizer_baichuan2.get_vocab().keys())

    tokenizer_xmodel = AutoTokenizer.from_pretrained('tokenizers/xmodel_32000', trust_remote_code=True)
    vocab_xmodel = set(tokenizer_xmodel.get_vocab().keys())

    tokenizer_xmodel2 = AutoTokenizer.from_pretrained('tokenizers/xmodel2', trust_remote_code=True)
    vocab_xmodel2 = set(tokenizer_xmodel2.get_vocab().keys())

    overlap_llama_baichuan1 = set.intersection(vocab_llama, vocab_baichuan1)
    # print(overlap_llama_baichuan1)
    print('len(overlap_llama_baichuan1): ' + str(len(overlap_llama_baichuan1)))

    overlap_llama_baichuan2 = set.intersection(vocab_llama, vocab_baichuan2)
    # print(overlap_llama_baichuan1)
    print('len(overlap_llama_baichuan2): ' + str(len(overlap_llama_baichuan2)))

    overlap_llama_xmodel = set.intersection(vocab_llama, vocab_xmodel)
    print('len(overlap_llama_xmodel): ' + str(len(overlap_llama_xmodel)))

    overlap_llama_xmodel2 = set.intersection(vocab_llama, vocab_xmodel2)
    print('len(overlap_llama_xmodel2): ' + str(len(overlap_llama_xmodel2)))

    overlap_baichuan1_baichuan2 = set.intersection(vocab_baichuan1, vocab_baichuan2)
    print('len(overlap_baichuan1_baichuan2): ' + str(len(overlap_baichuan1_baichuan2)))

    overlap_xmodel_baichuan1 = set.intersection(vocab_xmodel, vocab_baichuan1)
    print('len(overlap_xmodel_baichuan1): ' + str(len(overlap_xmodel_baichuan1)))

    overlap_xmodel_baichuan2 = set.intersection(vocab_llama, vocab_baichuan2)
    print('len(overlap_xmodel_baichuan2): ' + str(len(overlap_xmodel_baichuan2)))

    overlap_xmodel2_baichuan1 = set.intersection(vocab_xmodel2, vocab_baichuan1)
    print('len(overlap_xmodel2_baichuan1): ' + str(len(overlap_xmodel2_baichuan1)))

    overlap_xmodel2_baichuan2 = set.intersection(vocab_xmodel2, vocab_baichuan2)
    print('len(overlap_xmodel2_baichuan2): ' + str(len(overlap_xmodel2_baichuan2)))

    overlap_xmodel_xmodel2 = set.intersection(vocab_xmodel, vocab_xmodel2)
    print('len(overlap_xmodel_xmodel2): ' + str(len(overlap_xmodel_xmodel2)))


