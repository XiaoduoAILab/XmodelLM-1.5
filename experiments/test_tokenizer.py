from transformers import AutoTokenizer

if __name__ == "__main__":
    tokenizer_path = 'out_f_line/micro_run6_gc_0_4/iter-0250000'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, add_bos_token=False)

    toks = tokenizer.encode('')
    print('toks: ' + str(toks))

    toks = tokenizer.encode('h')
    print('toks: ' + str(toks))

    toks = tokenizer.encode('hello')
    print('toks: ' + str(toks))

    token = tokenizer.decode([1, 343])
    print('token: ' + str(repr(token)))

    token = tokenizer.decode([23405])
    print('token: ' + str(repr(token)))

    token = tokenizer.decode([343])
    print('token is None: ' + str(token is None))
    print('token == '': ' + str(token == ''))

    tokenizer_path = 'out_f_line/micro_run9_ccdata_v8_test4_128000/iter-0000001'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    print(tokenizer)

    toks = tokenizer.encode('')
    print('toks: ' + str(toks))

    toks = tokenizer.encode('h')
    print('toks: ' + str(toks))

    toks = tokenizer.encode('hello')
    print('toks: ' + str(toks))
