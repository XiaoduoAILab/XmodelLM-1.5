
# tokenizer_path = r'tokenizers\xmodel\v6\xmodel_32000\xmodel_32000.vocab'
tokenizer_path = r'tokenizers\xmodel\v8\xmodel_64000\xmodel_64000.vocab'
with open(tokenizer_path, 'r', encoding='utf-8') as fp:
    lines = fp.readlines()

# print(lines)

tokens = set()
dup = set()
for line in lines:
    toks = line.strip().split('\t')
    tok = toks[0]
    tok = tok.replace('▁', '')
    if tok in tokens:
        dup.add(tok)
    tokens.add(tok)

print(len(tokens))
print(len(dup))
print(dup)