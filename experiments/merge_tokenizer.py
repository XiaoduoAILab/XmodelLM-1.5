
import json

if __name__ == "__main__":
    with open('tokenizers/llama/tokenizer.json', 'r', encoding='utf-8') as fp:
        data = json.load(fp)

    print(list(data.keys()))
    vocab = data['model']['vocab']
    print(vocab)

    with open('tokenizers/xmodel_32000/xmodel_32000.vocab', 'r', encoding='utf-8') as fp:
        lines = fp.readlines()

    print(len(lines))

    for line in lines:
        ch = line.split('\t')[0]
        if not ch in vocab:
            vocab[ch] = len(vocab)

    print('vocab: ' + str(vocab))
    print('len(vocab): ' + str(len(vocab)))

    data['model']['vocab'] = vocab

    with open('tokenizers/xmodel_32000/tokenizer.json', 'w', encoding='utf-8') as fp:
        json.dump(data, fp, indent=2)