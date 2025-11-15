import argparse
import math
import time

import kenlm
import numpy as np

import text_normalizer


def get_argument_parser():
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default='/data/wangyichuan/cc_net/data/lm_sp/en.arpa.bin',
                        help="model path")
    parser.add_argument("--prompt", type=str, default="language modeling is fun.", help="prompt")

    return parser


if __name__ == "__main__":
    # model_path = '/data1/liuyang/code/kenlm-master/build/text.arpa'
    # model_path = '/data/wangyichuan/cc_net/data/lm_sp/en.arpa.bin'
    arg_parser = get_argument_parser()
    args = arg_parser.parse_args()

    # Load model trained on English wikipedia
    start_load = time.time()
    print(f"Loading {args.model_path}...")
    model = kenlm.LanguageModel(args.model_path)
    load_time = time.time() - start_load
    print(f"Loaded {args.model_path} (took {load_time / 60:.1f}min)")

    print('{0}-gram model'.format(model.order))

    # Get perplexity
    # sentence = "I am very perplexed"
    prompt = args.prompt
    print('prompt: ' + str(prompt))
    line = text_normalizer.normalize(prompt)
    print('line: ' + str(line))
    length = len(line.split()) + 1
    print('length: ' + str(length))
    log_score = model.score(line)
    print('log_score: ' + str(log_score))


    # Check that total full score = direct score
    def score(s):
        return sum(prob for prob, _, _ in model.full_scores(s))


    assert (abs(score(line) - model.score(line)) < 1e-3)

    # Show scores and n-gram matches
    words = ['<s>'] + line.split() + ['</s>']
    for i, (prob, length, oov) in enumerate(model.full_scores(line)):
        print('{0} {1}: {2}'.format(prob, length, ' '.join(words[i + 2 - length:i + 2])))
        if oov:
            print('\t"{0}" is an OOV'.format(words[i + 1]))

    # Find out-of-vocabulary words
    for w in words:
        if not w in model:
            print('"{0}" is an OOV'.format(w))

    ppl = model.perplexity(line)
    print('ppl: ' + str(ppl))
    # sentence = "im hella trippin"
    # ppl = model.perplexity("im hella trippin")
    # # 46793.5 (high perplexity, since the sentence is colloquial and contains grammar mistakes)
    # print(sentence)
    # print(ppl)

    sentence = " ".join(line)
    print('sentence: ' + str(sentence))
    m = model
    s = line

    ppl = m.perplexity(sentence)
    print('ppl: ' + str(ppl))
