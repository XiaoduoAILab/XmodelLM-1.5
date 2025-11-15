import argparse

from model import KenlmModel


def get_argument_parser():
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default='/data/wangyichuan/cc_net/data/lm_sp/',
                        help="model path")
    parser.add_argument("--prompt", type=str, default="language modeling is fun.", help="prompt")

    return parser


if __name__ == "__main__":
    arg_parser = get_argument_parser()
    args = arg_parser.parse_args()

    # Load model trained on English wikipedia
    # model_dataset = "/data/wangyichuan/cc_net/data/lm_sp"
    # model_dataset = '/data1/liuyang/llm_pretrain_xmodel/lm'
    model = KenlmModel(args.model_path, "en")

    # Get perplexity
    sentence = "I am very perplexed"
    ppl = model.get_perplexity(sentence)
    print(sentence)
    print(ppl)
    # 341.3 (low perplexity, since sentence style is formal and with no grammar mistakes)

    sentence = "im hella trippin"
    ppl = model.get_perplexity("im hella trippin")
    # 46793.5 (high perplexity, since the sentence is colloquial and contains grammar mistakes)
    print(sentence)
    print(ppl)
