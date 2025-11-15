import argparse

from models.configuration_xmodel import XModelConfig
from models.modeling_xmodel import XModelForCausalLM
from utils import get_model_numel, format_numel_str, print_model_ratio, calculate_norm, flops_calculator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='model')
    # general
    parser.add_argument('--model', type=str, default='7B')
    parser.add_argument('--vocab_size', type=int, default=32000)
    args = parser.parse_args()

    config = XModelConfig.from_name(args.model)
    config.vocab_size = args.vocab_size
    model = XModelForCausalLM(config)

    print(config)
    print(model)

    total_params = get_model_numel(model)
    print('total params: ' + str(total_params))
    params_non_embedding = total_params - 2 * config.vocab_size * config.hidden_size
    print('params(non-embedding): ' + str(params_non_embedding))

    flops_per_token_2048 = model.flops_per_token(max_length=2048, non_embedding=False)
    print('flops per token(DDP) block size=2048: ' + str(flops_per_token_2048))

    flops_per_token_4096 = model.flops_per_token(max_length=4096, non_embedding=False)
    print('flops per token(DDP) block size=4096: ' + str(flops_per_token_4096))

    flops_per_token_2048 = flops_calculator(config, batch_size=1, seq_len=2048)
    print('flops per token(Megatron-Deepspeed) block size=2048: ' + str(flops_per_token_2048 / 2048))

    numel_str = format_numel_str(total_params)
    print('numel_str: ' + str(numel_str))

    print_model_ratio(model)

    l1_norm, l2_norm = calculate_norm(model)
    print('\nl1_norm: ' + str(l1_norm))
    print('l2_norm: ' + str(l2_norm))

    # print('\nsparsity:')
    # sparsity_dict = calculate_sparsity(model)
    # for key in sparsity_dict.keys():
    #     print('\t{}: {:.8f}'.format(key, sparsity_dict[key]))
    #
    # print('\northog_mse:')
    # orthog_mse_dict = calculate_orthog_mse(model)
    # for key in orthog_mse_dict.keys():
    #     print('\t{}: {:.8f}'.format(key, orthog_mse_dict[key]))
