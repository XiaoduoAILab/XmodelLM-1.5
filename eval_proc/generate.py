import argparse
import transformers
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed

import sys
from pathlib import Path
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
from models.modeling_xmodel import XModelForCausalLM
from models.configuration_xmodel import XModelConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="hello, i am", help="")
    parser.add_argument("--device", type=str, default="cuda:6", help="")
    args = parser.parse_args()

    prompt, device = args.prompt, args.device

    prompt = "Once upon a time, in a barn near a farm house, there lived a little white kitten named Cotton. Cotton lived high up in a nice warm place above the barn where all of the farmer's horses slept. But Cotton wasn't alone in her little home above the barn, oh no. She shared her hay bed with her mommy and 5 other sisters. All of her sisters were cute and fluffy, like Cotton. But she was the only white one in the bunch. The rest of her sisters were all orange with beautiful white tiger stripes like Cotton's mommy. Being different made Cotton quite sad. She often wished she looked like the rest of her family. So one day, when Cotton found a can of the old farmer's orange paint, she used it to paint herself like them. When her mommy and sisters found her they started laughing. \n\n\"What are you doing, Cotton?!\" \n\n\"I only wanted to be more like you\". \n\nCotton's mommy rubbed her face on Cotton's and said \"Oh Cotton, but your fur is so pretty and special, like you. We would never want you to be any other way\". And with that, Cotton's mommy picked her up and dropped her into a big bucket of water. When Cotton came out she was herself again. Her sisters licked her face until Cotton's fur was all all dry. \n\n\"Don't ever do that again, Cotton!\" they all cried. \"Next time you might mess up that pretty white fur of yours and we wouldn't want that!\" \n\nThen Cotton thought, \"I change my mind. I like being special\".\n\nQuestion: Where did she live?\nAnswer: "
    prompt = "Sportsman Jhonathan Florez jumped from a helicopter above Bogota, the capital of Colombia, on Thursday. = Le sportif Jhonathan Florez a sauté jeudi d'un hélicoptère au-dessus de Bogota, la capitale colombienne.\n\nWearing a wingsuit, he flew past over the famous Monserrate Sanctuary at 160km/h. The sanctuary is located at an altitude of over 3000 meters and numerous spectators had gathered there to watch his exploit. = Equipé d'un wingsuit (une combinaison munie d'ailes), il est passé à 160 km/h au-dessus du célèbre sanctuaire Monserrate, situé à plus de 3 000 mètres d'altitude, où de nombreux badauds.\n\nA black box in your car? = Une boîte noire dans votre voiture ?\n\nThis really is a must for our nation. = Cela est vraiment indispensable pour notre nation.\n\nThe usually dull arena of highway planning has suddenly spawned intense debate and colorful alliances. = Le secteur généralement sans intérêt de la planification des grands axes a soudain provoqué un débat fort animé et des alliances mouvementées.\n\n"
    prompt += "Some transportation planners, though, wonder if all the talk about paying by the mile is just a giant distraction. = "

    model_dir = 'working_dir'
    model_name = 'medium'
    tokenizer = transformers.AutoTokenizer.from_pretrained('/data/wangyichuan/llm_pretrain_xmodel/tokenizers/llama', use_fast=False, trust_remote_code=True)
    # print(tokenizer)

    config = XModelConfig.from_name(model_name)
    model = XModelForCausalLM(config)
    PATH = 'working_dir_%s/pytorch_model.bin'%model_name
    model.load_state_dict(torch.load(PATH))
    model.eval()
    model.to(device)
    print('model loaded')

    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    num_tokens = inputs.shape[-1]
    outputs = model.generate(inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(outputs[0][num_tokens:]).strip()

    print('prompt: \n', prompt)
    print('answer: \n', text)

    generator = pipeline('text-generation', model='gpt2')
    set_seed(42)
    print('gpt2:\n', generator(prompt, max_length=300, num_return_sequences=1))
