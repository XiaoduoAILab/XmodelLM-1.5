import gradio as gr
import torch
import transformers
from transformers import StoppingCriteriaList

device = "cuda:0"
model_path = "/data2/liuyang/llm_pretrain_xmodel/out_f_line/xl_f_line/iter-0084000/"

tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
model = transformers.AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
model = model.eval()


def custom_stopping_criteria(input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
    if input_ids[0][-1] == 97:
        return True

    return False


def parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def predict(input, chatbot, max_length, top_p, temperature, history):
    print('input: ' + str(input))
    print('history: ' + str(history))
    chatbot.append((parse_text(input), ""))
    history.append(input)
    prompt = input
    print('prompt: ' + str(prompt))
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    num_tokens = inputs.shape[-1]
    outputs = model.generate(inputs, max_length=max_length, top_p=top_p, do_sample=True,
                             temperature=temperature,
                             stopping_criteria=StoppingCriteriaList([custom_stopping_criteria]))
    print('outputs[0]: ' + str(outputs[0]))
    response = tokenizer.decode(outputs[0][num_tokens:-1]).strip()
    history.append(response)
    print('response: ' + str(response))

    # for response, history in:
    #     chatbot[-1] = (parse_text(input), parse_text(response))
    #
    #     yield chatbot, history

    chatbot[-1] = (parse_text(input), parse_text(response))
    return chatbot, history


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], []


with gr.Blocks() as demo:
    demo.title = 'F-line (xl)--iter-0156000'
    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                    container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(0, 4096, value=32, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)

    history = gr.State([])

    submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history], [chatbot, history],
                    show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)

demo.queue().launch(server_name='0.0.0.0', server_port=30004)
