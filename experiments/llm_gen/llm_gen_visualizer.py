import argparse
import json

import matplotlib.pyplot as plt


def visulize_text(sent, data):
    Blues = plt.get_cmap('viridis')

    html = ''
    for item in data:
        token = item['token']
        length = len(token)
        position = sent.index(token)
        token = sent[:position + length]
        token = token.replace('\n', '\\n<br/>')
        sent = sent[position + length:]

        prob = item['prob']
        r, g, b, _ = Blues(prob)
        r, g, b = int(r * 255), int(g * 255), int(b * 255),
        color = '#%02x%02x%02x' % (r, g, b)
        html += '''<span style="color: white; background-color: {}">{}</span>'''.format(color, token)

    with open('output.html', 'w') as fp:
        fp.write(html)


if __name__ == "__main__":
    # sent = 'The prime minister of the United Kingdom is David Cameron.'
    # data = [{'token': 'The', 'prob': 0}, {'token': 'prime', 'prob': 0.00014892354374751449},
    #         {'token': 'minister', 'prob': 0.5262304544448853}, {'token': 'of', 'prob': 0.0444093756377697},
    #         {'token': 'the', 'prob': 0.1427033245563507}, {'token': 'United', 'prob': 0.1501968950033188},
    #         {'token': 'Kingdom', 'prob': 0.8870892524719238}, {'token': 'is', 'prob': 0.1316700279712677},
    #         {'token': 'David', 'prob': 0.03595937788486481}, {'token': 'Camer', 'prob': 0.9657541513442993},
    #         {'token': 'on', 'prob': 0.9995817542076111}, {'token': '.', 'prob': 0.5228220820426941}]

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="output.json")
    args = parser.parse_args()

    print(args)

    with open(args.data_file, 'r') as fp:
        data = json.load(fp)
        tokens = [it['token'] for it in data]
        sent = ''.join(tokens)

    visulize_text(sent, data)
