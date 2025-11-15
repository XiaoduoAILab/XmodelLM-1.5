import json
from math import log

import matplotlib.pyplot as plt
import numpy as np


def smooth(scalars, alpha=0.01):
    result = []
    for i in range(1, len(scalars)):
        result.append(np.mean(scalars[0:i]) * (1 - alpha) + scalars[i] * alpha)
    return result


if __name__ == "__main__":
    with open('output.json', 'r') as fp:
        data = json.load(fp)

    probs = [item["prob"] for item in data]
    probs = [1 / 32000 if prob == 0 else prob for prob in probs]
    loss = [-log(prob) for prob in probs]
    print('np.mean(loss): ' + str(np.mean(loss)))

    example_seps = []
    question_answer_seps = []
    start = 0
    for i in range(1, len(loss)):
        if data[i]['token'] == '\n' and data[i - 1]['token'] == '\n':
            example_seps.append(i)
            print(np.mean(loss[start:i]))
            start = i

        if i == len(loss) - 1:
            print(np.mean(loss[start:i]))

        if data[i]['token'] == '\n' and data[i - 1]['token'] == 'step':
            question_answer_seps.append(i)

    losses = smooth(loss)

    for s in example_seps:
        plt.axvline(x=s, color='b', label='example seps')

    for s in question_answer_seps:
        plt.axvline(x=s, color='green', ls=':', label='question-answer seps')

    plt.plot(losses)
    plt.title('In-Context Learning Curve')
    plt.xlabel('token index')
    plt.ylabel('loss')
    plt.show()
