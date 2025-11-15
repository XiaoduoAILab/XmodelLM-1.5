from bytepiece import Trainer


class corpus:
    def __iter__(self):
        with open(data_path) as f:
            for l in f:
                yield l  # 每次返回一个Unicode


# data_path = '/data4/mishiqian/data/msg.txt'
data_path = 'train.txt'
trainer = Trainer(order=6, max_vocab_size=32000, min_count=32, isolate_digits=True)
trainer.train(corpus(), workers=64, batch_size=1000)
trainer.save('xmodel_32000.model')
