from bytepiece import Tokenizer
tokenizer1 = Tokenizer('tokenizers/xmodel2/tokenizer.model')
tokenizer1.convert_to_sentencepiece('tokenizers/xmodel2/tokenizer_sp.model')

import sentencepiece as spm
tokenizer2 = spm.SentencePieceProcessor('tokenizers/xmodel2/tokenizer_sp.model')

tokenizer1.encode('今天天气不错')
tokenizer2.encode('今天天气不错')