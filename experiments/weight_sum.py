data_config = [
    ("arxiv", 0.0169),
    ("book", 0.0318),
    ("c4", 0.1059),
    ("cc", 0.6339),  # 0.5772 + 0.0568
    ("github", 0.0159),
    # ("starcoder", 0.0568),
    ("stackexchange", 0.0184),
    ("wiki", 0.0551),

    ("amps", 0.0003),
    ("Bookcorpus", 0.0006),
    ("fanfics", 0.0021),
    ("openwebmath", 0.0085),
    ("pubmed", 0.0106),

    # 中文
    # ("law", 0.0067),
    # ("news", 0.0033),
    # ("patent", 0.0033),
    # ("webtext", 0.0430),
    # ("PTD", 0.0437),

    # 小语种
    # ("culturax", 0.08),
    # ("wiki_multi_lang", 0.02),
]
sum_1 = sum_2 = 0.0

for item in data_config:
    sum_1 += item[1]
    print(item[0], round(item[1] * 90 / 85, 4))
    sum_2 += round(item[1] * 90 / 85, 4)

print(sum_1, sum_2)
