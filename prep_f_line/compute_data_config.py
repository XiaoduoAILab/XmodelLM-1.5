if __name__ == "__main__":
    data_config = [
        ("arxiv", 0.0151),
        ("book", 0.0282),
        ("c4", 0.0941),
        ("cc", 0.5131),
        ("github", 0.0141),
        ("starcoder", 0.0504),
        ("stackexchange", 0.0164),
        ("wiki", 0.0489),

        ("AMPS", 0.0003),
        ("BookCorpus", 0.0006),
        ("fanfics", 0.0019),
        ("Openwebmath", 0.0075),
        ("PubMed", 0.0094),
    ]

    weights = [item[1] for item in data_config]
    _sum = round(sum(weights), 5)
    print('_sum: ' + str(_sum))

    ratio = 80 / 85
    for item in data_config:
        print('(\"{}\", {:.4f}),'.format(item[0], item[1] * ratio))
