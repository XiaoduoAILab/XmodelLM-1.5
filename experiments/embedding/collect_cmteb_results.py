import json
import os

import numpy as np


def get_result(cls, item):
    result = 0
    if cls == 'Retrieval':
        result = item['dev']['ndcg_at_10']

    elif cls == 'STS':
        if 'test' in item:
            if 'cos_sim' in item['test']:
                result = item['test']['cos_sim']['spearman']
            else:
                result = item['test']['zh']['cos_sim']['spearman']
        else:
            result = item['validation']['cos_sim']['spearman']

    elif cls == 'PairClassification':
        result = item['validation']['cos_sim']['ap']

    elif cls == 'Classification':
        if 'test' in item:
            if 'accuracy' in item['test']:
                result = item['test']['accuracy']
            elif 'zh' in item['test']:
                result = item['test']['zh']['accuracy']
            elif 'zh-CN' in item['test']:
                result = item['test']['zh-CN']['accuracy']
        else:
            result = item['validation']['accuracy']

    elif cls == 'Reranking':
        if 'test' in item:
            result = item['test']['map']
        else:
            result = item['dev']['map']

    elif cls == 'Clustering':
        result = item['test']['v_measure']

    return round(result, 5)


if __name__ == "__main__":
    benchmarks = dict()
    benchmarks['Retrieval'] = ['T2Retrieval', 'MMarcoRetrieval', 'DuRetrieval', 'CovidRetrieval', 'CmedqaRetrieval',
                               'EcomRetrieval', 'MedicalRetrieval', 'VideoRetrieval']
    benchmarks['STS'] = ['ATEC', 'BQ', 'LCQMC', 'PAWSX', 'STSB', 'AFQMC', 'QBQTC', 'STS22']
    benchmarks['PairClassification'] = ['Ocnli', 'Cmnli']
    benchmarks['Classification'] = ['TNews', 'IFlyTek', 'MultilingualSentiment', 'JDReview', 'OnlineShopping', 'Waimai', 'AmazonReviewsClassification', 'MassiveIntentClassification', 'MassiveScenarioClassification']
    benchmarks['Reranking'] = ['T2Reranking', 'MMarcoReranking', 'CMedQAv1', 'CMedQAv2']
    benchmarks['Clustering'] = ['CLSClusteringS2S', 'CLSClusteringP2P', 'ThuNewsClusteringS2S', 'ThuNewsClusteringP2P']

    base_dir = '/data1/liuyang/code/FlagEmbedding/C_MTEB/zh_results/'

    models = [m for m in os.listdir(base_dir)]

    for cls in ['Retrieval', 'STS', 'PairClassification', 'Classification', 'Reranking', 'Clustering']:
        print(f'\n{cls}')

        for m in models:
            model_path = os.path.join(base_dir, m)

            results = []
            for b in benchmarks[cls]:
                result_path = os.path.join(model_path, b) + '.json'
                with open(result_path, 'r') as fp:
                    item = json.load(fp)
                    result = get_result(cls, item)
                    print(f'{m}\t{b}\t{result}')
                    results.append(result)
            avg = np.mean(results)
            print(f'{m}\tAvg\t{round(avg, 5)}')
