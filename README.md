<h1 align="center">
Xmodel_LM1.5-1B
</h1>

<h5 align="center">

[![hf_space](https://img.shields.io/badge/🤗-Xiaoduo%20HuggingFace-blue.svg)](https://huggingface.co/XiaoduoAILab/XmodelLM1.5)
[![arXiv](https://img.shields.io/badge/Arxiv-2406.02856-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2406.02856) 
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/XiaoduoAILab/XmodelLM-1.5.git)[![github](https://img.shields.io/github/stars/XiaoduoAILab/XmodelLM.svg?style=social)](https://github.com/XiaoduoAILab/XmodelLM-1.5.git)  


</h5>

## 🌟 Introduction

We introduce Xmodel-LM1.5, a novel 1-billion-parameter multilingual large model pretrained on approximately 2 trillion tokens. The model demonstrates strong performance across several languages, with particularly notable results in Thai, Arabic, and French, alongside its effectiveness in Chinese and English. In addition, we contribute to the research community by releasing a Thai evaluation dataset, which includes hundreds of questions annotated by students from Chulalongkorn University’s School of Integrated Innovation. While the results are promising, we acknowledge that there is still room for improvement. We hope this work advances ongoing efforts in multilingual AI research and promotes better cross-linguistic understanding in various natural language processing tasks
## 📊 Benchmark

### Commonsense Reasoning

| Model                | ARC-c     | ARC-e     | Boolq     | HellaSwag | OpenbookQA | PiQA     | SciQ     | Winogrande| Avg    |
|----------------------|-----------|-----------|-----------|-----------|------------|----------|----------|-----------|---------|
| OPT-1.3B             | 23.29     | 57.03     | 57.80     | 41.52     | 23.20      | 71.71    | 84.30    | 59.59     | 52.32   |
| Pythia-1.4B          | 25.60     | 57.58     | 60.34     | 39.81     | 20.20      | 71.06    | 85.20    | 56.20     | 53.38   |
| TinyLLaMA-3T-1.1B    | 27.82     | 60.31     | 57.83     | 44.98     | 21.80      | 73.34    | 88.90    | 59.12     | 54.26   |
| MobileLLaMA-1.4B     | 26.28     | 61.32     | 57.92     | 42.87     | 23.60      | 71.33    | 87.40    | 58.25     | 53.60   |
| InternLM2-1.8B       | 37.54     | 70.20     | 69.48     | 46.52     | 24.40      | 75.57    | 93.90    | 65.67     | 60.41   |
| Qwen2.5-1.5B         | 40.36     | 74.83     | 73.27     | 50.09     | 31.40      | 75.95    | 94.90    | 63.06     | 62.98   |
| **Xmodel-LM1.5-1B**  | 28.92     | 64.31     | 62.78     | 45.94     | 22.20      | 72.20    | 89.10    | 60.62     | 55.76   |




### Performance on multilingual tasks (Thai, Arabic, French, Chinese)

#### Performance on Thai language tasks
| Model        | **belebele_tha_Thai** | **xcopa_th** |
|--------------|-----------------------|--------------|
| PolyLM-1.7B  | 0.2267                | 0.56         |
| PolyLM-13B   | 0.2367                | 0.586        |
| **Xmodel-LM1.5-1B** | 0.2756         | 0.59         |

#### Performance on Arabic language tasks
| Model        | **arc_ar** | **hellaswag_ar** | **m_mmlu_ar** | **piqa_ar** |
|--------------|------------|------------------|---------------|-------------|
| PolyLM-1.7B  | 0.2173     | 0.2818           | 0.2288        | 0.5381      |
| PolyLM-13B   | 0.2284     | 0.3296           | 0.2434        | 0.5653      |
| **Xmodel-LM1.5-1B** | 0.2344     | 0.3279           | 0.2454        | 0.5789      |

#### Performance on French language tasks
| Model        | **hellaswag_fr** | **m_mmlu_fr** | **paws_fr** | **piqa_fr** |
|--------------|------------------|---------------|-------------|-------------|
| PolyLM-1.7B  | 0.3085           | 0.2458        | 0.548       | 0.5381      |
| PolyLM-13B   | 0.4064           | 0.2602        | 0.539       | 0.5653      |
| **Xmodel-LM1.5-1B** | 0.37        | 0.2525        | 0.5325      | 0.5789      |

#### Performance on Chinese language tasks
| Model        | **arc_zh** | **xcopa_zh** |
|--------------|------------|--------------|
| PolyLM-1.7B  | 0.1957     | 0.5381       |
| PolyLM-13B   | 0.2803     | 0.5653       |
| **Xmodel-LM1.5-1B** | 0.259     | 0.5789       |


## 🛠️ Install

1. Clone this repository and navigate to XmodelLM folder
   ```bash
   git clone https://github.com/XiaoduoAILab/XmodelLM-1.5.git
   cd xmodellm
   ```

2. Install Package
    ```Shell
    pip install -r requirements.txt
    ```

## 🗝️ Quick Start

#### Download Xmodel_LM model

Our model files are fully open source on huggingface, you can download them at [here](https://huggingface.co/XiaoduoAILab/XmodelLM-1.5).

#### Example for Xmodel_LM model inference
Download the model files first and save them in your folder. Then you can run the scripts below, we recommend entering an absolute path as the parameter.
```bash
python generate.py --model_path path/to/folder --device cuda:0
```

## ✏️ Reference

If you find Xmodel_LM useful in your research or applications, please consider giving a star ⭐ and citing using the following BibTeX:

```
@misc{wang2024xmodellm1.5,
    title={Xmodel-LM1.5: An 1B-scale Multilingual LLM},
    author={Qun Wang and Yang Liu and QingQuan Lin  and Ling Jiang},
    year={2024},
    eprint={2411.10083},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

