---

---
license: mit
language:
- en
- id
- jv
- km
- lo
- ms
- my
- su
- ta
- th
- tl
- vi
- zh
---
![Banner!](ModernBERT-300M.png "banner-ModernBERT-300M")
# Model Card for SEA-LION-ModernBERT-300M

*Last update: 2026-03-16*

SEA-LION is a collection of Large Language Models (LLMs) which have been pretrained and fine-tuned for the Southeast Asia (SEA) region.

This encoder-only model leverages the advanced **ModernBERT** architecture combined with the Gemma 3 SentencePiece tokenizer. The adoption of the **Gemma 3 tokenizer** with ModernBERT allows the model to achieve highly efficient and culturally nuanced text processing. This combination significantly improves the tokenization fertility and compression rates for complex regional scripts and diverse Southeast Asian languages, enabling the model to handle longer context windows and cross-lingual tasks with greater computational efficiency.

To achieve this level of performance, the model was developed through a rigorous, multi-stage training pipeline. The foundation was established through extensive **pre-training on 2 Trillion (2T) tokens**, followed by a mid-training phase on an additional 1 Trillion (1T) tokens. Both of these massive training phases comprehensively covered code alongside 13 specific languages: Burmese, Chinese, English, Filipino, Indonesian, Javanese, Khmer, Lao, Malay, Sundanese, Tamil, Thai, and Vietnamese.

## Model Details

### Model Description

The SEA-LION-ModernBERT-300M models are built on the [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) architecture and has a vocabulary size of 262K.

For tokenization, the model employs our custom [Gemma3](https://storage.googleapis.com/deepmind-media/gemma/Gemma3Report.pdf) tokenizer, which has excellent performance for SEA languages, ensuring optimal model performance.

- **Developed by:** AI Products Pillar, AI Singapore
- **Funded by:** Singapore NRF
- **Shared by:** AI Products Pillar, AI Singapore
- **Model type:** Encoder
- **Context length:** 8k
- **Languages:** Burmese, Chinese, English, Filipino, Indonesian, Javanese, Khmer, Lao, Malay, Sundanese, Tamil, Thai, and Vietnamese
- **License:** [MIT](https://tlo.mit.edu/understand-ip/exploring-mit-open-source-license-comprehensive-guide)

### Model Sources

- **Repository:** The weights for this model and its various training stages are being released to support transparency, research, and diverse downstream applications. [**Link to HF Repo**](https://huggingface.co/aisingapore/EA-LION-ModernBERT-300M)

## Uses

This model card details one of the variants available within this collection.

| Model Variant | Model Repository | Suggesting Applications & Use Cases |
| --- | --- | --- |
| **Fine-tuned Embedding Models** | - [aisingapore/SEA-LION-E5-Embedding-600M](https://huggingface.co/aisingapore/SEA-LION-E5-Embedding-600M) <br> - [aisingapore/SEA-LION-ModernBERT-Embedding-300M](https://huggingface.co/aisingapore/SEA-LION-ModernBERT-Embedding-300M) <br> - [aisingapore/SEA-LION-ModernBERT-Embedding-600M](https://huggingface.co/aisingapore/SEA-LION-ModernBERT-Embedding-600M) | - Retrieval-Augmented Generation (RAG) <br> - Information retrieval, and search <br> - Similarity comparisons |
| **Pre-trained Encoder Models** | - [aisingapore/SEA-LION-ModernBERT-300M](https://huggingface.co/aisingapore/SEA-LION-ModernBERT-300M) <br> - [aisingapore/SEA-LION-ModernBERT-600M](https://huggingface.co/aisingapore/SEA-LION-ModernBERT-600M) <br> | - Fill mask <br>- Text classification <br> - Fine-tuning for downstream tasks (e.g., sentiment analysis, classification). |
| **Pre-trained Model Checkpoints** | - [aisingapore/SEA-LION-ModernBERT-300M-checkpoints](https://huggingface.co/aisingapore/SEA-LION-ModernBERT-300M-checkpoints) <br> - [aisingapore/SEA-LION-ModernBERT-600M-checkpoints](https://huggingface.co/aisingapore/SEA-LION-ModernBERT-600M-checkpoints) <br> | - Continued Pre-Training (CPT) <br> - Fine-tuning for downstream tasks (e.g., sentiment analysis, classification). |

*Note: If you are deploying our models for your specific use case, we would love to hear from you! Please feel free to [contact us](mailto:sealion@aisingapore.org) to share your experience or explore potential collaborations.*

### Bias, Risks, and Limitations

The model was not tested for robustness against adversarial usage. It is important for users to be aware that our model exhibits certain limitations that warrant consideration. Users should also exercise caution in continue-implementing and validating the model's responses due to the potential inconsistencies.

### Recommendations

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model.

## How to Get Started with the Model

Use the code below to download the model locally.

```python
pip install -U transformers>=4.48.0
```

```python
import torch
from transformers import pipeline

pipeline = pipeline(
    task="fill-mask",
    model="aisingapore/SEA-LION-ModernBERT-300M",
    dtype=torch.float16,
    device=0
)
pipeline("Plants create  through a process known as photosynthesis.")
```

---

## Training Details

The models are pre-trained from scratch through a two-phase pipeline, beginning with an extensive initial stage on 2 trillion tokens, followed by a mid-training phase on an additional 1 trillion tokens. Both phases incorporated a diverse dataset covering programming code and 13 languages: Burmese, Chinese, English, Filipino, Indonesian, Javanese, Khmer, Lao, Malay, Sundanese, Tamil, Thai, and Vietnamese.

### Training Data

**SEA-LION-ModernBERT-300M-checkpoints** was pre-trained from scratch on a number of trillion tokens corpus with the following linguistic and thematic distribution:

| Data Source | Percentage |
| --- | --- |
| code | 10% |
| EN - English | 35% |
| ID - Indonesian | 8% |
| JV - Javanese | 0.5% |
| KM - Khmer | 1.5% |
| LO - Lao | 0.5% |
| MS - Malay | 4.75% |
| MY - Burmese | 1.75% |
| SU - Sundanese | 0.5% |
| TA - Tamil | 4.5% |
| TH - Thai | 8% |
| TL - Filipino | 2.5% |
| VI - Vietnamese | 8.5% |
| ZH - Chinese | 14% |

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model is evaluated across three primary benchmark suites to provide a comprehensive assessment of embedding quality across Southeast Asian, Chinese, and English contexts:

- **[SEA-BED (Southeast Asia Embedding Benchmark)]**(<https://arxiv.org/pdf/2508.12243>): The primary testing suite, consisting of 169 datasets across 10 Southeast Asian languages (Burmese, Filipino, Indonesian, Khmer, Malay, Lao, Tamil, Tetum, Thai, and Vietnamese). Notably, 71% of these datasets are native-authored or human-curated to preserve regional linguistic properties.
- **CMTEB (Chinese Massive Text Embedding Benchmark)**: A specialised subset of MTEB focused on Chinese language tasks, used to evaluate performance in one of the region's most prominent scripts.
- **MTEB (Massive Text Embedding Benchmark)**: The industry-standard global benchmark used to gauge general-purpose English embedding performance across a wide array of tasks.

### Factors

Evaluation factors are categorised by task type and linguistic diversity to ensure the model's "fertility" and "nuance" are captured accurately:

- **Linguistic Coverage**: Evaluation spans across 10+ languages, including complex Brahmic scripts (Burmese, Khmer, Lao, Tamil, Thai) and Latin-based SEA scripts (Indonesian, Filipino, Malay, Tetum, Vietnamese).
- **Task Modality**:

	- **Retrieval/Reranking:** Efficiency in finding relevant documents within a large corpus.
	- **Semantic Textual Similarity (STS):** Precision in sentence-level semantic alignment.
	- **Clustering & Classification:** Ability to group or categorize text based on latent semantic meaning.
	- **Summarisation & Bitext Mining:** High-level semantic matching and cross-lingual alignment.
- **Architecture Efficiency:** Performance is measured in the context of the ModernBERT architecture and Gemma 3 tokenizer to assess computational efficiency versus embedding quality.

#### Metrics

To provide a standardized view of performance, we report the following metrics across the benchmark suites:

- **Classification:** F1-score.
- **Multi-label Classification:** F1-score
- **Pair Classification:** Average Precision (AP).
- **Semantic Textual Similarity (STS):** Cosine similarity scores.
- **Clustering:** V-Measure Score.
- **Bitext Mining:** F1-score.
- **Retrieval & Reranking:** NDCG@10 (Primary) and MAP.
- **Instruction Retrieval:** NNDCG@5

## Results

![SEAEmbeddingResults!](SEA-Embedding-Results.png "Performance comparison of embedding models on SEA-BED")
Performance comparison of embedding models on SEA-BED (<https://leaderboard.sea-lion.ai/embedding/SEA>). Captured on 13/03/2026 02:50pm.

## Environmental Impact

Carbon emission was estimated using the fact sheet from TRG [Datacenters](https://www.trgdatacenters.com/resource/h200-power-consumption/).

- **Hardware Type:** Nvidia H200 140GB GPUs
- **Hours used:** 1,825 GPU hours
- **Cloud Provider:** SMC H200
- **Compute Region:** Singapore
- **Carbon Emitted:** appx. 513.27 kg CO2 e

## Technical Specifications

### Model Architecture and Objective

SEA-LION-ModernBERT-300M is an encoder model using the ModernBERT architecture.

| Parameter | SEA-LION-ModernBERT |
| --- | --- |
| Layers | 22  |
| d_model | 768  |
| head_dim | 12 |
| Vocabulary | 262144 |
| Sequence Length | 8k |

### Compute Infrastructure

#### Hardware

- **Hardware Type:** Nvidia H200 140GB GPUs
- **Cloud Provider:** SMC H200

#### Software

SEA-LION was trained using the [ModernBERT code base](https://github.com/AnswerDotAI/ModernBERT) which is powered by the [Composer](https://github.com/mosaicml/composer) training framework from MosaicML.

## Glossary

- **SEA-BED:** Southeast Asia Embedding Benchmark – a comprehensive evaluation suite for embedding models on SEA languages.
- **Asymmetric Retrieval:** Retrieval tasks where query and document formulations differ.
- **Mean Pooling:** Aggregating token embeddings by averaging (weighted by attention mask) to produce a fixed-size sentence representation.

## More Information

This is the repository for the commercial fine-tuned model. The model has *not* been aligned for safety. Developers and users should perform their own safety fine-tuning and related security measures. In no event shall the authors be held liable for any claims, damages, or other liabilities arising from the use of the released weights and codes.

For more info, please contact us at [sealion@aisingapore.org](mailto:sealion@aisingapore.org)

## Team

Ahmed Dabeer, Ahn Jeongmi, Antonyrex Sajeban, Chan Hok Teng Adwin, Cheng Zi Yi Nicholas, Choa Hsueh Mei Esther, Heng Jonathan, Huang Yuli, Jann Railey Estrada Montalan, Lee Chwan Ren, Leong Wai Yi, Leong Wei Qi, Liew Rachel, **Limkonchotiwat Peerat**, Muhammad Ridzuan Bin Mokhtar, Nagarajan Karthik, **Ng Boon Cheong Raymond**, Ngee Chia Tai, Ngui Jian Gang, Nguyen Thanh Ngan, Ong Tat-Wee David, Ong Zhi Hao, Pereira Mark, Poon Joseph, Rengarajan Hamsawardhini, Siow Wei Kang Bryan, Susanto Yosephine, Sutaveephamochanon Anocha, Tan Choon Meng, Tan Chor Phin Evelyn, Tan Siao Wei Jessica, Tan Yixian, Tee Jun Yun, Teng Kok Wai Walter, Teo Eng Sipp Leslie, Tjhi William, Wu Donghang, Yeo Yeow Tong, Yong Xianbin, Zhang Haoyang, Zhang Zhou

## **Acknowledgement**

This project is supported by the National Research Foundation Singapore and Infocomm Media Development Authority (IMDA), Singapore under its National Large Language Model Funding Initiative.

## Contact

[sealion@aisingapore.org](mailto:sealion@aisingapore.org)
