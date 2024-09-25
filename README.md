
# Improving Text Emotion Detection through Comprehensive Dataset Quality Analysis

This repository contains the source code for the experiments conducted in our paper **"Improving Text Emotion Detection through Comprehensive Dataset Quality Analysis"**. The code implements a comprehensive framework for assessing the impact of dataset quality on text emotion detection (TED) performance.

Key components:

- Data Processing: The code processes and prepares datasets for text emotion detection tasks, implementing various quality metrics.
- Quality Metrics: We implement 14 quantitative metrics across four dimensions: representativity, readability, structure, and part-of-speech tag distribution. Key metrics include:

    - Automated Readability Index (ARI)
    - Coleman-Liau Index (CLI)
    - Gini Index
    - Word Count Index (WCI)


## Model Training and Evaluation
We use two types of models:

- BERT (Bidirectional Encoder Representations from Transformers)
- BiLSTM (Bidirectional Long Short-Term Memory)


## Experimentation

The code creates multiple datasets with varying quality characteristics and trains models on each, allowing for comprehensive analysis of quality impact.

```python

def calculate_ari(text):
    characters = len(text)
    words = len(text.split())
    sentences = len(text.split('.'))
    if sentences == 0:
        sentences = 1
    sentence_length = characters / sentences
    word_length = characters / words
    ari = 4.76 * (sentence_length / 100) + 0.59 * (word_length / 100) - 21.43
    return ari

def create_and_save_datasets(directory, iterations, df, sample_size):
    for subfolder in ['high', 'mid', 'low']:
        os.makedirs(os.path.join(directory, subfolder), exist_ok=True)

    for i in range(1, iterations + 1):
        high_df, mid_df, low_df = redistribute_ari(df.copy(), sample_size)
        high_df.to_csv(os.path.join(directory, 'high', f'high_ari_{i}.csv'), index=False)
        mid_df.to_csv(os.path.join(directory, 'mid', f'mid_ari_{i}.csv'), index=False)
        low_df.to_csv(os.path.join(directory, 'low', f'low_ari_{i}.csv'), index=False)

```

This code calculates the ARI for each text, creates datasets with high, medium, and low ARI scores, and saves them for further processing and model training.

The repository structure includes separate Python files for each quality metric (ARI.py, CLI.py, GINI.py, WCI.py), model training scripts (BERT_REGRESSION.py, BiLSTM_REGRESSION.py), utility functions (utils.py), and a main script (main.py) to orchestrate the entire experiment pipeline.

This codebase enables researchers to replicate our experiments, extend the framework with new quality metrics, or apply it to different text classification tasks beyond emotion detection.


## Authors

- [Mahdi Zareei](https://orcid.org/0000-0001-6623-1758)
- [Alejandro De León Languré](https://orcid.org/0000-0002-8362-2045)


## Feedback

If you have any feedback, please reach out to us at alejandro@deleonlangure.com


## License

[MIT](https://choosealicense.com/licenses/mit/)


## Tech Stack

**Language:** Python

**ML framework:** Pytoch

