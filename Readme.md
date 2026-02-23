# Word2Vec's Continuous Bag of Words (CBOW) implementation in numpy

This implementation tries to mimic the original word2vec C implementation but there
are some minor changes like uniform negative sampling, Averaging the projections rather than
keeping only the sum of the vectors.

To get the dataset used, run the below snippet in terminal (linux/mac os)
```bash
wget "https://data.statmt.org/news-crawl/en/news.2025.en.shuffled.deduped.gz"
gzip -d < news.2025.en.shuffled.deduped.gz > news.2025.en.shuffled.deduped
```
Copy and paste `news.2025.en.shuffled.deduped` in the `/datasets` folder or
you can update the path in the [experiments.ipynb](/notebooks/experiments.ipynb) notebook

- Experiments and testing can be found in [experiments.ipynb](/notebooks/experiments.ipynb)
- Source code in .py can be found in [main.py](/src/main.py]

## Requirements
- Only `numpy` for **main.py**
- `numpy` & `matplotlib` for **experiments.ipynb**

To setup the environment - `pip install -r requirements.txt`

## Implementation

### Continuous Bag of Words with Negative Sampling

Architecture - 

- 2 Layers:
  - Projection Layer - Takes the N input vectors and projects (averages) to a single hidden size (d) vector
  - Output Layer -
     - Training: $$\sigma(x_{proj} \cdot W_{out}[target])$$ - `Sigmoid Activation` -> output = 0 to 1
     - Inference: $$argmax(x_{proj} \cdot W_{out})$$ - `Argmax Activation` -> output = index of predicted token
    
