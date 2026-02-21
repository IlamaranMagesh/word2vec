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
you can update the path in the `environments.ipynb` notebook
