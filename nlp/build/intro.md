# Advanced NLP for finance

Machine Learning for Finance, Dr. Arun B Ayyar. This notebook produces every number on the slides.

Four parts: attention inside a small transformer, financial sentiment, named entity recognition, and document
classification and information extraction.

The sentences, pages and loan agreements are simulated by `nlp/build/make_data.py`. The sentiment data is the public
`zeroshot/twitter-financial-news-sentiment` dataset (MIT licence), downloaded by the notebook.

The notebook takes a few minutes on a GPU (Colab: Runtime, Change runtime type, T4 GPU) and much longer on a CPU.
