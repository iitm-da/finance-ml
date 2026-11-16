# LLMs in depth

Machine Learning for Finance, Dr. Arun B Ayyar. This notebook produces every number on the slides.

Four parts: retrieval-augmented generation over an internal knowledge base, evaluating answers and catching
hallucinations, prompt engineering, and prompting against LoRA against full fine-tuning.

The knowledge base belongs to a fictional bank and is written by `llm/build/make_data.py`; every policy and number in it
is invented. The language model is `Qwen2.5-1.5B-Instruct`, an open model that runs on a free Colab GPU
(Runtime, Change runtime type, T4 GPU). No API key is needed.
