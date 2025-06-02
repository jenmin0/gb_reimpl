# Ghostbuster-Reimplementation

This repository contains a reimplementation of the **Ghostbuster** system for detecting AI-generated text, as originally described in:

**Ghostbuster: Detecting Text Ghostwritten by Large Language Models**  
Vivek Verma, Eve Fleisig, Nicholas Tomlin, Dan Klein  
[NAACL 2024 Paper](https://arxiv.org/abs/2305.15047) | [Original Codebase](https://github.com/vivek3141/ghostbuster)

Licensed under [Creative Commons Attribution 3.0 Unported](https://creativecommons.org/licenses/by/3.0/)

---

## About This Project

This project is an **unofficial reimplementation** of Ghostbuster, adapted to:

- Replacing closed models with open-source LLMs
- Regenerating token log probability from source data using open models
- Updating symbolic data and results

# Installation

The installation process is the same as in the original project:

Each of our files are pickled with `python3.10`, so we highly reccomend creating a new conda environment as follows:

```
conda create -n ghostbuster python=3.10
conda activate ghostbuster
```

Then clone this reponsitory or download zip file, and 

```
cd ghostbuster
```

Lastly, install the dependencies and the package:

```
pip install -r requirements.txt
pip install -e .
```

You may also need to open a `python` shell to install the following nltk `brown` model:
```python
import nltk
nltk.download('brown')
```

# Usage

The symbolic data and text data are already prepared. After extraction or clone, please place them in the root directory.

## data 
either clone <a href="https://github.com/jenmin0/gb-data-llama.git">[data]</a> or extract `llamas_data.zip` and rename the folder to `data`.

## symbolic data

extract `symbolic_data.zip`.

Then you can run with selection parameters:

```
python ghostbuster/train.py
python ghostbuster/run.py
```



