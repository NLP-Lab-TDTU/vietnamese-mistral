## Step 0. Prepare

### Install miniconda

These four commands quickly and quietly install the latest 64-bit version of the installer and then clean up after themselves.
After installing, initialize your newly-installed Miniconda. The following commands initialize for bash and zsh shells:

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

conda init bash
conda init zsh
```

### Setup environment

Create new evironment with Python 3.10. Activate the environment every time you work related to LLM training.

```bash
conda create -n training python=3.10

conda activate training
```

### Install packages/libraries

```bash
pip install -r requirements.txt
```

### Setting HF token
```bash
huggingface-cli login

    _|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|
    _|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|
    _|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|
    _|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|
    _|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|

    A token is already saved on your machine. Run `huggingface-cli whoami` to get more information or `huggingface-cli logout` if you want to log out.
    Setting a new token will erase the existing one.
    To login, `huggingface_hub` requires a token generated from https://huggingface.co/settings/tokens .
Token:
Add token as git credential? (Y/n) Y
Token is valid (permission: read).
Cannot authenticate through git-credential as no helper is defined on your machine.
You might have to re-authenticate when pushing to the Hugging Face Hub.
Run the following command in your terminal in case you want to set the 'store' credential helper as default.

git config --global credential.helper store

Read https://git-scm.com/book/en/v2/Git-Tools-Credential-Storage for more details.
Token has not been saved to git credential helper.
Your token has been saved to /home/hieunguyen/.cache/huggingface/token
Login successful
```

### Download tokenizer and model

Download tokenizer and model from Huggingface, internet required. All files will be saved at ./init_model .

```bash
bash scripts/download_tokenizer.sh
bash scripts/download_model.sh
```


## Step 1. Download dataset

Download datasets from Huggingface, internet required. All datasets will be saved at ./data .

```bash
bash scripts/download_datasets.sh
```

## Step 2. Combine and tokenize dataset

Before combining datasets and tokenize datasets, copy folders ./init_model and ./data to training cluster.
The results will be saved at ./processed_data .

```bash
python scripts/combine_and_tokenize_dataset.py
```

## Step 3. Training

Run script below to training.
```bash
bash scripts/run_mistral_7b_8k.sh
```
