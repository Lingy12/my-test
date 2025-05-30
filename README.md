# my-test

requirements: Running at 1 A100 80GB

## Data set-up

```bash
mkdir data

cd data

wget https://www.dropbox.com/scl/fi/i9yvfqpf7p8uye5o8k1sj/common_voice.zip?rlkey=lz3dtjuhekc3xw4jnoeoqy5yu&dl=0 

mv common_voice.zip\?rlkey=lz3dtjuhekc3xw4jnoeoqy5yu\&dl=0 common_voice.zip    

unzip common_voice.zip
```

## Task 1

### 1. Starting asr docker image

```bash

docker compose -f docker-compose-asr.yaml up --build -d 

```

### 2. Inference on the API

```bash

python asr/cv_decode.py # The result is saved in the data/cv-valid-dev, but since data directory is ignored in git, it's in example_result/cv-valid-dev_asr.csv

```

## Task 2

### 1. Finetuning the Model

```bash
cd asr-train

python cv-train-2a.py
```

### 2. Plot the result, and run inference comparasion
