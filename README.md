# my-test

requirements: Running at 1 A100 80GB

## Envrionment set-up

```bash
git lfs clone git@github.com:Lingy12/my-test.git # Use lfs in order to fetch the existing training results, if you want to rerun the whole pipeline, normal git clone will do.

bash setup.sh
```

## Data set-up

```bash
mkdir data

cd data

wget https://www.dropbox.com/scl/fi/i9yvfqpf7p8uye5o8k1sj/common_voice.zip?rlkey=lz3dtjuhekc3xw4jnoeoqy5yu&dl=0 

mv common_voice.zip\?rlkey=lz3dtjuhekc3xw4jnoeoqy5yu\&dl=0 common_voice.zip    

unzip common_voice.zip
```

## Output Model

The whole pipeline will have a model saved at models/wav2vec-large-960h-cv. If you want to skip the training, you could run the following:

```bash
mkdir models

cd models

git lfs clone https://huggingface.co/lingy/wav2vec2-large-960h-cv
```

## Task 2

### 1. Starting asr docker image

```bash

docker compose -f docker-compose-asr.yaml up --build -d 

```

### 2. Inference on the API

```bash

python asr/cv_decode.py # The result is saved in the asr/cv-valid-dev_asr.txt

```

## Task 3

Assumption: The wav2vec2 tokenizer only consume upper case text without any punctuation. The due to the GPU memory limitation, I limit the time of the audio to 20 seconds. 

Note: I wrote a python script for long run training. But the code I kept a copy in the notebook, the results and other part are in notebook.

### 1. Finetuning the Model

```bash
cd asr-train

python cv-train-2a.py # Note the pipeline is also in cv-train-2a.ipynb, but I put it as seperate python script because I could keep it in tmux session. 

# This will create the model and results.json containing intermediate train and validation logging result.

mv ../models/checkpoints/best_model ../models/wav2vec2-large-960h-cv
```

### 2. Plot the result, and run inference comparasion. 

```bash

# Use part after Training in cv-train-2a.ipynb. The interpretation of intermediate result also in the notebook.

```

## Task 5

1. cv-hotword-5a.ipynb: Run the finetuned model inference, and save a inference_results.csv. Also, output detected.txt which contains the transcription that contains hot words.
2. cv-hostword-similarity-5b.ipynb: Use the inference_results.csv produced in previous step, then use the embedding to check whether similar phases is in transcription. Then output cv-valid-dev-updated.csv.