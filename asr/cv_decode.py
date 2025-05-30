import os

import pandas as pd
import requests
from tqdm import tqdm

DATA_ROOT = os.environ.get("DATA_ROOT", "./data")
TARGET = os.environ.get("TARGET", "cv-valid-dev")

DATA_CSV = os.path.join(DATA_ROOT, f"{TARGET}.csv")
TARGET_DIR = os.path.join(DATA_ROOT, TARGET)
data_index = pd.read_csv(DATA_CSV)
print(data_index.head())

def inferece_asr(audio_path, protocal, host, port):
    """
    Send audio file to ASR API for transcription
    
    Args:
        audio_path (str): Path to the audio file
        host (str): Host address of the ASR API
        port (int): Port number of the ASR API
    
    Returns:
        str: Transcribed text
    """
    url = f"{protocal}://{host}:{port}/asr"
    
    try:
        with open(audio_path, 'rb') as audio_file:
            files = {'file': audio_file}
            response = requests.post(url, files=files)
            
        if response.status_code == 200:
            result = response.json()
            return result['transcription']
        else:
            raise Exception(f"ASR API error: {response.status_code} - {response.text}")
            
    except Exception as e:
        raise Exception(f"Error during ASR request: {str(e)}")
    
results = []
for i in tqdm(range(len(data_index))):
    row = data_index.iloc[i]
    audio_path = os.path.join(TARGET_DIR, row['filename'])
    transcription = inferece_asr(audio_path, "http", "localhost", 8001)
    # print(transcription)
    results.append(transcription)
    # break

data_index['generated_text'] = results
data_index.to_csv(f"./{TARGET}_asr.csv", index=False)
