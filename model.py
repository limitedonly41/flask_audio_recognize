import torch
import librosa

from evaluate import load

from transformers import WhisperForConditionalGeneration, WhisperProcessor
from datasets import load_dataset

GPU = False




print('libs loaded')
model = WhisperForConditionalGeneration.from_pretrained("./whisper-small-ukrainian/")

# model = WhisperForConditionalGeneration.from_pretrained("/home/ubuntu/whisper-ukrainian/whisper-small-uk/checkpoint-test")
if GPU:
  model.to('cuda')
processor = WhisperProcessor.from_pretrained("./whisper-small-ukrainian/")

# processor = WhisperProcessor.from_pretrained("/home/ubuntu/whisper-ukrainian/whisper-small-uk/checkpoint-test")


def return_arr(path):

    # audio, sr = librosa.load('./1.m4a')
    audio, sr = librosa.load(path)
    audios = []

    buffer = 10 * sr

    samples_total = len(audio)
    samples_wrote = 0
    counter = 1

    while samples_wrote < samples_total:

        # print(samples_wrote)

        #check if the buffer is not exceeding total samples 
        if buffer > (samples_total - samples_wrote):
            buffer = samples_total - samples_wrote

        if counter > 1:
                block = audio[samples_wrote - 2*sr : (samples_wrote + buffer)]
        else:
                block = audio[samples_wrote : (samples_wrote + buffer)]

        audios.append(block)

        # out_filename = "split_" + str(counter) + "_" + file_name

        # Write 2 second segment
        # librosa.output.write_wav(out_filename, block, sr)
        counter += 1
        samples_wrote += buffer

    return audios


def return_text(audios):
    input_features = processor(audios, return_tensors="pt", sampling_rate=16_000).input_features

    if GPU:
            input_features = input_features.to('cuda')

    generated_ids = model.generate(inputs=input_features)

    transcription = processor.batch_decode(generated_ids, normalize=True, skip_special_tokens=True)

    ans = [processor.tokenizer._normalize(it) for it in transcription]

    return ans


def listRightIndex(alist, value):
    return len(alist) - alist[-1::-1].index(value) -1


def text_from_arr(arr):

    # print(len(arr))
    new_arr = []
    for i in range(0, len(arr), 1):


        if i%2 == 0 or i==len(arr):

            if i != len(arr)-1:


              arr1 = arr[i].split()
              arr2 = arr[i+1].split()

              # print(arr1)

              temp_arr = []
              r_ind = listRightIndex(arr1, arr2[2])
              temp_arr.extend(arr1[:r_ind-1])
              temp_arr.extend(arr2[1:])

              new_arr.extend(temp_arr)

            else:

              arr1 = arr[i-1].split()
              arr2 = arr[i].split()

              # print(arr1)

              temp_arr = []
              r_ind = listRightIndex(arr1, arr2[2])
              temp_arr.extend(arr1[:r_ind-1])
              temp_arr.extend(arr2[1:])

              new_arr.extend(temp_arr)
    return new_arr
  



def predict_result(audio):

    audios = return_arr(audio)
    ans = return_text(audios)
    text = text_from_arr(ans)

    edited_text = [t.replace(' ', "'") for t in text]

    return " ".join(edited_text)