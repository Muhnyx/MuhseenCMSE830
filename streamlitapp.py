import streamlit as st
import pandas as pd
from PIL import Image
import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForCTC
from audiorecorder import audiorecorder

st.title("ASR Finetuning Project")
st.markdown("## Introduction")
st.image(Image.open("files/ASR_model_paper_streamlit.png"), caption='Schematic of our method integrating data augmentation and fine-tuning of wav2vec 2.0 for stuttered speech, exemplified by augmentations from a FluencyBank speech sample Credit: Mujtaba et al', use_column_width=True)
st.markdown("Automatic Speech Recognition (ASR) systems often struggle with disfluencies related to stuttering, such as involuntary pauses and word repetitions, leading to inaccurate transcriptions. A significant challenge in improving ASR performance is the limited availability of large, annotated datasets that specifically capture disfluent speech patterns.")

st.markdown("In response to this issue, we present an inclusive design approach for ASR that combines large-scale self-supervised learning on standard speech with targeted fine-tuning and data augmentation on a curated dataset of disfluent speech. Our innovative data augmentation techniques enrich the training datasets by incorporating diverse disfluency examples, thereby enhancing the ASR system's ability to accurately process these speech variations.")

st.markdown("Our findings demonstrate that fine-tuning the wav2vec 2.0 model using a relatively small labeled dataset, complemented by data augmentation, can lead to significant reductions in word error rates for disfluent speech. This approach not only enhances the inclusivity of ASR systems for individuals who stutter but also sets the stage for ASR technologies that can accommodate a broader range of speech variations.")


st.markdown("## Dataset for ASR Performance Evaluation")

st.markdown("To evaluate the performance of ASR systems on authentic stuttered speech, we utilize the FluencyBank dataset, which includes video recordings of individuals who stutter in two contexts: reading passages and interviews. The dataset features 7 readings and 12 interview videos from 12 participants, each accompanied by transcripts that follow the Codes for the Human Analysis of Transcripts (CHAT) standard. These transcripts break down speech into individual utterances and include disfluency event annotations.")
st.markdown("### Raw footage")
st.video("files/interview-26f.mp4")
st.markdown("Due to issues with temporal misalignment and inaccuracies in the original CHAT transcripts, we re-annotated the dataset with the help of trained speech-language pathologists specializing in stuttering, ensuring high inter-annotator agreement. The final dataset comprises 1,373 utterances and a total audio duration of 2.21 hours.")
labels = pd.read_csv("label.csv")
file_names=set()
for label in labels["Audio File ID"].to_list():
    file_names.add(f"{label.split("-")[0]}-{label.split("-")[1]}")
if labels["Audio File ID"].str.contains("24fa").any()==True:
    labels_original=labels
    labels=labels[~labels["Audio File ID"].str.contains("24fa")] # remove 24fa elements
    file_names.discard('interview-24fa')
    file_names.discard('reading-24fa')


labels_sorted = labels.sort_values(by='Start Time')
st.dataframe(labels_sorted[labels_sorted["Audio File ID"].str.contains("interview-26f")])

st.markdown("## CPU Parallelised Parsing by Start and End time")
st.markdown("Run CPU parallelised parsing to get 16Khz Wav files using FFMPEG, results in 1300 samples")

st.markdown("interview-26f-16.wav")
st.audio("files/interview-26f-16.wav")
st.markdown("interview-26f-19.wav")
st.audio("files/interview-26f-19.wav")
st.markdown("interview-26f-20.wav")
st.audio("files/interview-26f-20.wav")


st.markdown("""
### Data Augmentation Strategy for Stuttered Speech

To enhance the diversity of small stuttered speech datasets, our data augmentation strategy introduces a variety of disfluency events that better mirror the natural speech patterns of individuals who stutter. While the LibriStutter dataset provides a preliminary attempt at simulating stuttering, it lacks the complexity and variability we aim to capture.

Our method generates diverse disfluency events, such as word repetitions (e.g., “my my my name is”), phrase repetitions (e.g., “my name my name my name is”), and interjections (e.g., “um um my name uh is”), which are inserted randomly during ASR training. This approach improves the realism of training data and enhances ASR accuracy for various stuttered speech patterns.

We augment FluencyBank utterances by incorporating varying numbers of augmented samples, starting from disfluency-free transcripts. Using the OpenAI TTS API, we synthesize speech with inserted disfluencies. The frequencies and placements of these events are randomized to include:
1. **Word repetitions:** 1 to 4 additional repeats.
2. **Phrase repetitions:** 1 to 3 repeats of 2 to 4-word phrases.
3. **Interjections:** 1 to 4 instances of “uh” or “um.”

We initially evaluate N augmented samples (N = 500, 1000, 2000, 3000) and later increase the variability, generating up to 6000 samples to assess their impact on ASR performance. 
""")

st.markdown("interview-50fb-57")
st.audio("files/disfluent-1.wav")
st.markdown("interview-35mb-83")
st.audio("files/disfluent-12.wav")
st.markdown("interview-54f-37")
st.audio("files/disfluent-21.wav")
st.dataframe(labels)

st.markdown("## Testing ASR working (Wav2vec2.0 untrained)")
st.markdown("Record yourself and watch Wav2vec instantly transcribe what you say")




audio = audiorecorder("Click to record", "Click to stop recording")
if len(audio) > 0:
    # To save audio to a file, use pydub export method:
    audio.export("audio.wav", format="wav")



    processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
    model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    speech,rate=speech, rate = torchaudio.load("audio.wav")

    # Resample the audio if necessary (Wav2Vec 2.0 expects 16 kHz audio)
    if rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=rate, new_freq=16000)
        speech = resampler(speech)

    # Convert stereo (2 channels) to mono (1 channel) if needed
    speech = speech.mean(dim=0)

    # Preprocess the audio file (convert it to input values for the model)
    input_values = processor(speech.numpy(), rate=16000, return_tensors="pt").input_values

    # Perform inference to get logits
    with torch.no_grad():
        logits = model(input_values).logits

    # Get the predicted IDs from the logits (most likely tokens)
    predicted_ids = torch.argmax(logits, dim=-1)

    # Decode the predicted IDs to get the transcription
    transcription = processor.batch_decode(predicted_ids)[0]

    # Print the transcription
    st.markdown(f"Transcription: {transcription}")



st.markdown("## Conclusion and Future Work")

st.markdown("In this midterm project, we explored the basics of the Wav2Vec 2.0 model on the FluencyBank dataset. While we have successfully set up a framework for transcription and preliminary assessments, we intend to run the actual fine-tuning of the model later. Our future work will involve enhancing the model's accuracy through targeted training on disfluent speech samples, with the aim of reducing word error rates (WER) and improving FBERT metrics.")

st.markdown("We look forward to sharing the performance outcomes of our fine-tuning efforts, including detailed metrics and insights, in the final presentation ")


