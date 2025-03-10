from kokoro import KPipeline
import numpy as np


pipeline = KPipeline(lang_code='a')
voice = 'af_heart'

def generate_audio(text):
    generator = pipeline(
        text, 
        voice=voice,
        speed=1, 
        split_pattern=r'\n+'
    )
    
    all_audio = []

    for i, (gs, ps, audio) in enumerate(generator):
        all_audio.append(audio) 

    # Concatenate all audio fragments into one
    if all_audio:
        combined_audio = np.concatenate(all_audio)  
        return combined_audio

    return None 