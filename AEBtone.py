import simpleaudio as sa

def play_alert():
    """Plays the preprocessed AEB alert sound."""
    wav_file = "modified_alert.wav"
    wave_obj = sa.WaveObject.from_wave_file(wav_file)
    play_obj = wave_obj.play()
    #play_obj.wait_done()  # Wait for the audio to finish playing

if __name__ == "__main__":
    print("Playing AEB alert...")
    play_alert()
