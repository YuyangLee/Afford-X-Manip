"""
pip install sounddevice openai-whisper

"""

import sounddevice as sd
import whisper


def record_audio(duration=5, sample_rate=16000):
    """
    Record audio from the microphone.
    Args:
    duration (int): Duration of the recording in seconds.
    sample_rate (int): Sampling rate of the audio in Hz.

    Returns:
    np.ndarray: Recorded audio data as a NumPy array.
    """
    print("Recording...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")
    return recording.flatten()


def transcribe_audio(audio, model="base"):
    """
    Transcribe audio to text using Whisper.
    Args:
    audio (np.ndarray): Audio data.
    model (str): Model size of Whisper to use.

    Returns:
    str: Transcribed text.
    """
    # Load the Whisper model
    model = whisper.load_model(model)
    print("Transcribing...")
    # Transcribe the audio
    result = model.transcribe(audio)
    return result["text"]


if __name__ == "__main__":
    # Parameters
    DURATION = 3
    SAMPLE_RATE = 16000  # Audio sample rate

    # Record the audio
    audio_data = record_audio(DURATION, SAMPLE_RATE)

    # Transcribe the recorded audio
    transcription = transcribe_audio(audio_data)
    print("Transcription:", transcription)
