from TTS.api import TTS
import pygame
import os

pygame.init()

# Initialize the TTS model
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")


def generate_speech(
    text, language, speaker, output_file="output.wav", split_sentences=True
):
    """
    Generate speech for the given text in the specified language and save it to a file.

    Parameters:
    - text: The text to be spoken.
    - language: The language code for the text (e.g., 'de' for German).
    - speaker: The speaker name.
    - output_file: The output file path where the speech will be saved.
    - split_sentences: Whether to split sentences during processing.
    """
    tts.tts_to_file(
        text=text,
        file_path=output_file,
        speaker=speaker,
        language=language,
        split_sentences=split_sentences,
    )
    print(os.getcwd())
    file_path = os.path.join(os.getcwd(), output_file)

    # Check if the file exists
    if os.path.exists(file_path):
        # Load the sound file
        my_sound = pygame.mixer.Sound(file_path)

        # Play the sound
        my_sound.play()

        # Wait for the sound to finish playing
        while pygame.mixer.get_busy():
            pygame.time.Clock().tick(10)

    else:
        print("The file does not exist.")


# These are the languages this model supports
# English (en), Spanish (es), French (fr), German (de), Italian (it), Portuguese (pt), Polish (pl), Turkish (tr),
# Russian (ru), Dutch (nl), Czech (cs), Arabic (ar), Chinese (zh-cn), Japanese (ja), Hungarian (hu) and Korean (ko).
# Example usage
if __name__ == "__main__":
    # Example for german language
    # german_text = "Es hat mich eine ganze Weile gekostet, eine Stimme zu entwickeln, und jetzt, da ich sie habe, werde ich nicht still sein."
    # language_code = "de"
    # speaker_name = "Ana Florence"  # Make sure this speaker supports the chosen language

    # # Generate speech
    # generate_speech(german_text, language_code, speaker_name, output_file="output_german.wav")

    # example for english language
    text = "Today was a busy day."
    language_code = "en"
    speaker_name = "Ana Florence"  # Update the speaker name if necessary

    generate_speech(
        text,
        language_code,
        speaker_name,
        output_file="Today was a busy day.wav",
    )

    # # example for arabic language
    # arabic_text = "مرحبا كيف حالك."
    # language_code = "ar"
    # speaker_name = "Ana Florence"  # Update the speaker name if necessary

    # generate_speech(arabic_text, language_code, speaker_name, output_file="output_arabic.wav")
