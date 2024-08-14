import os
from transformers import MarianMTModel, MarianTokenizer
from huggingface_hub import HfApi

def download_marian_mt_models(src_langs, tgt_langs, save_directory="models"):
    api = HfApi()

    for src_lang in src_langs:
        for tgt_lang in tgt_langs:
            if src_lang != tgt_lang:  # Avoid downloading models for the same source and target languages
                model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
                model_dir = os.path.join(save_directory, model_name.replace('/', '_'))
                print(os.path.join(model_dir))
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                print("Model dir: ", model_dir)
                print(f"Checking model: {model_name}")
                try:
                    # Verify if the model exists on Hugging Face
                    api.model_info(model_name)

                    # If the model exists, download it
                    print(f"Downloading model: {model_name}")
                    tokenizer = MarianTokenizer.from_pretrained(model_name)
                    model = MarianMTModel.from_pretrained(model_name)
                    # Save the model and tokenizer to a specific directory
                    model.save_pretrained(model_dir)
                    tokenizer.save_pretrained(model_dir)
                    print(f"Successfully downloaded and saved model: {model_name}")
                except Exception as e:
                    print(f"Model {model_name} does not exist or error occurred: {e}")


def get_language_codes():
    language_codes = [
        'ar', 'en', 'de', 'fr', 'zh', 'es', 'ru', 'ja', 'ko', 'it',
        'nl', 'pl', 'pt', 'ro', 'sv', 'tr', 'uk', 'vi', 'el', 'cs',
        'fi', 'hu', 'da', 'no', 'he', 'id', 'th', 'sk', 'bg', 'hr',
        'lt', 'lv', 'et', 'sl', 'ms', 'mt', 'ga', 'cy', 'sq', 'sr',
        'bs', 'mk', 'is', 'hi', 'bn', 'gu', 'kn', 'ml', 'mr', 'pa',
        'ta', 'te', 'ur', 'fa', 'am', 'hy', 'az', 'ka', 'kk', 'ky',
        'tk', 'uz', 'af', 'sw', 'zu', 'xh', 'st', 'sn', 'yo', 'ig',
        'ha', 'lg', 'om', 'ti', 'rw', 'so', 'ne', 'si', 'km', 'lo',
        'my', 'dz', 'bo', 'mn', 'ps', 'ku', 'sd'
    ]
    return language_codes

if __name__ == "__main__":
    # Get the list of all language codes
    language_codes = get_language_codes()
    print(f"Total language codes to process: {len(language_codes)}")

    # Download all MarianMT models
    download_marian_mt_models(language_codes, language_codes)
