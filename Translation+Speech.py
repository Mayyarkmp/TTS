from transformers import MarianMTModel, MarianTokenizer


# Function responsible for translation and speech
# It translates the text and says it aloud
def translate_text(text, src_lang, tgt_lang):
    model_name = f"./models/Helsinki-NLP_opus-mt-{src_lang}-{tgt_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
    tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated][0]

    return tgt_text


"""
 Here you give the text we want to translate
 + the source language (the language of the text we want to translate)
 + the target language (the language we want to translate to)

 Note: make sure to use the right abbreviation, for example to indicate the Arabic language use 'ar', to use
 English language use 'en', and so on. If you don't know the right abbreviation for the language you want,
 you can look up the abbreviations in this link https://developers.google.com/admin-sdk/directory/v1/languages
"""

# Example for translating Arabic to English
source_language = "ar"
target_language = "en"
text = "مرحبا بك في لبنان"
translated_text = translate_text(text, source_language, target_language)
print(translated_text)
