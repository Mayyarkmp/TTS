from transformers import MarianMTModel, MarianTokenizer, TrainingArguments, Trainer
from datasets import Dataset

# Load dataset for fine-tuning to retrained model
# here you set the data like paires of source (Orginal text) : target (desired transleted text)
# you have to main categories in dataset :
# 1- train : this data for fine-tuning model
# 2- validation : this data for validation in model $$$ note : this data size should be like 20% of overall size of dataset $$$
data = {
    "train": [
        {"source": "شعبوز", "target": "schoschen"},
    ],
    "validation": [
        {"source": "انه شعبوز", "target": "das ist schoschen"},
        {"source": "شعبوز", "target": "schoschen"},
    ],
}


# this is the main function to make fine-tuning model
# u should pass the dataset and the model and the tokenizer and source language and the target language


def fineTuning_Model(dataset, model, tokenizer, src_lang, tgt_lang):
    # loading dataset and separate it to training data and validation data
    train_dataset = Dataset.from_list(dataset["train"])
    val_dataset = Dataset.from_list(dataset["validation"])

    # recombine dataset but like a dict in python
    # dataset = DatasetDict({"train": train_dataset, "validation": val_dataset})

    # Define the preprocess_function to tokenize the input and target texts.
    def preprocess_function(examples):
        inputs = [ex for ex in examples["source"]]
        targets = [ex for ex in examples["target"]]
        model_inputs = tokenizer(
            inputs, text_target=targets, max_length=128, padding=True
        )
        return model_inputs

    # Apply the preprocess function on the dataset  using the 'map' method
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
    tokenized_val_dataset = val_dataset.map(preprocess_function, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,  # inital learning rate
        per_device_train_batch_size=16,  # batch size during training
        per_device_eval_batch_size=16,  # batch size during validation
        weight_decay=0.01,  # parameter is a regularization hyperparameter that controls the strength of the penalty
        # applied to large weights during training, helping to prevent overfitting
        # and improve the generalization performance of the model.
        save_total_limit=3,  # control the maximum number of model check points
        num_train_epochs=10,  # number of epochs
        logging_dir="./logs",
        logging_steps=10,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
    )
    # Fine-tune the model
    trainer.train()
    # Save the model
    model.save_pretrained(f"./fine-tuned-marianmt-{src_lang}-{tgt_lang}")
    tokenizer.save_pretrained(f"./fine-tuned-marianmt-{src_lang}-{tgt_lang}")


# Function responsible for translation and speech
# It translates the text and says it aloud
def translate_text(text, premodel, tokenizer, src_lang, tgt_lang):
    translated = premodel.generate(**tokenizer(text, return_tensors="pt", padding=True))
    tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated][0]
    return tgt_text


def load_model(src_lang, tgt_lang):
    try:
        model_name = f"./models/Helsinki-NLP_opus-mt-{src_lang}-{tgt_lang}"
        model = MarianMTModel.from_pretrained(model_name)
        print("loaded from local machine ")
        return model
    except Exception as e:
        print(e)
        return None


def load_tokenizer(src_lang, tgt_lang):
    try:
        model_name = f"./fine-tuned-marianmt-{src_lang}-{tgt_lang}"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        print("loaded from local machine ")
        return tokenizer
        # print("x")
    except Exception as e:
        model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(f"./fine-tuned-marianmt-{src_lang}-{tgt_lang}")
        print("downloaded and saved")
        return tokenizer


"""
 Here you give the text we want to translate
 + the source language(the language of the text we want to translate)
 + the target language(the language we want to translate to)


 Note: make sure to use the right abbreviation, for example to indicate the arabic language use 'ar', to use
 english language use 'en', and so on, if you don't know the right abbreviation to the language you want
 you can look up the abbreviations in this link https://developers.google.com/admin-sdk/directory/v1/languages
"""

source_language = "ar"
target_language = "de"

model = load_model(source_language, target_language)
tokenizers = load_tokenizer(source_language, target_language)

# Example for translating arabic to german
translated_text = translate_text(
    "ترجم هذا النص من فضلك", model, tokenizers, source_language, target_language
)
print(translated_text)

fineTuning_Model(data, model, tokenizers, source_language, target_language)
