"""
language.py - Cebuano <-> Tagalog translation (GGUF model, CPU).

Used by main.py AFTER a live session ends.
Importing this file is cheap and safe: the model (about 2.4 GB) only loads on the
first translate() call. Run standalone for the console loop: python language.py

STATUS: untested. The GGUF loading path was proposed earlier and never confirmed
on the target machine. The prompt format is an assumption (see translate()).
"""

import os

MODEL_REPO = "nielle003/Gemma_3_4B_Cebuano_Ilokano_Tagalog"
MODEL_FILE = "gemma-3-4b-FINAL.gguf"
CHUNK_WORDS = 30  # translate() caps output at 100 tokens, so long text is split.

_model = None


def load_model():
    """Load the GGUF model once and reuse it. The first call downloads the file if needed."""
    global _model
    if _model is None:
        # Imported here so that importing this file can never fail or stall the GUI.
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama

        model_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE
        )
        _model = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=os.cpu_count(),
            verbose=False
        )
    return _model


def translate(text, source_language, target_language):
    model = load_model()
    prompt = (
        f"Translate the following {source_language} text "
        f"into {target_language}.\n\n"
        f"Text: {text}\n"
        f"Translation:"
    )

    response = model.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.2
    )

    return response["choices"][0]["message"]["content"].strip()


def split_into_chunks(text, chunk_words=CHUNK_WORDS):
    """Split text into groups of words. Live transcripts often have no punctuation."""
    words = text.split()
    return [
        " ".join(words[start:start + chunk_words])
        for start in range(0, len(words), chunk_words)
    ]


def translate_long(text, source_language, target_language, on_progress=None):
    """
    Translate a whole transcript chunk by chunk.
    on_progress(done, total) is called after each chunk, if given.
    TODO: chunking by words can cut sentences in half. Split on pauses or
    sentence boundaries once the speech engine provides them.
    """
    chunks = split_into_chunks(text)
    translated = []
    for index, chunk in enumerate(chunks, start=1):
        translated.append(translate(chunk, source_language, target_language))
        if on_progress is not None:
            on_progress(index, len(chunks))
    return " ".join(translated)


def run_console():
    while True:
        text = input("\nEnter text (or type 'exit'): ")

        if text.lower() == "exit":
            break

        direction = input(
            "Direction (1 = Cebuano → Tagalog, "
            "2 = Tagalog → Cebuano): "
        )

        if direction == "1":
            result = translate(
                text,
                "Cebuano",
                "Tagalog"
            )
        elif direction == "2":
            result = translate(
                text,
                "Tagalog",
                "Cebuano"
            )
        else:
            print("Invalid direction.")
            continue

        print("Translation:", result)


if __name__ == "__main__":
    run_console()