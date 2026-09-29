import os

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

MODEL_REPO = "nielle003/Gemma_3_4B_Cebuano_Ilokano_Tagalog"
MODEL_FILE = "gemma-3-4b-FINAL.gguf"

model_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename=MODEL_FILE
)

model = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=os.cpu_count(),
    verbose=False
)


def translate(text, source_language, target_language):
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