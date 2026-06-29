import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

LORA_MODEL_PATH = "./finetuning/model"
BASE_MODEL_NAME = "bert-base-multilingual-cased"

print("Caricamento Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

print("Caricamento Modello Base...")
base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL_NAME,
    num_labels=2
)

print("Iniezione dei pesi LoRA...")
model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)
model.eval() # Modalità inferenza

def predici_testo(testo: str):
    inputs = tokenizer(testo, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predizione = torch.argmax(logits, dim=-1).item()

    classe = "INFORMATIVE (1) 🟢" if predizione == 1 else "JUNK/STORE (0) 🔴"
    return classe

if __name__ == "__main__":
    testi_da_provare = [
        "Recensione completa di Elden Ring: il combat system è punitivo ma incredibilmente appagante. La lore scritta da George R.R. Martin è criptica.",
        "Acquista ora la tua copia per PS5 a soli 59,99€! Aggiungi al carrello, spedizione gratuita in 24 ore. Effettua il login per i punti fedeltà."
    ]

    print("\n--- INIZIO TEST INFERENZA ---")
    for t in testi_da_provare:
        print(f"\nTesto: {t[:80]}...")
        print(f"Risultato: {predici_testo(t)}")
