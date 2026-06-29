import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

class MLEvaluator:
    _instance = None

    def __new__(cls): # cls rappresenta la classe, in questo modo siamo sicuri che ci sia sempre e solo 1 classificatore caricato in memoria (singleton)
        if cls._instance is None:
            cls._instance = super(MLEvaluator, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        print("[ML Manager] Caricamento del classificatore LoRA in memoria...")
        self.lora_path = "./finetuning/model"
        self.base_name = "bert-base-multilingual-cased"

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_name)
            base_model = AutoModelForSequenceClassification.from_pretrained(self.base_name, num_labels=2)
            self.model = PeftModel.from_pretrained(base_model, self.lora_path)
            self.model.eval()
            print("[ML Manager] Modello caricato con successo! Pronto per filtrare.")
        except Exception as e:
            print(f"[ML Manager] ERRORE GRAVE di caricamento: {e}")
            self.model = None

    def is_informative(self, text: str) -> bool:
        """Restituisce True se il testo è Informative (1), False se è Junk (0)"""
        if self.model is None:
            return True

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1).item()

        return prediction == 1
