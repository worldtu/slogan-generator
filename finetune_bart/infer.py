import torch

class SloganGenerator:
    """
    Generates slogans from a fine-tuned model based on DistilBART.
    """
    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    def generate(self, desc: str, max_length: int = 64, num_beams: int = 4, **kwargs) -> str:
        self.model.eval()

        inputs = self.tokenizer(
            desc,
            max_length=128, # Max input length for description
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length, # Max length of the generated slogan
            num_beams=num_beams,
            early_stopping=True,
            **kwargs
        )
        
        # Decode the generated ids
        generated_slogan = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_slogan
