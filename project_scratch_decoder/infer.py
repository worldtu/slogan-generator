import torch
import torch.nn.functional as F

class SloganGenerator:
    """
    Generates slogans from a trained decoder-only model.
    """
    def __init__(self, model, tokenizer, device: str = "cpu",
                str_start: str = "Desctiption: ", str_end: str = "Generate Slogan: "):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.str_start = str_start
        self.str_end = str_end

    def generate_greedy(self, desc: str, max_len: int = 32) -> str:
        """
        Generates a slogan using greedy decoding.
        """
        self.model.eval()
        init = f"{self.tokenizer.bos_token} {self.str_start}{desc} {self.str_end}"
        ids = torch.tensor(self.tokenizer.encode(init), dtype=torch.long, device=self.device).unsqueeze(1)
        for _ in range(max_len):
            with torch.no_grad():
                logits = self.model(ids)
            next_id = logits[-1].argmax(dim=-1, keepdim=True)
            next_id = next_id.view(1, 1)
            ids = torch.cat([ids, next_id], dim=0)
            # if next_id.item() == self.tokenizer.eos_token_id:
            #     break
        seq = ids.squeeze(1).tolist()
        return self.tokenizer.decode(seq, skip_special_tokens=True)

    def generate_beam(self, desc: str, max_len: int = 32, beam_width: int = 5):
        """
        Generates a slogan using beam search.
        """
        
        prompt = f"{self.tokenizer.bos_token} {self.str_start}{desc} {self.str_end}"
        eos_token_id = self.tokenizer.eos_token_id

        # Encode the prompt
        input_ids = torch.tensor(self.tokenizer.encode(prompt), device=self.device).unsqueeze(1)   # shape [T0, 1]
        # Beam stores tuples (sequence_ids, score)
        beams = [(input_ids, 0.0)]

        for _ in range(max_len):
            all_candidates = []
            for seq, score in beams:
                # Forward one step
                with torch.no_grad():
                    logits = self.model(seq)           # [T, 1, V]
                
                # Takes the raw scores (logits) for the last time step (-1) of the first batch element (0).
                # Applies the softmax function to turn scores into probabilities, then a log to get log-probabilities.
                # Using log-probs is numerically more stable when you’re summing scores across multiple steps (as in beam search).
                log_probs = F.log_softmax(logits[-1, 0], dim=-1)
                # Finds the beam_width highest values in the log_probs vector.
                # Returns two tensors:
                #     •	topk_logps: the top log-probability values
                #     •	topk_ids: the corresponding token indices in your vocabulary
                # These give you the best candidate next tokens and their scores to expand each beam.
                topk_logps, topk_ids = torch.topk(log_probs, beam_width)

                # Expand each candidate
                for logp, token_id in zip(topk_logps.tolist(), topk_ids.tolist()):
                    new_seq = torch.cat([seq, torch.tensor([[token_id]], device=self.device)], dim=0)
                    new_score = score + logp
                    all_candidates.append((new_seq, new_score))

            # Select the best beam_width beams
            beams = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

            # If every beam ended in <eos>, we’re done
            if all(seq[-1, 0].item() == eos_token_id for seq, _ in beams):
                break

        # Choose the highest-scoring sequence
        best_seq = beams[0][0].squeeze().tolist()
        # Decode and strip off the prompt portion
        # (assumes prompt length = len(self.tokenizer.encode(prompt)))
        return self.tokenizer.decode(best_seq)
