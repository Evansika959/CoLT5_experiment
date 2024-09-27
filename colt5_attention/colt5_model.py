from colt5_attention.transformer_block import ConditionalRoutedTransformerBlock, ConditionalRoutedDecoderBlock
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Tokenizer

# Define the CoLT5 Encoder

seq_len = 64
num_heavy_tokens = 4

class CoLT5Encoder(nn.Module):
    def __init__(self, num_layers, dim):
        super(CoLT5Encoder, self).__init__()
        self.layers = nn.ModuleList([ConditionalRoutedTransformerBlock(dim, num_heavy_attn_tokens_kv=num_heavy_tokens, num_heavy_attn_tokens_q=num_heavy_tokens, num_heavy_ff_tokens=num_heavy_tokens) for _ in range(num_layers)])
        self.embed_tokens = nn.Embedding(32128, dim)  # Vocab size of 32,128 tokens

    def forward(self, input_ids, mask=None, keep_routing_history=False):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, mask, keep_routing_history=keep_routing_history)
        return x


# Define the CoLT5 Decoder with Cross-Attention
class CoLT5Decoder(nn.Module):
    def __init__(self, num_layers, dim):
        super(CoLT5Decoder, self).__init__()
        self.layers = nn.ModuleList([ConditionalRoutedDecoderBlock(dim, num_heavy_attn_tokens_kv=num_heavy_tokens, num_heavy_attn_tokens_q=num_heavy_tokens, num_heavy_ff_tokens=num_heavy_tokens) for _ in range(num_layers)])
        self.embed_tokens = nn.Embedding(32128, dim)  # Vocab size of 32,128 tokens

    def forward(self, input_ids, encoder_hidden_states, mask=None, keep_routing_history=False):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, encoder_hidden_states, mask, keep_routing_history=keep_routing_history)
        return x


# Define the full CoLT5 model
class CoLT5(nn.Module):
    def __init__(self, num_layers, dim):
        super(CoLT5, self).__init__()
        self.encoder = CoLT5Encoder(num_layers=num_layers, dim=dim)
        self.decoder = CoLT5Decoder(num_layers=num_layers, dim=dim)
        self.lm_head = nn.Linear(dim, 32128)  # Output layer to vocab size
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')

    def forward(self, input_ids, decoder_input_ids, mask=None, decoder_mask=None, keep_routing_history=False):
        # Encode the input
        encoder_hidden_states = self.encoder(input_ids, mask=mask, keep_routing_history=keep_routing_history)
        # Decode the input
        decoder_hidden_states = self.decoder(decoder_input_ids, encoder_hidden_states=encoder_hidden_states, mask=mask, keep_routing_history=keep_routing_history)
        # Generate final token predictions
        logits = self.lm_head(decoder_hidden_states)
        return logits
    

    # def generate(self, input_ids, encoder_mask, max_new_tokens, temperature=1.0, top_k=None, verbose=False):
    #     """
    #     Generate text from the CoLT5 model.

    #     Args:
    #         input_ids (torch.Tensor): The input tensor of shape (batch_size, seq_length).
    #         max_new_tokens (int): The maximum number of tokens to generate.
    #         temperature (float): Temperature for sampling.
    #         top_k (int, optional): If specified, only the top_k tokens will be considered for sampling.

    #     Returns:
    #         Generated sequence (torch.Tensor).
    #     """
    #     # Set the model to evaluation mode
    #     self.eval()
        
    #     # Initialize decoder input with a PAD token
    #     decoder_input_ids = torch.full((1, 1), self.tokenizer.pad_token_id, dtype=torch.long).to('cuda')

    #     for _ in range(max_new_tokens):
    #          # Prepare the causal mask
    #         seq_length = decoder_input_ids.size(1)
    #         decoder_mask = torch.zeros(seq_length, dtype=torch.bool).to('cuda')

    #         # Forward pass through the model
    #         with torch.no_grad():
    #             logits = self(input_ids=input_ids, decoder_input_ids=decoder_input_ids, mask=encoder_mask)
            
    #         # Get the logits for the last token in the sequence
    #         logits = logits[:, -1, :] / temperature

    #         # Apply top-k filtering
    #         if top_k is not None:
    #             v, _ = torch.topk(logits, top_k)
    #             logits[logits < v[:, [-1]]] = float('-inf')

    #         # Convert logits to probabilities
    #         probs = F.softmax(logits, dim=-1)

    #         # Sample from the distribution
    #         next_token_id = torch.multinomial(probs, num_samples=1)

    #         # Append the predicted token to the decoder input
    #         decoder_input_ids = torch.cat((decoder_input_ids, next_token_id), dim=1)

    #         if verbose:
    #             # Convert the generated token IDs to a list
    #             generated_ids = decoder_input_ids.squeeze().cpu().tolist()  # Remove batch dimension and move to CPU

    #             # Decode the generated token IDs to text
    #             generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
    #             print(f"Generated: {generated_text}")

    #         # If the predicted token is the end of sequence token, break
    #         if next_token_id.item() == self.tokenizer.eos_token_id:
    #             break

    #     return decoder_input_ids

    def generate(self, input_ids, encoder_mask, max_new_tokens, temperature=1.0, top_k=None, verbose=False, keep_routing_history=False):
        """
        Generate text from the CoLT5 model while maintaining a fixed decoder sequence length.

        Args:
            input_ids (torch.Tensor): The input tensor of shape (batch_size, seq_length).
            encoder_mask (torch.Tensor): The attention mask for the encoder.
            max_new_tokens (int): The maximum number of tokens to generate.
            temperature (float): Temperature for sampling.
            top_k (int, optional): If specified, only the top_k tokens will be considered for sampling.
            verbose (bool): If True, print generated text at each step.

        Returns:
            Generated sequence (torch.Tensor).
        """
        # Set the model to evaluation mode
        self.eval()
        
        device = input_ids.device
        batch_size = input_ids.size(0)
        
        # Define the maximum decoder sequence length
        decoder_max_length = 64  # Adjust based on your model's requirements
        
        # Initialize decoder input with PAD tokens
        decoder_input_ids = torch.full((batch_size, decoder_max_length), self.tokenizer.pad_token_id, dtype=torch.long).to(device)
        
        # Optionally, set the first token to BOS (Beginning of Sequence) if applicable
        if hasattr(self.tokenizer, 'bos_token_id') and self.tokenizer.bos_token_id is not None:
            decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        else:
            # If no BOS token, you can leave it as PAD or use another starting token
            pass

        # Create a mask where the first 10 positions are True and the rest are False
        decoder_mask = torch.zeros((1, decoder_max_length), dtype=torch.bool).to('cuda')
        current_allowed_tokens = 1

        # Keep track of the current generation step per batch
        generated_steps = torch.zeros(batch_size, dtype=torch.long).to(device)
        
        for step in range(max_new_tokens):
            # Forward pass through the model
            with torch.no_grad():
                logits = self(input_ids=input_ids, decoder_input_ids=decoder_input_ids, mask=encoder_mask, keep_routing_history=keep_routing_history)
            
            # Get the logits for the last non-PAD token in each sequence
            # Find the first PAD position per batch
            pad_mask = (decoder_input_ids == self.tokenizer.pad_token_id)
            # Count non-PAD tokens per batch
            non_pad_counts = (~pad_mask).sum(dim=1)
            # The next token to predict is at position 'non_pad_counts'
            # Clamp to ensure it does not exceed decoder_max_length - 1
            next_token_positions = torch.clamp(non_pad_counts, max=decoder_max_length - 1)
            
            # Gather logits for the next token positions
            batch_indices = torch.arange(batch_size).to(device)
            next_logits = logits[batch_indices, next_token_positions, :] / temperature
            
            # Apply top-k filtering if specified
            if top_k is not None:
                # Get the top_k logits
                topk_logits, topk_indices = torch.topk(next_logits, top_k, dim=-1)
                # Create a mask for logits not in the top_k
                mask = next_logits < topk_logits[:, -1].unsqueeze(-1)
                next_logits[mask] = float('-inf')
            
            # Convert logits to probabilities
            probs = F.softmax(next_logits, dim=-1)
            
            # Sample the next token
            next_token_id = torch.multinomial(probs, num_samples=1).squeeze(1)  # Shape: (batch_size,)
            
            # Insert the new token into the decoder_input_ids at the next_token_positions
            decoder_input_ids[batch_indices, next_token_positions] = next_token_id
            
            if verbose:
                # Convert the generated token IDs to a list
                generated_ids = decoder_input_ids.cpu().tolist()
        
                # Decode the generated token IDs to text
                generated_text = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]
                for text in generated_text:
                    print(f"Generated: {text}")
            
            # Check for end-of-sequence tokens and stop generation if all batches have generated EOS
            eos_mask = (next_token_id == self.tokenizer.eos_token_id)
            if eos_mask.all():
                break

            if current_allowed_tokens < decoder_max_length:
                current_allowed_tokens += 1
                decoder_mask[:, current_allowed_tokens - 1] = True  # Add one more True
                
                if verbose:
                    print(f"Step {step + 1}: Allowed tokens up to position {current_allowed_tokens}")
            
            # Optionally, stop generating for individual batches that have generated EOS
            # This requires handling each batch separately, which is more complex
            # For simplicity, this example stops only if all batches have generated EOS

        return decoder_input_ids


