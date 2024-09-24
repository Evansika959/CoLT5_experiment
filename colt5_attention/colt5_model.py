from colt5_attention.transformer_block import ConditionalRoutedTransformerBlock, ConditionalRoutedDecoderBlock
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Tokenizer

# Define the CoLT5 Encoder

seq_len = 512
num_heavy_tokens = 32

class CoLT5Encoder(nn.Module):
    def __init__(self, num_layers, dim):
        super(CoLT5Encoder, self).__init__()
        self.layers = nn.ModuleList([ConditionalRoutedTransformerBlock(dim, num_heavy_attn_tokens_kv=num_heavy_tokens, num_heavy_attn_tokens_q=num_heavy_tokens, num_heavy_ff_tokens=num_heavy_tokens) for _ in range(num_layers)])
        self.embed_tokens = nn.Embedding(32128, dim)  # Vocab size of 32,128 tokens

    def forward(self, input_ids, mask=None):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, mask)
        return x


# Define the CoLT5 Decoder with Cross-Attention
class CoLT5Decoder(nn.Module):
    def __init__(self, num_layers, dim):
        super(CoLT5Decoder, self).__init__()
        self.layers = nn.ModuleList([ConditionalRoutedDecoderBlock(dim, num_heavy_attn_tokens_kv=num_heavy_tokens, num_heavy_attn_tokens_q=num_heavy_tokens, num_heavy_ff_tokens=num_heavy_tokens) for _ in range(num_layers)])
        self.embed_tokens = nn.Embedding(32128, dim)  # Vocab size of 32,128 tokens

    def forward(self, input_ids, encoder_hidden_states, mask=None):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, encoder_hidden_states, mask)
        return x


# Define the full CoLT5 model
class CoLT5(nn.Module):
    def __init__(self, num_layers, dim):
        super(CoLT5, self).__init__()
        self.encoder = CoLT5Encoder(num_layers=num_layers, dim=dim)
        self.decoder = CoLT5Decoder(num_layers=num_layers, dim=dim)
        self.lm_head = nn.Linear(dim, 32128)  # Output layer to vocab size
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')

    def forward(self, input_ids, decoder_input_ids, mask=None, decoder_mask=None):
        # Encode the input
        encoder_hidden_states = self.encoder(input_ids, mask=mask)
        # Decode the input
        decoder_hidden_states = self.decoder(decoder_input_ids, encoder_hidden_states=encoder_hidden_states, mask=decoder_mask)
        # Generate final token predictions
        logits = self.lm_head(decoder_hidden_states)
        return logits
    

    def generate(self, input_ids, encoder_mask, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate text from the CoLT5 model.

        Args:
            input_ids (torch.Tensor): The input tensor of shape (batch_size, seq_length).
            max_new_tokens (int): The maximum number of tokens to generate.
            temperature (float): Temperature for sampling.
            top_k (int, optional): If specified, only the top_k tokens will be considered for sampling.

        Returns:
            Generated sequence (torch.Tensor).
        """
        # Set the model to evaluation mode
        self.eval()
        
        # Initialize decoder input with a PAD token
        decoder_input_ids = torch.full((1, 1), self.tokenizer.pad_token_id, dtype=torch.long).to('cuda')

        for _ in range(max_new_tokens):
             # Prepare the causal mask
            seq_length = decoder_input_ids.size(1)
            decoder_mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool().to('cuda')

            # Forward pass through the model
            with torch.no_grad():
                logits = self(input_ids=input_ids, decoder_input_ids=decoder_input_ids, mask=encoder_mask, decoder_mask=decoder_mask)
            
            # Get the logits for the last token in the sequence
            logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample from the distribution
            next_token_id = torch.multinomial(probs, num_samples=1)

            # Append the predicted token to the decoder input
            decoder_input_ids = torch.cat((decoder_input_ids, next_token_id), dim=1)

            # Print the current output word
            current_word = self.tokenizer.decode(next_token_id.item())
            print(f"Generated: {current_word}")

            # If the predicted token is the end of sequence token, break
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break

        return decoder_input_ids

