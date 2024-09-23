from colt5_attention.transformer_block import ConditionalRoutedTransformerBlock, ConditionalRoutedDecoderBlock
import torch
import torch.nn as nn
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

    def forward(self, input_ids, decoder_input_ids, mask=None):
        # Encode the input
        encoder_hidden_states = self.encoder(input_ids, mask)
        # Decode the input
        decoder_hidden_states = self.decoder(decoder_input_ids, encoder_hidden_states=encoder_hidden_states, mask=mask)
        # Generate final token predictions
        logits = self.lm_head(decoder_hidden_states)
        return logits
