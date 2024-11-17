import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    # constructor: Need to tell the d_model(=512, as per paper) and the vocab_size
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    # operation: maps numbers to the same vector every time
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    # constructor
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len # max length of the input sentence = seq_len
        self.dropout_layer = nn.Dropout(dropout) # Fixed: Create dropout layer instead of storing float

        # create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(1000.0) / d_model))
        # sin= even and cos = odd positions
        # apply the sin to even pos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout_layer(x) # Fixed: Use dropout_layer instead of dropout


class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps

        self.alpha = nn.Parameter(torch.ones(1)) # multiplicative
        self.bias = nn.Parameter(torch.zeros(1)) # addetive

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)

        return self.alpha * (x -mean) / (std + self.eps) + self.bias




    
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))



class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model must be divisible by h"
        
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model) # Wq
        self.w_k = nn.Linear(d_model, d_model) # Wk  
        self.w_v = nn.Linear(d_model, d_model) # Wv
        self.w_o = nn.Linear(d_model, d_model) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):  # Fixed syntax error in parameter
        d_k = query.shape[-1]
        # query: (batch, h, seq_len, d_k)
        # key: (batch, h, seq_len, d_k) 
        # key.transpose: (batch, h, d_k, seq_len)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # Fixed using d_k instead of self.d_k
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)  # Fixed variable name
        if dropout is not None:
            attention_scores = dropout(attention_scores)  # Added dropout application
        attention_scores = attention_scores.softmax(dim = -1)  # (batch, h, seq_len, seq_len)

        return (attention_scores @ value), attention_scores


    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]

        # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # (batch, seq_len, d_model) -> (batch, h, seq_len, d_k)
        query = query.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        key = key.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)  # Fixed using key instead of k
        value = value.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (batch, h, seq_len, d_k) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        # Apply layer normalization first, then sublayer, then dropout
        # Finally add the original input (residual connection)
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForward, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask=None):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForward, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
        return self.proj(x)

class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbeddings,
        tgt_embed: InputEmbeddings,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        projection_layer: ProjectionLayer
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len) -> (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        # (batch, seq_len) -> (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
        return self.projection_layer(x)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(encoder_output, src_mask, tgt, tgt_mask)
        return self.project(decoder_output)

def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048
) -> Transformer:
    # Create embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward = FeedForward(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention, feed_forward, dropout)
        encoder_blocks.append(encoder_block)

    # Create decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward = FeedForward(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention, decoder_cross_attention, feed_forward, dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(
        encoder,
        decoder,
        src_embed,
        tgt_embed,
        src_pos,
        tgt_pos,
        projection_layer
    )

    # Initialize parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer

def main():
    # Test parameters
    d_model = 512
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    src_seq_len = 50
    tgt_seq_len = 50
    batch_size = 2
    h = 8  # Number of heads
    N = 6  # Number of encoder/decoder blocks
    dropout = 0.1
    d_ff = 2048

    # Create sample input data
    src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))  # Source sequence
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))  # Target sequence

    # Create transformer model
    transformer = build_transformer(
        src_vocab_size,
        tgt_vocab_size,
        src_seq_len,
        tgt_seq_len,
        d_model,
        N,
        h,
        dropout,
        d_ff
    )

    # Test forward pass
    out = transformer(src, tgt)
    print("Transformer output shape:", out.shape)
    print(f"Expected shape: [batch_size={batch_size}, tgt_seq_len={tgt_seq_len}, tgt_vocab_size={tgt_vocab_size}]")

    # Test individual components
    src_embed = transformer.src_embed
    src_pos = transformer.src_pos
    encoder = transformer.encoder
    decoder = transformer.decoder

    # Test source embeddings and positional encoding
    src_embedded = src_embed(src)
    print("\nSource embedding shape:", src_embedded.shape)
    src_encoded = src_pos(src_embedded)
    print("Source positional encoding shape:", src_encoded.shape)

    # Test encoder
    encoder_output = encoder(src_encoded)
    print("\nEncoder output shape:", encoder_output.shape)

    # Test target embeddings and positional encoding
    tgt_embedded = transformer.tgt_embed(tgt)
    tgt_encoded = transformer.tgt_pos(tgt_embedded)
    print("\nTarget embedding shape:", tgt_embedded.shape)
    print("Target positional encoding shape:", tgt_encoded.shape)

    # Test decoder
    decoder_output = decoder(tgt_encoded, encoder_output)
    print("\nDecoder output shape:", decoder_output.shape)

    # Test projection layer
    final_output = transformer.projection_layer(decoder_output)
    print("\nFinal projection output shape:", final_output.shape)

if __name__ == "__main__":
    main()
