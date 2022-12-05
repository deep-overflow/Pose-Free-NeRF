import torch.nn as nn

# CNN Encoder ==================================================
class RUSTCNNEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
        )
    
    def forward(self, inputs):
        return self.block(inputs)

class RUSTCNNEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.encoder = nn.Sequential(
            RUSTCNNEncoderBlock(3, args.embed_dim // 4),
            RUSTCNNEncoderBlock(args.embed_dim // 4, args.embed_dim // 2),
            RUSTCNNEncoderBlock(args.embed_dim // 2, args.embed_dim),
        )

    def forward(self, inputs):
        return self.encoder(inputs)

# Transformer Encoder ==================================================

class RUSTTransformerEncoderBlock(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.MultiHeadAttention = nn.MultiheadAttention(
            embed_dim=self.args.embed_dim,
            num_heads=self.args.num_multi_heads,
            batch_first=True,
        )

        self.SelfMultiHeadAttention = nn.MultiheadAttention(
            embed_dim=self.args.embed_dim,
            num_heads=self.args.num_multi_heads,
            batch_first=True
        )

        self.MLP = nn.Linear(self.args.embed_dim, self.args.embed_dim)

    def forward(self, querys, SLSRs):
        embedding, _ = self.MultiHeadAttention(querys, SLSRs, SLSRs)
        embedding, _ = self.SelfMultiHeadAttention(embedding, embedding, embedding)
        embedding = self.MLP(embedding)
        return embedding

class RUSTTransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.EncoderBlocks = nn.ModuleList()

        for _ in range(self.args.num_encoder_blocks):
            self.EncoderBlocks.append(RUSTTransformerEncoderBlock(self.args))
        
    def forward(self, querys, SLSRs):
        embedding = querys
        
        for i in range(self.args.num_encoder_blocks):
            embedding = self.EncoderBlocks[i](embedding, SLSRs)
        
        return embedding

# Pose Network ==================================================

class RUSTPoseNet(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.num_features = args.img_h * args.img_w // 128

        # CNN Encoder
        self.CNNEncoder = RUSTCNNEncoder(self.args)

        # Transformer Encoder
        self.TransformerEncoder = RUSTTransformerEncoder(self.args)

        # Projection
        self.Projection = nn.Sequential(
            nn.AvgPool1d(self.args.embed_dim, 1),
            nn.Flatten(),
            nn.Linear(self.num_features, 8)
        )

    def forward(self, inputs, SLSRs):
        embeddings = self.CNNEncoder(inputs)
        embeddings = embeddings.reshape((-1, 512, self.args.embed_dim))
        embeddings = self.TransformerEncoder(embeddings, SLSRs)
        return self.Projection(embeddings)