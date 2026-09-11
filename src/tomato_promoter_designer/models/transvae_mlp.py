from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


DNA_TO_TOKEN = {"A": 0, "C": 1, "G": 2, "T": 3}
TISSUE_ORDER = ("root", "stem", "leaf", "fruit")


@dataclass(frozen=True, slots=True)
class TransVAEConfig:
    sequence_length: int = 165
    vocabulary_size: int = 23
    output_vocabulary_size: int = 22
    model_dimension: int = 128
    feedforward_dimension: int = 512
    latent_dimension: int = 128
    attention_heads: int = 4
    layers: int = 3
    dropout: float = 0.1


def _clones(module: nn.Module, count: int) -> nn.ModuleList:
    return nn.ModuleList(copy.deepcopy(module) for _ in range(count))


def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout: nn.Dropout | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
    if mask is not None:
        if mask.size(-1) > scores.size(-1):
            mask = mask[..., : scores.size(-1)]
        elif mask.size(-1) < scores.size(-1):
            mask = F.pad(mask, (0, scores.size(-1) - mask.size(-1)), value=True)
        scores = scores.masked_fill(~mask.bool(), -1e9)
    probabilities = F.softmax(scores, dim=-1)
    if dropout is not None:
        probabilities = dropout(probabilities)
    return torch.matmul(probabilities, value), probabilities


class LayerNorm(nn.Module):
    def __init__(self, features: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = tensor.mean(-1, keepdim=True)
        std = tensor.std(-1, keepdim=True)
        return self.a_2 * (tensor - mean) / (std + self.epsilon) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size: int, dropout: float) -> None:
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tensor: torch.Tensor, sublayer) -> torch.Tensor:
        return tensor + self.dropout(sublayer(self.norm(tensor)))


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads: int, model_dimension: int, dropout: float = 0.1) -> None:
        super().__init__()
        if model_dimension % heads != 0:
            raise ValueError("model_dimension must be divisible by attention_heads")
        self.d_k = model_dimension // heads
        self.h = heads
        self.linears = _clones(nn.Linear(model_dimension, model_dimension), 4)
        self.dropout = nn.Dropout(dropout)
        self.attn: torch.Tensor | None = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is not None:
            mask = mask.unsqueeze(1)
        batches = query.size(0)
        query, key, value = [
            linear(tensor).view(batches, -1, self.h, self.d_k).transpose(1, 2)
            for linear, tensor in zip(self.linears, (query, key, value))
        ]
        attended, self.attn = _attention(query, key, value, mask, self.dropout)
        attended = attended.transpose(1, 2).contiguous().view(batches, -1, self.h * self.d_k)
        return self.linears[-1](attended)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, model_dimension: int, hidden_dimension: int, dropout: float) -> None:
        super().__init__()
        self.w_1 = nn.Linear(model_dimension, hidden_dimension)
        self.w_2 = nn.Linear(hidden_dimension, model_dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(F.relu(self.w_1(tensor))))


class PositionalEncoding(nn.Module):
    def __init__(self, model_dimension: int, dropout: float, max_length: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        encoding = torch.zeros(max_length, model_dimension)
        position = torch.arange(0, max_length, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, model_dimension, 2, dtype=torch.float32)
            * -(math.log(10000.0) / model_dimension)
        )
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("pe", encoding.unsqueeze(0))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor + self.pe[:, : tensor.size(1)].requires_grad_(False)
        return self.dropout(tensor)


class Embeddings(nn.Module):
    def __init__(self, model_dimension: int, vocabulary_size: int) -> None:
        super().__init__()
        self.lut = nn.Embedding(vocabulary_size, model_dimension)
        self.model_dimension = model_dimension

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.lut(tokens) * math.sqrt(self.model_dimension)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        size: int,
        source_length: int,
        self_attention: MultiHeadedAttention,
        feed_forward: PositionwiseFeedForward,
        dropout: float,
    ) -> None:
        super().__init__()
        self.size = size
        self.src_len = source_length
        self.self_attn = self_attention
        self.feed_forward = feed_forward
        self.sublayer = _clones(SublayerConnection(size, dropout), 2)

    def forward(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tensor = self.sublayer[0](
            tensor, lambda value: self.self_attn(value, value, value, mask)
        )
        return self.sublayer[1](tensor, self.feed_forward)


class DecoderLayer(nn.Module):
    def __init__(
        self,
        size: int,
        target_length: int,
        self_attention: MultiHeadedAttention,
        source_attention: MultiHeadedAttention,
        feed_forward: PositionwiseFeedForward,
        dropout: float,
    ) -> None:
        super().__init__()
        self.size = size
        self.tgt_len = target_length
        self.self_attn = self_attention
        self.src_attn = source_attention
        self.feed_forward = feed_forward
        self.sublayer = _clones(SublayerConnection(size, dropout), 3)

    def forward(
        self,
        tensor: torch.Tensor,
        memory_key: torch.Tensor,
        memory_value: torch.Tensor,
        source_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        tensor = self.sublayer[0](
            tensor, lambda value: self.self_attn(value, value, value, target_mask)
        )
        tensor = self.sublayer[1](
            tensor,
            lambda value: self.src_attn(value, memory_key, memory_value, source_mask),
        )
        return self.sublayer[2](tensor, self.feed_forward)


class ConvBottleneck(nn.Module):
    def __init__(self, size: int, source_length: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        self.conv_list: list[int] = []
        input_channels = size
        current_length = source_length
        for index, kernel_size in enumerate((9, 8, 8)):
            output_channels = int((input_channels - 64) // 2 + 64)
            if index == 2:
                output_channels = 64
            layers.append(
                nn.Sequential(
                    nn.Conv1d(input_channels, output_channels, kernel_size),
                    nn.MaxPool1d(kernel_size=2),
                )
            )
            input_channels = output_channels
            current_length = ((current_length - kernel_size) + 1) // 2
            self.conv_list.append(current_length)
        self.conv_layers = nn.ModuleList(layers)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        for layer in self.conv_layers:
            tensor = F.relu(layer(tensor))
        return tensor


class DeconvBottleneck(nn.Module):
    def __init__(self, size: int, source_length: int, conv_lengths: list[int]) -> None:
        super().__init__()
        shapes = [source_length + 1, *conv_lengths]
        layers: list[nn.Module] = []
        input_channels = 64
        for index in range(3):
            input_length = shapes[3 - index]
            output_length = shapes[2 - index]
            output_channels = (size - input_channels) // 4 + input_channels
            if index == 2:
                output_channels = size
            kernel_size = (output_length - 1) - 2 * (input_length - 1) + 1
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose1d(
                        input_channels, output_channels, kernel_size, stride=2
                    )
                )
            )
            input_channels = output_channels
        self.deconv_layers = nn.ModuleList(layers)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        for layer in self.deconv_layers:
            tensor = F.relu(layer(tensor))
        return tensor


class VAEEncoder(nn.Module):
    def __init__(self, layer: EncoderLayer, count: int, latent_dimension: int) -> None:
        super().__init__()
        self.layers = _clones(layer, count)
        self.conv_bottleneck = ConvBottleneck(layer.size, layer.src_len)
        self.norm = LayerNorm(layer.size)
        flattened_dimension = 64 * self.conv_bottleneck.conv_list[-1]
        self.predict_len1 = nn.Linear(latent_dimension, latent_dimension * 2)
        self.predict_len2 = nn.Linear(latent_dimension * 2, layer.size)
        self.z_means = nn.Linear(flattened_dimension, latent_dimension)
        self.z_var = nn.Linear(flattened_dimension, latent_dimension)

    def encode_distribution(
        self, tensor: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            tensor = layer(tensor, mask)
        memory = self.norm(tensor).permute(0, 2, 1)
        memory = self.conv_bottleneck(memory).flatten(start_dim=1)
        return self.z_means(memory), self.z_var(memory)

    def prediction_representation(self, mean: torch.Tensor) -> torch.Tensor:
        # The retained joint wrapper feeds this historical length head to the MLP.
        return self.predict_len2(self.predict_len1(mean))

    def forward(
        self, tensor: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_variance = self.encode_distribution(tensor, mask)
        latent = mean + torch.randn_like(mean) * torch.exp(0.5 * log_variance)
        predicted_length = self.prediction_representation(mean)
        return latent, mean, log_variance, predicted_length


class VAEDecoder(nn.Module):
    def __init__(
        self,
        encoder_layer: EncoderLayer,
        decoder_layer: DecoderLayer,
        count: int,
        latent_dimension: int,
        conv_lengths: list[int],
    ) -> None:
        super().__init__()
        self.final_encodes = _clones(encoder_layer, 1)
        self.layers = _clones(decoder_layer, count)
        self.norm = LayerNorm(decoder_layer.size)
        self.tgt_len = decoder_layer.tgt_len
        self.conv_out = conv_lengths[-1]
        self.deconv_bottleneck = DeconvBottleneck(
            decoder_layer.size, encoder_layer.src_len, conv_lengths
        )
        self.linear = nn.Linear(latent_dimension, 64 * self.conv_out)

    def forward(
        self,
        tensor: torch.Tensor,
        memory: torch.Tensor,
        source_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = F.relu(self.linear(memory)).view(-1, 64, self.conv_out)
        memory = self.deconv_bottleneck(memory).permute(0, 2, 1)
        for layer in self.final_encodes:
            memory = layer(memory, source_mask)
        memory = self.norm(memory)
        for layer in self.layers:
            tensor = layer(tensor, memory, memory, source_mask, target_mask)
        tensor = self.norm(tensor)
        if tensor.size(1) > self.tgt_len:
            tensor = tensor[:, : self.tgt_len]
        elif tensor.size(1) < self.tgt_len:
            tensor = F.pad(tensor, (0, 0, 0, self.tgt_len - tensor.size(1)))
        return tensor


class TokenGenerator(nn.Module):
    def __init__(self, model_dimension: int, output_vocabulary_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(model_dimension, output_vocabulary_size)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.proj(tensor)


class TransformerVAE(nn.Module):
    def __init__(self, config: TransVAEConfig) -> None:
        super().__init__()
        self.config = config
        attention = MultiHeadedAttention(
            config.attention_heads, config.model_dimension, config.dropout
        )
        feed_forward = PositionwiseFeedForward(
            config.model_dimension, config.feedforward_dimension, config.dropout
        )
        encoder_layer = EncoderLayer(
            config.model_dimension,
            config.sequence_length,
            copy.deepcopy(attention),
            copy.deepcopy(feed_forward),
            config.dropout,
        )
        decoder_layer = DecoderLayer(
            config.model_dimension,
            config.sequence_length,
            copy.deepcopy(attention),
            copy.deepcopy(attention),
            copy.deepcopy(feed_forward),
            config.dropout,
        )
        self.encoder = VAEEncoder(encoder_layer, config.layers, config.latent_dimension)
        self.decoder = VAEDecoder(
            encoder_layer,
            decoder_layer,
            config.layers,
            config.latent_dimension,
            self.encoder.conv_bottleneck.conv_list,
        )
        self.src_embed = nn.Sequential(
            Embeddings(config.model_dimension, config.vocabulary_size),
            PositionalEncoding(config.model_dimension, config.dropout),
        )
        self.tgt_embed = nn.Sequential(
            Embeddings(config.model_dimension, config.vocabulary_size),
            PositionalEncoding(config.model_dimension, config.dropout),
        )
        self.generator = TokenGenerator(
            config.model_dimension, config.output_vocabulary_size
        )

    def encode_distribution(
        self, tokens: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder.encode_distribution(self.src_embed(tokens), mask)

class ExpressionPredictor(nn.Module):
    def __init__(self, latent_dimension: int, targets: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dimension, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, targets),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


class TransVAEMLP(nn.Module):
    def __init__(self, config: TransVAEConfig | None = None) -> None:
        super().__init__()
        self.config = config or TransVAEConfig()
        self.transvae = TransformerVAE(self.config)
        self.vocab_to_base = nn.Linear(self.config.output_vocabulary_size, 4)
        self.predictor = ExpressionPredictor(self.config.latent_dimension)

    def score_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.size(1) != self.config.sequence_length:
            raise ValueError(
                "TransVAE-MLP scoring expects a two-dimensional [batch, 165] token tensor."
            )
        source = tokens[:, :-1]
        # Compatibility behavior: the retained wrapper used token 0 for both A and padding.
        source_mask = (source != DNA_TO_TOKEN["A"]).unsqueeze(-2)
        mean, _ = self.transvae.encode_distribution(source, source_mask)
        representation = self.transvae.encoder.prediction_representation(mean)
        return self.predictor(representation)


def fruit_bias_fitness(
    scores: torch.Tensor, negative_penalty: float = 10.0
) -> torch.Tensor:
    """Compute the thesis GA objective without performing unvalidated design."""
    if scores.ndim != 2 or scores.size(1) != len(TISSUE_ORDER):
        raise ValueError("scores must have shape [batch, 4] in root/stem/leaf/fruit order.")
    if negative_penalty < 0:
        raise ValueError("negative_penalty must be non-negative.")
    non_fruit = scores[:, :3]
    margin = scores[:, 3] - non_fruit.max(dim=1).values
    penalty = negative_penalty * torch.relu(-non_fruit).sum(dim=1)
    return margin - penalty


def encode_dna(sequence: str, expected_length: int = 165) -> torch.Tensor:
    normalized = sequence.strip().upper()
    if len(normalized) != expected_length:
        raise ValueError(
            f"TransVAE-MLP requires {expected_length}-bp sequences; received {len(normalized)} bp."
        )
    invalid = sorted(set(normalized) - set(DNA_TO_TOKEN))
    if invalid:
        raise ValueError(
            "TransVAE-MLP accepts only unambiguous A/C/G/T sequences; "
            f"found {', '.join(invalid)}."
        )
    return torch.tensor([DNA_TO_TOKEN[base] for base in normalized], dtype=torch.long)


def load_transvae_mlp(
    checkpoint: str | Path, device: str | torch.device = "cpu"
) -> TransVAEMLP:
    target_device = torch.device(device)
    model = TransVAEMLP().to(target_device)
    state = torch.load(Path(checkpoint), map_location=target_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model
