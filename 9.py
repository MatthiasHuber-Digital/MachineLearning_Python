# transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import enum

from torch import Tensor
from torchinfo import summary
from typing import List, Optional

class TokenInitialization(enum.Enum):
    UNIFORM = 'uniform'
    NORMAL = 'normal'

    @classmethod
    def from_str(cls, initialization: str) -> 'TokenInitialization':
        return cls(initialization)

    def apply(self, x: Tensor, d: int) -> None:
        d_sqrt_inv = 1 / math.sqrt(d)
        if self == TokenInitialization.UNIFORM:
            # used in the paper "Revisiting Deep Learning Models for Tabular Data";
            # is equivalent to `nn.init.kaiming_uniform_(x, a=math.sqrt(5))` (which is
            # used by torch to initialize nn.Linear.weight, for example)
            nn.init.uniform_(x, a=-d_sqrt_inv, b=d_sqrt_inv)
        elif self == TokenInitialization.NORMAL:
            nn.init.normal_(x, std=d_sqrt_inv)

class NumericalFeatureTokenizer(nn.Module):
    """
    Transforms continuous features to tokens (embeddings).
    See `FeatureTokenizer` for the illustration.
    For one feature, the transformation consists of two steps:
    * the feature is multiplied by a trainable vector
    * another trainable vector is added
    Note that each feature has its separate pair of trainable vectors, i.e. the vectors
    are not shared between features.
    """
    def __init__(self, n_features: int, d_token: int, bias: bool, initialization: str) -> None:
        """
        Args:
            n_features: the number of continuous (scalar) features
            d_token: the size of one token
            bias: if `False`, then the transformation will include only multiplication.
                **Warning**: :code:`bias=False` leads to significantly worse results for
                Transformer-like (token-based) architectures.
            initialization: initialization policy for parameters. Must be one of
                :code:`['uniform', 'normal']`. Let :code:`s = d ** -0.5`. Then, the
                corresponding distributions are :code:`Uniform(-s, s)` and :code:`Normal(0, s)`.
        """
        super().__init__()
        initialization_ = TokenInitialization.from_str(initialization)
        self.weight = nn.Parameter(Tensor(n_features, d_token))
        self.bias = nn.Parameter(Tensor(n_features, d_token)) if bias else None
        for parameter in [self.weight, self.bias]:
            if parameter is not None:
                initialization_.apply(parameter, d_token)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.weight[None] * x[..., None]
        if self.bias is not None:
            x = x + self.bias[None]
        return x
    
class CategoricalFeatureTokenizer(nn.Module):
    """
    Transforms categorical features to tokens (embeddings).
    See `FeatureTokenizer` for the illustration.
    The module efficiently implements a collection of `torch.nn.Embedding` (with
    optional biases).
    """

    def __init__(self, cardinalities: List[int], d_token: int, bias: bool, initialization: str) -> None:
        """
        Args:
            cardinalities: the number of distinct values for each feature. For example,
                :code:`cardinalities=[3, 4]` describes two features: the first one can
                take values in the range :code:`[0, 1, 2]` and the second one can take
                values in the range :code:`[0, 1, 2, 3]`.
            d_token: the size of one token.
            bias: if `True`, for each feature, a trainable vector is added to the
                embedding regardless of feature value. The bias vectors are not shared
                between features.
            initialization: initialization policy for parameters. Must be one of
                :code:`['uniform', 'normal']`. Let :code:`s = d ** -0.5`. Then, the
                corresponding distributions are :code:`Uniform(-s, s)` and :code:`Normal(0, s)`. In
                the paper [gorishniy2021revisiting], the 'uniform' initialization was
                used.
        """
        super().__init__()
    
        initialization_ = TokenInitialization.from_str(initialization)

        category_offsets = torch.tensor([0] + cardinalities[:-1]).cumsum(0)
        self.register_buffer('category_offsets', category_offsets, persistent=False)
        self.embeddings = nn.Embedding(sum(cardinalities), d_token)
        self.bias = nn.Parameter(Tensor(len(cardinalities), d_token)) if bias else None

        for parameter in [self.embeddings.weight, self.bias]:
            if parameter is not None:
                initialization_.apply(parameter, d_token)
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.embeddings(x + self.category_offsets[None])
        if self.bias is not None:
            x = x + self.bias[None]
        return x
    

class FeatureTokenizer(nn.Module):
    """
    Combines `NumericalFeatureTokenizer` and `CategoricalFeatureTokenizer`.
    """

    def __init__(self, n_num_features: int, cat_cardinalities: List[int], d_token: int) -> None:
        """
        Args:
            n_num_features: the number of continuous features. Pass :code:`0` if there
                are no numerical features.
            cat_cardinalities: the number of unique values for each feature. See
                `CategoricalFeatureTokenizer` for details. Pass an empty list if there
                are no categorical features.
            d_token: the size of one token.
        """
        super().__init__()

        self.initialization = 'uniform'
        self.d_token = d_token

        self.num_tokenizer = (
            NumericalFeatureTokenizer(
                n_features=n_num_features,
                d_token=d_token,
                bias=True,
                initialization=self.initialization,
            )
            if n_num_features
            else None
        )

        self.cat_tokenizer = (
            CategoricalFeatureTokenizer(
                cat_cardinalities, d_token, True, self.initialization
            )
            if cat_cardinalities
            else None
        )
    
    def forward(self, x_num: Optional[Tensor], x_cat: Optional[Tensor]) -> Tensor:
        """Perform the forward pass.
        Args:
            x_num: continuous features. Must be presented if :code:`n_num_features > 0`
                was passed to the constructor.
            x_cat: categorical features (see `CategoricalFeatureTokenizer.forward` for
                details). Must be presented if non-empty :code:`cat_cardinalities` was
                passed to the constructor.
        Returns:
            tokens
        Raises:
            AssertionError: if the described requirements for the inputs are not met.
        """
        x = []
        if self.num_tokenizer is not None:
            x.append(self.num_tokenizer(x_num))
        if self.cat_tokenizer is not None:
            x.append(self.cat_tokenizer(x_cat))
        return x[0] if len(x) == 1 else torch.cat(x, dim=1)

class CLSToken(nn.Module):
    """
    [CLS]-token for BERT-like inference.
    To learn about the [CLS]-based inference, see [devlin2018bert].
    When used as a module, the [CLS]-token is appended **to the end** of each item in
    the batch.
    """

    def __init__(self, d_token: int, initialization: str) -> None:
        """
        Args:
            d_token: the size of token
            initialization: initialization policy for parameters. Must be one of
                :code:`['uniform', 'normal']`. Let :code:`s = d ** -0.5`. Then, the
                corresponding distributions are :code:`Uniform(-s, s)` and :code:`Normal(0, s)`.
        """

        super().__init__()
        initialization_ = TokenInitialization.from_str(initialization)
        self.weight = nn.Parameter(Tensor(d_token))
        initialization_.apply(self.weight, d_token)

    def expand(self, *leading_dimensions: int) -> Tensor:
        """
        Expand (repeat) the underlying [CLS]-token to a tensor with the given leading dimensions.
        A possible use case is building a batch of [CLS]-tokens. See `CLSToken` for
        examples of usage.
        Note:
            Under the hood, the `torch.Tensor.expand` method is applied to the
            underlying :code:`weight` parameter, so gradients will be propagated as
            expected.
        Args:
            leading_dimensions: the additional new dimensions
        Returns:
            tensor of the shape :code:`(*leading_dimensions, len(self.weight))`
        """
        if not leading_dimensions:
            return self.weight
        new_dims = (1,) * (len(leading_dimensions) - 1)
        return self.weight.view(*new_dims, -1).expand(*leading_dimensions, -1)

    def forward(self, x: Tensor) -> Tensor:
        """Append self **to the end** of each item in the batch (see `CLSToken`)."""
        return torch.cat([x, self.expand(len(x), 1)], dim=1)

class MultiHeadAttention(nn.Module):
    """
    Multihead Attention (self-/cross-) with optional 'linear' attention.
    To learn more about Multihead Attention, see [devlin2018bert]. See the implementation
    of `Transformer` and the examples below to learn how to use the compression technique
    from [wang2020linformer] to speed up the module when the number of tokens is large.
    """

    def __init__(
            self, 
            d_token: int, 
            n_heads: int, 
            dropout: float, 
            bias: bool,
            initialization: str,
        ) -> None:
        """
        Args:
            d_token: the token size. Must be a multiple of :code:`n_heads`.
            n_heads: the number of heads. If greater than 1, then the module will have
                an addition output layer (so called "mixing" layer).
            dropout: dropout rate for the attention map. The dropout is applied to
                *probabilities* and do not affect logits.
            bias: if `True`, then input (and output, if presented) layers also have bias.
                `True` is a reasonable default choice.
            initialization: initialization for input projection layers. Must be one of
                :code:`['kaiming', 'xavier']`. `kaiming` is a reasonable default choice.
        Raises:
            AssertionError: if requirements for the inputs are not met.
        """
        super().__init__()

        self.W_q = nn.Linear(d_token, d_token, bias)
        self.W_k = nn.Linear(d_token, d_token, bias)
        self.W_v = nn.Linear(d_token, d_token, bias)
        self.W_out = nn.Linear(d_token, d_token, bias) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None


        for m in [self.W_q, self.W_k, self.W_v]:
            # the "xavier" branch tries to follow torch.nn.MultiheadAttention;
            # the second condition checks if W_v plays the role of W_out; the latter one
            # is initialized with Kaiming in torch
            if initialization == 'xavier' and (
                m is not self.W_v or self.W_out is not None
            ):
                # gain is needed since W_qkv is represented with 3 separate layers (it
                # implies different fan_out)
                nn.init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if self.W_out is not None:
            nn.init.zeros_(self.W_out.bias)

    def _reshape(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )
    
    def forward(self, x_q: Tensor, x_kv: Tensor) -> Tensor:
        """
        Perform the forward pass.
        Args:
            x_q: query tokens
            x_kv: key-value tokens
            key_compression: Linformer-style compression for keys
            value_compression: Linformer-style compression for values
        Returns:
            (tokens, attention_stats)
        """
        
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        #reshape for multihead
        q = self._reshape(q)
        k = self._reshape(k)

        attention_logits = q @ k.transpose(1, 2) / math.sqrt(d_head_key)

        attention_probs = F.softmax(attention_logits, dim=-1)
        if self.dropout is not None:
            attention_probs = self.dropout(attention_probs)

        x = attention_probs @ self._reshape(v)

        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )

        if self.W_out is not None:
            x = self.W_out(x)
        return x

class FFN(nn.Module):
    """The Feed-Forward Network module used in every `Transformer` block."""

    def __init__(
            self, 
            d_token: int, 
            d_hidden: int, 
            bias_first: bool = True, 
            bias_second: bool = True, 
            dropout: float = 0
            ):
        super().__init__()

        self.linear_first = nn.Linear(d_token, d_hidden, bias_first)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear_second = nn.Linear(d_hidden, d_token, bias_second)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear_first(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear_second(x)
        return x

class Head(nn.Module):
    """The final module of the `Transformer` that performs BERT-like inference."""

    def __init__(
            self, 
            d_in: int, 
            d_out: int,  
            bias: bool = True
            ):
        super().__init__()
        self.normalization = nn.LayerNorm(d_in)
        self.activation = nn.ReLU(d_in)
        self.linear = nn.Linear(d_in, d_out, bias)

    def forward(self, x: Tensor) -> Tensor:
        x = x[:, -1]
        x = self.normalization(x)
        x = self.activation(x)
        x = self.linear(x)
        return x

class Transformer(nn.Module):
    """
    Transformer with extra features.
    This module is the backbone of `FTTransformer`.
    """

    def __init__(
            self,
            d_token: int,
            n_blocks: int,
            attention_n_heads: int,
            attention_dropout: float,
            attention_initialization: str,
            ffn_d_hidden: int,
            ffn_dropout: float,
            residual_dropout: float,
            d_out: int,
        ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList([])
        
        for _ in range(n_blocks):
            layer = nn.ModuleDict(
                {
                    'attention': MultiHeadAttention(
                        d_token=d_token,
                        n_heads=attention_n_heads,
                        dropout=attention_dropout,
                        bias=True,
                        initialization=attention_initialization,
                    ),
                    'ffn': FFN(
                        d_token=d_token,
                        d_hidden=ffn_d_hidden,
                        bias_first=True,
                        bias_second=True,
                        dropout=ffn_dropout,
                    ),
                    'attention_residual_dropout': nn.Dropout(residual_dropout),
                    'ffn_residual_dropout': nn.Dropout(residual_dropout),
                    'attention_layerNorm': nn.LayerNorm(d_token),
                    'ffn_layerNorm': nn.LayerNorm(d_token),
                    'output': nn.Identity(),  # for hooks-based introspection
                }
            )

            self.blocks.append(layer)
        
        self.head = Head(
            d_in=d_token,
            d_out=d_out,
            bias=True,
        )
    
    def forward(self, x: Tensor) -> Tensor:
        
        for layer_idx, layer in enumerate(self.blocks):

            #Attention block
            x_residual_attention = x
            x = layer['attention_layerNorm'](x)
            if layer_idx + 1 == len(self.blocks):
                x = layer['attention'](x[:, [-1], :], x)
            else:
                x = layer['attention'](x, x)
            x = layer['attention_residual_dropout'](x)
            if layer_idx + 1 == len(self.blocks):
                x_residual_attention = x_residual_attention[:, [-1]]
            x += x_residual_attention

            #FFN block
            x_residual_ffn = x
            x = layer['ffn_layerNorm'](x)
            x = layer['ffn'](x)
            x = layer['ffn_residual_dropout'](x)
            x += x_residual_ffn
            x = layer['output'](x)
        
        x = self.head(x)

        return x


class FTTransformer(nn.Module):
    """
    The FT-Transformer model.
    Transforms features to tokens with `FeatureTokenizer` and applies `Transformer` [vaswani2017attention]
    to the tokens. The following illustration provides a high-level overview of the
    architecture.
    """

    def __init__(self, feature_tokenizer: FeatureTokenizer, transformer: Transformer) -> None:
        super().__init__()

        self.feature_tokenizer = feature_tokenizer
        self.cls_token = CLSToken(feature_tokenizer.d_token, feature_tokenizer.initialization)
        self.transformer = transformer

    @classmethod
    def default(cls, n_num_features, cat_cardinalities, d_out):
        """
        This baseline you need only put count of numerical and categorial features
        """

        return FTTransformer(

            FeatureTokenizer(
                n_num_features=n_num_features,
                cat_cardinalities=cat_cardinalities, 
                d_token=192,
                ),

            Transformer(
                d_token = 192,
                n_blocks = 3,
                attention_n_heads = 8,
                attention_dropout = 0.2,
                attention_initialization = 'xavier',
                ffn_d_hidden = 768,
                ffn_dropout = 0.1,
                residual_dropout = 0,
                d_out = d_out,
                )
            )

    @classmethod
    def custom(cls, n_num_features, cat_cardinalities, d_token, n_blocks, attention_n_heads, attention_dropout, attention_initialization, ffn_d_hidden, ffn_dropout, residual_dropout, d_out):
        return FTTransformer(

            FeatureTokenizer(
                n_num_features=n_num_features,
                cat_cardinalities=cat_cardinalities,
                d_token=d_token,
            ),

            Transformer(
                d_token=d_token,
                n_blocks=n_blocks,
                attention_n_heads=attention_n_heads,
                attention_dropout=attention_dropout,
                attention_initialization=attention_initialization,
                ffn_d_hidden=ffn_d_hidden,
                ffn_dropout=ffn_dropout,
                residual_dropout=residual_dropout,
                d_out=d_out,
            )
        )

    def forward(self, x_num: Optional[Tensor], x_cat: Optional[Tensor] = None) -> Tensor:
        x = self.feature_tokenizer(x_num, x_cat)
        x = self.cls_token(x)
        x = self.transformer(x)
        return x



def example_numerical_and_categorial():
    """
    Example with numerical and categorial features
    """
    model = FTTransformer(
        feature_tokenizer=FeatureTokenizer(
            n_num_features=23,
            cat_cardinalities=[3, 5, 10], 
            d_token=192,
            ),

        transformer=Transformer(
            d_token = 192,
            n_blocks = 3,
            attention_n_heads = 8,
            attention_dropout = 0.2,
            attention_initialization = 'xavier',
            ffn_d_hidden = 768,
            ffn_dropout = 0.1,
            residual_dropout = 0,
            d_out = 1,
        )
    )
    summary(model, input_size=((10, 23), (10, 3)), dtypes=[torch.float, torch.long], 
            depth=6, col_names=["input_size", "output_size", "num_params", "kernel_size"])

def example_numerical():
    """
    Example with only numerical features
    """
    model = FTTransformer(
        feature_tokenizer=FeatureTokenizer(
            n_num_features=5,
            cat_cardinalities=[], 
            d_token=192,
            ),

        transformer=Transformer(
            d_token = 192,
            n_blocks = 3,
            attention_n_heads = 8,
            attention_dropout = 0.2,
            attention_initialization = 'xavier',
            ffn_d_hidden = 768,
            ffn_dropout = 0.1,
            residual_dropout = 0,
            d_out = 25,
        )
    )
    summary(model, input_size=(10, 5), dtypes=[torch.float], 
            depth=6, col_names=["input_size", "output_size", "num_params", "kernel_size"])

def example_default():
    """
    This baseline you need only put count of numerical and categorial features
    """
    model = FTTransformer.default(n_num_features=5, cat_cardinalities=[])
    summary(model, (10, 5), depth=6, col_names=["input_size", "output_size", "num_params", "kernel_size"])

if __name__ == "__main__":
    example_default()
