"""Custom layers implementation."""
from collections import OrderedDict

import torch
import torch.nn as nn

from .utils import Squeeze, get_device, Temperature, Unsqueeze

DEVICE = get_device()


def dense_layer(
    input_size, hidden_size, act_fn=nn.ReLU(), batch_norm=False, dropout=0.0
):
    return nn.Sequential(
        OrderedDict(
            [
                ('projection', nn.Linear(input_size, hidden_size)),
                (
                    'batch_norm',
                    nn.BatchNorm1d(hidden_size)
                    if batch_norm else nn.Identity(),
                ),
                ('act_fn', act_fn),
                ('dropout', nn.Dropout(p=dropout)),
            ]
        )
    )


def dense_attention_layer(
    number_of_features: int,
    temperature: float = 1.0,
    dropout=0.0
) -> nn.Sequential:
    
    return nn.Sequential(
        OrderedDict(
            [
                ('dense', nn.Linear(number_of_features, number_of_features)),
                ('dropout', nn.Dropout(p=dropout)),
                ('temperature', Temperature(temperature)),
                ('softmax', nn.Softmax(dim=-1)),
            ]
        )
    )


def convolutional_layer(
    num_kernel,
    kernel_size,
    act_fn=nn.ReLU(),
    batch_norm=False,
    dropout=0.0,
    input_channels=1,
):

    return nn.Sequential(
        OrderedDict(
            [
                (
                    'convolve',
                    torch.nn.Conv2d(
                        input_channels,  # channel_in
                        num_kernel,  # channel_out
                        kernel_size,  # kernel_size
                        padding=[kernel_size[0] // 2,
                                 0],  # pad for valid conv.
                    ),
                ),
                ('squeeze', Squeeze()),
                ('act_fn', act_fn),
                ('dropout', nn.Dropout(p=dropout)),
                (
                    'batch_norm',
                    nn.BatchNorm1d(num_kernel)
                    if batch_norm else nn.Identity(),
                ),
            ]
        )
    )


class ContextAttentionLayer(nn.Module):

    def __init__(
        self,
        reference_hidden_size: int,
        reference_sequence_length: int,
        context_hidden_size: int,
        context_sequence_length: int = 1,
        attention_size: int = 16,
        individual_nonlinearity: type = nn.Sequential(),
        temperature: float = 1.0,
    ):
        super().__init__()

        self.reference_sequence_length = reference_sequence_length
        self.reference_hidden_size = reference_hidden_size
        self.context_sequence_length = context_sequence_length
        self.context_hidden_size = context_hidden_size
        self.attention_size = attention_size
        self.individual_nonlinearity = individual_nonlinearity
        self.temperature = temperature

        # Project the reference into the attention space
        self.reference_projection = nn.Sequential(
            OrderedDict(
                [
                    (
                        'projection',
                        nn.Linear(reference_hidden_size, attention_size),
                    ),
                    ('act_fn', individual_nonlinearity),
                ]
            )
        )  # yapf: disable

        # Project the context into the attention space
        self.context_projection = nn.Sequential(
            OrderedDict(
                [
                    (
                        'projection',
                        nn.Linear(context_hidden_size, attention_size),
                    ),
                    ('act_fn', individual_nonlinearity),
                ]
            )
        )  # yapf: disable

        # Optionally reduce the hidden size in context
        if context_sequence_length > 1:
            self.context_hidden_projection = nn.Sequential(
                OrderedDict(
                    [
                        (
                            'projection',
                            nn.Linear(
                                context_sequence_length,
                                reference_sequence_length,
                            ),
                        ),
                        ('act_fn', individual_nonlinearity),
                    ]
                )
            )  # yapf: disable
        else:
            self.context_hidden_projection = nn.Sequential()

        self.alpha_projection = nn.Sequential(
            OrderedDict(
                [
                    ('projection', nn.Linear(attention_size, 1, bias=False)),
                    ('squeeze', Squeeze()),
                    ('temperature', Temperature(self.temperature)),
                    ('softmax', nn.Softmax(dim=1)),
                ]
            )
        )

    def forward(
        self,
        reference: torch.Tensor,
        context: torch.Tensor,
        average_seq: bool = True
    ):
        
        assert len(reference.shape) == 3, 'Reference tensor needs to be 3D'
        assert len(context.shape) == 3, 'Context tensor needs to be 3D'

        reference_attention = self.reference_projection(reference)
        context_attention = self.context_hidden_projection(
            self.context_projection(context).permute(0, 2, 1)
        ).permute(0, 2, 1)
        alphas = self.alpha_projection(
            torch.tanh(reference_attention + context_attention)
        )

        output = reference * torch.unsqueeze(alphas, -1)
        output = torch.sum(output, 1) if average_seq else torch.squeeze(output)

        return output, alphas


def gene_projection(num_genes, attention_size, ind_nonlin=nn.Sequential()):
    return nn.Sequential(
        OrderedDict(
            [
                ('projection', nn.Linear(num_genes, attention_size)),
                ('act_fn', ind_nonlin),
                ('expand', Unsqueeze(1)),
            ]
        )
    ).to(DEVICE)


def smiles_projection(
    smiles_hidden_size, attention_size, ind_nonlin=nn.Sequential()
):
    return nn.Sequential(
        OrderedDict(
            [
                ('projection', nn.Linear(smiles_hidden_size, attention_size)),
                ('act_fn', ind_nonlin),
            ]
        )
    ).to(DEVICE)


def alpha_projection(attention_size):
    return nn.Sequential(
        OrderedDict(
            [
                ('projection', nn.Linear(attention_size, 1, bias=False)),
                ('squeeze', Squeeze()),
                ('softmax', nn.Softmax(dim=1)),
            ]
        )
    ).to(DEVICE)
