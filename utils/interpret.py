import torch
from torch import nn

MAX_STD = 0.5
MIN_STD = 0.


def monte_carlo_dropout(
    model, regime='loader', loader=None, tensors=None, repetitions=20
):

    if regime != 'loader' and regime != 'tensors':
        raise ValueError("Choose regime from {'loader', 'tensors'}")

    # Activate dropout layers while keeping other rest in eval mode.
    def enable_dropout(m):
        if type(m) == nn.Dropout:
            m.train()

    model.eval()
    model.apply(enable_dropout)

    if regime == 'loader':

        # Error handling
        if not isinstance(
            loader.sampler, torch.utils.data.sampler.SequentialSampler
        ):
            raise AttributeError(
                'Data loader does not use sequential sampling. Consider set'
                'ting shuffle=False when instantiating the data loader.'
            )

        # Run over all batches in the loader

        def call_fn():
            preds = []
            for ind, inputs in enumerate(loader):
                # inputs is a tuple with the last element being the labels
                # outs can be a n-tuple returned by the model
                outs = model(*inputs[:-1])
                preds.append(outs[0] if isinstance(outs, tuple) else outs)

            return torch.cat(preds)

    elif regime == 'tensors':

        if (
            not isinstance(tensors, tuple)
            and not isinstance(tensors, torch.Tensor)
        ):
            raise ValueError('Tensor needs to either tuple or torch.Tensor')

        inputs = tensors if isinstance(tensors, tuple) else (tensors, )

        def call_fn():
            outs = model(*inputs)
            return outs[0] if isinstance(outs, tuple) else outs

    with torch.no_grad():
        predictions = [
            torch.unsqueeze(call_fn(), -1) for _ in range(repetitions)
        ]
    predictions = torch.cat(predictions, dim=-1)

    # Scale confidences to [0, 1]
    confidences = -1 * (
        (predictions.std(dim=-1) - MIN_STD) / (MAX_STD - MIN_STD)
    ) + 1

    model.eval()

    return confidences, torch.mean(predictions, -1)


def test_time_augmentation(
    model,
    regime='loader',
    loader=None,
    tensors=None,
    repetitions=20,
    augmenter=None,
    tensors_to_augment=None
):

    if regime != 'loader' and regime != 'tensors':
        raise ValueError("Choose regime from {'loader', 'tensors'}")

    model.eval()

    if regime == 'loader':

        # Error handling
        if not isinstance(
            loader.sampler, torch.utils.data.sampler.SequentialSampler
        ):
            raise AttributeError(
                'Data loader does not use sequential sampling. Consider set'
                'ting shuffle=False when instantiating the data loader.'
            )

        # Run over all batches in the loader

        def call_fn():
            preds = []
            for ind, inputs in enumerate(loader):
                # inputs is a tuple with the last element being the labels
                # outs can be a n-tuple returned by the model
                outs = model(*inputs[:-1])
                preds.append(outs[0] if isinstance(outs, tuple) else outs)

            return torch.cat(preds)

    elif regime == 'tensors':

        if (
            not isinstance(tensors, tuple)
            and not isinstance(tensors, torch.Tensor)
        ):
            raise ValueError('Tensor needs to either tuple or torch.Tensor')
        if (
            not isinstance(tensors_to_augment, list)
            and not isinstance(tensors_to_augment, int)
        ):
            raise ValueError('tensors_to_augment needs to be list or int')

        # Convert input to common formats (tuples and lists)
        tensors_to_augment = (
            [tensors_to_augment]
            if isinstance(tensors_to_augment, int) else tensors_to_augment
        )
        inputs = tensors if isinstance(tensors, tuple) else (tensors, )
        aug_fns = augmenter if isinstance(augmenter, tuple) else (augmenter, )

        # Error handling
        if not len(aug_fns) == len(tensors_to_augment):
            raise ValueError(
                'Provide one augmenter for each tensor you want to augment.'
            )
        if max(tensors_to_augment) > len(inputs):
            raise ValueError(
                'tensors_to_augment should be indexes to the tensors used for '
                f'augmentation. {max(tensors_to_augment)} is larger than '
                f'length of inputs ({len(inputs)}).'
            )

        def call_fn():
            # Perform augmentation on all designated functions
            augmented_inputs = [
                aug_fns[tensors_to_augment[tensors_to_augment == ind]](tensor)
                if ind in tensors_to_augment else tensor
                for ind, tensor in enumerate(tensors)
            ]
            outs = model(*augmented_inputs)
            return outs[0] if isinstance(outs, tuple) else outs

    with torch.no_grad():
        predictions = [
            torch.unsqueeze(call_fn(), -1) for _ in range(repetitions)
        ]
    predictions = torch.cat(predictions, dim=-1)

    # Scale confidences to [0, 1]
    confidences = -1 * (
        (predictions.std(dim=-1) - MIN_STD) / (MAX_STD - MIN_STD)
    ) + 1

    return torch.clamp(confidences, min=0), torch.mean(predictions, -1)
