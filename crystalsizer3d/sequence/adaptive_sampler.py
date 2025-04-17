from typing import List

import torch
from torch import Tensor

from crystalsizer3d.util.ema import EMA


class AdaptiveSampler:
    def __init__(
            self,
            sequence_length: int,
            ema_decay: float = 0.9,
            ema_init: float = 1.0,
            always_include_idxs: List[int] | None = None,
            always_include_min: int = 1,
    ):
        self.sequence_length = sequence_length
        self.emas = [EMA(val=ema_init, decay=ema_decay) for _ in range(sequence_length)]
        self.always_include_idxs = always_include_idxs if always_include_idxs is not None else []
        self.always_include_min = always_include_min
        if 0 < len(self.always_include_idxs) < always_include_min:
            raise ValueError(
                f'Number of always include indices ({len(self.always_include_idxs)}) '
                f'cannot be smaller than always_include_min ({always_include_min})')

    @property
    def errors(self) -> Tensor:
        return torch.tensor([ema.val for ema in self.emas])

    def update_errors(self, frame_indices: List[int], errors: Tensor):
        """
        Update exponential moving averages.
        """
        for i, frame_idx in enumerate(frame_indices):
            self.emas[frame_idx](errors[i].item())

    def get_sampling_probabilities(self) -> Tensor:
        """
        Convert errors to probabilities by normalising to sum to 1.
        """
        errors = self.errors
        probabilities = errors / errors.sum()
        return probabilities

    def sample_frames(self, batch_size: int) -> Tensor:
        """
        Sample frames according to the adaptive probabilities.
        """
        probabilities = self.get_sampling_probabilities()

        # Ensure always include frames
        if self.always_include_min > 0 and self.always_include_idxs:
            # Uniformly sample always_include_min from the always_include_idxs
            always_include_frame_idxs = [
                self.always_include_idxs[i]
                for i in torch.randperm(len(self.always_include_idxs))[:self.always_include_min]
            ]

            # Set the sampling probabilities to 0 for the always include frames
            probabilities[always_include_frame_idxs] = 0
            batch_size = batch_size - self.always_include_min

        # Sample frame idxs
        frame_idxs = torch.multinomial(probabilities, batch_size, replacement=False)

        # Add the always_include frames
        if self.always_include_min > 0 and self.always_include_idxs:
            frame_idxs = torch.cat((frame_idxs, torch.tensor(always_include_frame_idxs)))

        return frame_idxs
