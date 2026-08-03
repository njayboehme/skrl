from __future__ import annotations

from typing import Literal

import torch

from skrl.memories.torch import Memory
import numpy as np
import gymnasium
from skrl.utils.spaces.torch import compute_space_size

class RandomChunkedMemory(Memory):
    def __init__(
        self,
        *,
        memory_size: int,
        num_envs: int = 1,
        device: str | torch.device | None = None,
        export: bool = False,
        export_format: Literal["pt", "npz", "csv"] = "pt",
        export_directory: str = "",
        replacement: bool = False,
        action_chunk_size: int = 1,
    ) -> None:
        """Random sampling memory (sample a batch from memory randomly).

        :param memory_size: Maximum number of elements in the first dimension for each tensor.
        :param num_envs: Number of parallel environments.
        :param device: Data allocation and computation device. If not specified, the default device will be used.
        :param export: Export the memory to a file. If ``True``, the memory will be exported once it is filled
            and before the circular buffer starts to overwrite the oldest data.
        :param export_format: File format to export the memory.
            Supported formats: PyTorch (``"pt"``), NumPy (``"npz"``) or comma separated values (``"csv"``).
        :param export_directory: Directory where the memory files will be exported.
            If not specified, the agent's experiment directory will be used.
        :param replacement: Flag to indicate whether the sample is with or without replacement.
            Replacement implies that a value can be selected multiple times (the batch size is always guaranteed).
            Sampling without replacement will return a batch of maximum memory size if the memory size is less than
            the requested batch size.

        :raises ValueError: Unsupported export format.
        """
        super().__init__(
            memory_size=memory_size,
            num_envs=num_envs,
            device=device,
            export=export,
            export_format=export_format,
            export_directory=export_directory,
        )

        self._replacement = replacement
        self._tensor_inds = {}
        offset = num_envs * action_chunk_size
        self.chunk_inds = torch.concat([torch.arange(i*offset, (i+1)*offset).reshape(-1, num_envs).T for i in range(memory_size)]).to(device)

    def create_tensor(
            self,
            name: str,
            *,
            size: int | list[int] | gymnasium.Space | None,
            dtype: torch.dtype | None = None,
            keep_dimensions: bool = False,
            scale: int = 1,
        ) -> bool:
            """Create a new internal tensor in memory.
    
            The tensor will have a 3-dimensional with shape ``(memory_size, num_envs, data_size)``.
            The internal representation will use ``_tensor_<name>`` as the name of the class property.
    
            :param name: Tensor name (the name must follow the python PEP 8 style).
            :param size: Number of elements in the last dimension (effective data size).
                If a space is provided, the size will be computed as the number of elements occupied by the space.
            :param dtype: Data type. If not specified, the global default data type for PyTorch will be used.
            :param keep_dimensions: Whether to create a tensor with the original data dimensions.
                If enabled, only sequences of integers are supported as data ``size``.
            :param scale: A scale factor to increase the memory size for the given tensor. 
                            If a tensor has a scale, it will be tracked such that adding data and sampling will coincide with non-scaled tensors.
    
            :return: True if the tensor was created, otherwise False.
    
            :raises ValueError: A tensor with the same name exists already but its size and/or dtype is different.
            """
            # don't create a tensor for None
            if size is None:
                return False
            if keep_dimensions:
                if not isinstance(size, (tuple, list)):
                    raise ValueError("Only sequences of integers are supported as `size` when `keep_dimensions` is enabled")
            else:
                size = compute_space_size(size, occupied_size=True)
            # check dtype and size if the tensor exists already
            if name in self.tensors:
                tensor = self.tensors[name]
                if tensor.shape[-1] != size:
                    raise ValueError(f"Tensor size ({size}) doesn't match the existing one ({tensor.shape[-1]}): '{name}'")
                if dtype is not None and tensor.dtype != dtype:
                    raise ValueError(f"Tensor dtype ({dtype}) doesn't match the existing one ({tensor.dtype}): '{name}'")
                return False
            # create tensor (_tensor_<name>) and add it to the internal storage
            shape = (self.memory_size * scale, self.num_envs, *(size if keep_dimensions else [size]))
            setattr(self, f"_tensor_{name}", torch.zeros(shape, device=self.device, dtype=dtype))
            if scale != 1:
                self._tensor_inds[name] = 0
            # update internal variables
            self.tensors[name] = getattr(self, f"_tensor_{name}")
            self.tensors_view[name] = self.tensors[name].view((-1, *shape[2:]))
            # fill (float) tensors with NaN. This is useful for early misuse detection.
            for tensor in self.tensors.values():
                if torch.is_floating_point(tensor):
                    tensor.fill_(float("nan"))
            return True

    def sample(
        self, names: list[str], *, batch_size: int, mini_batches: int = 1, sequence_length: int = 1, chunk_size=1
    ) -> list[list[torch.Tensor]]:
        """Sample a batch from memory randomly.

        :param names: Tensors names from which to obtain the samples.
        :param batch_size: Number of elements to sample.
        :param mini_batches: Number of mini-batches to sample.
        :param sequence_length: Length of each sequence.

        :return: Sampled data from tensors sorted according to their position in the list of names.
            The sampled tensors will have the following shape: ``(batch_size, data_size)``.
        """
        # compute valid memory sizes
        size = len(self)
        if sequence_length > 1:
            sequence_indexes = torch.arange(0, self.num_envs * sequence_length, self.num_envs)
            size -= sequence_indexes[-1].item()
        # generate random indexes
        if self._replacement:
            indexes = torch.randint(0, size, (batch_size,))
        else:
            # details about the random sampling performance can be found here:
            # https://discuss.pytorch.org/t/torch-equivalent-of-numpy-random-choice/16146/19
            indexes = torch.randperm(size, dtype=torch.long)[:batch_size]
        # generate sequence indexes
        if sequence_length > 1:
            indexes = (sequence_indexes.repeat(indexes.shape[0], 1) + indexes.view(-1, 1)).view(-1)
        # sample by indexes
        self.sampling_indexes = indexes
        return self.sample_by_index(names=names, indexes=indexes, mini_batches=mini_batches, chunk_size=chunk_size)


    def sample_by_index(
            self, names: list[str], *, indexes: list | np.ndarray | torch.Tensor, mini_batches: int = 1, chunk_size: int = 1,
        ) -> list[list[torch.Tensor]]:
        """Sample data from memory according to their indexes.

        :param names: Tensors names from which to obtain the samples.
        :param indexes: Indexes used for sampling.
        :param mini_batches: Number of mini-batches to sample.

        :return: Sampled data from tensors sorted according to their position in the list of names.
            The sampled tensors will have the following shape: ``(number_of_indexes, data_size)``.
        """
        if mini_batches > 1:
            batches = np.array_split(indexes, mini_batches)
            return [
                [self.tensors_view[name][batch if self._tensor_inds.get(name, None) is None else self.chunk_inds[batch].flatten()] if name in self.tensors else None for name in names]
                for batch in batches
            ]
        return [[self.tensors_view[name][indexes] if name in self.tensors else None for name in names]]


    def add_samples(self, inc_memory_index=False, **tensors: dict[str, torch.Tensor]) -> None:
        """Add/store samples in memory.

        .. important::

            All tensors must have the same dimensions (2 dimensions) and shape: ``(current_num_envs, data_size)``.
            If the tensors have one dimension, it is assumed that ``current_num_envs`` is 1.

            No check is performed for compatibility of the shapes or for memory write overflow.

        According to the number of environments, the following behavior is performed:

        * ``current_num_envs = num_envs``: store samples and increment the memory index (1st index) by one.
        * ``current_num_envs < num_envs``: store samples and increment the environment index (2nd index)
            by the current number of environments.
        * ``current_num_envs > num_envs`` and ``num_envs = 1``: store multiple samples and increment the memory index
            (1st index) by the number of samples. If the number of samples is greater than the remaining memory size,
            the memory will be filled and circular buffer will overwrite the oldest data with the remaining samples.

        :param tensors: Sample data, as key-value arguments (keys: tensor names). Non-existing tensors will be skipped.

        :raises ValueError: No tensors were provided or the tensors have incompatible shapes.
        """
        if not tensors:
            raise ValueError(
                "There are no samples. Provide samples as key-value arguments, where keys are the tensor names"
            )

        # dimensions and shapes of the tensors (assume all tensors have the dimensions of the first tensor)
        tmp = tensors.get("observations", tensors[next(iter(tensors))])  # ask for observations first
        dim, shape = tmp.ndim, tmp.shape

        # multi environment (current_num_envs = num_envs)
        if dim == 2 and shape[0] == self.num_envs:
            for name, tensor in tensors.items():
                if name in self.tensors and tensor is not None:
                    cur_ind = self._tensor_inds.get(name, None)
                    if cur_ind is None:
                        self.tensors[name][self.memory_index].copy_(tensor)
                    else:
                        self.tensors[name][cur_ind].copy_(tensor)
                        self._tensor_inds[name] = (self._tensor_inds[name] + 1) % self.tensors[name].shape[0]
            # If adding to all points of data (not just intermediate data), increment the memory index
            if inc_memory_index:
                self.memory_index += 1
        # multi environment (current_num_envs < num_envs)
        elif dim == 2 and shape[0] < self.num_envs:
            for name, tensor in tensors.items():
                if name in self.tensors and tensor is not None:
                    self.tensors[name][self.memory_index, self.env_index : self.env_index + shape[0]].copy_(tensor)
            self.env_index += shape[0]
        # single environment - multi sample (num_envs = 1, current_num_envs > 1)
        elif dim == 2 and self.num_envs == 1:
            for name, tensor in tensors.items():
                if name in self.tensors and tensor is not None:
                    num_samples = min(shape[0], self.memory_size - self.memory_index)
                    # store the first n samples
                    self.tensors[name][self.memory_index : self.memory_index + num_samples].copy_(
                        tensor[:num_samples].unsqueeze(dim=1)
                    )
                    # store remaining samples
                    remaining_samples = shape[0] - num_samples
                    if remaining_samples > 0:
                        self.tensors[name][:remaining_samples].copy_(tensor[num_samples:].unsqueeze(dim=1))
                        self.memory_index = remaining_samples
                    else:
                        self.memory_index += num_samples
        # single environment (current_num_envs = 1, implicit)
        elif dim == 1:
            for name, tensor in tensors.items():
                if name in self.tensors and tensor is not None:
                    self.tensors[name][self.memory_index, self.env_index].copy_(tensor)
            self.env_index += 1
        else:
            raise ValueError(
                f"Expected shape (current_num_envs, data_size) where current_num_envs <= {self.num_envs}, got {shape}"
            )

        # update indexes and flags
        if self.env_index >= self.num_envs:
            self.env_index = 0
            self.memory_index += 1
        if self.memory_index >= self.memory_size:
            self.memory_index = 0
            self.filled = True

            # export tensors to file
            if self.export:
                self.save(directory=self.export_directory, format=self.export_format)
