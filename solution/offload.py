"""Offloading saved activations to pinned host memory, overlapped with compute.

The MLP's gate and up projections, which SwiGLU's backward needs, are about half of what a layer
keeps for backward. Inside `Offloader.scope()` every large tensor autograd saves is copied to
pinned host memory on a side stream as soon as it is produced (its GPU memory is released when the
copy finishes) and copied back during backward, one layer ahead of where backward is, so the copy
overlaps the GEMMs instead of waiting for them. The values are bit-identical; only where they
wait between forward and backward changes. Offloading in our own code is allowed by the task.
"""

import contextlib

import torch

MIN_BYTES = 1 << 20  # tensors smaller than this stay on the GPU
_STREAMS = {}


def _side_stream(device):
    if device not in _STREAMS:
        _STREAMS[device] = torch.cuda.Stream(device=device)
    return _STREAMS[device]


class _Saved:
    __slots__ = ("host", "stored", "device", "fetched", "fetched_event", "index", "unpacked")


class Offloader:
    """One per training step: saved tensors in pack order, fetched back in reverse order."""

    def __init__(self, device):
        self.stream = _side_stream(device)
        self.items = []

    @contextlib.contextmanager
    def scope(self):
        with torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack):
            yield

    def _pack(self, tensor):
        if not tensor.is_cuda or tensor.numel() * tensor.element_size() < MIN_BYTES:
            return tensor
        item = _Saved()
        item.host = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
        item.device, item.fetched, item.fetched_event = tensor.device, None, None
        item.unpacked = None
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            item.host.copy_(tensor, non_blocking=True)
            item.stored = torch.cuda.Event()
            item.stored.record()
        tensor.record_stream(self.stream)  # its memory is reused only after the copy is done
        item.index = len(self.items)
        self.items.append(item)
        return item

    def _fetch(self, item):
        if item.fetched is not None or item.host is None:
            return
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(item.stored)
            item.fetched = torch.empty(item.host.shape, dtype=item.host.dtype, device=item.device)
            item.fetched.copy_(item.host, non_blocking=True)
            item.fetched_event = torch.cuda.Event()
            item.fetched_event.record()

    def _unpack(self, item):
        if isinstance(item, torch.Tensor):
            return item
        if item.unpacked is not None:  # saved_tensors read again
            return item.unpacked
        self._fetch(item)
        for previous in self.items[max(0, item.index - 2):item.index]:  # backward runs in reverse
            self._fetch(previous)
        compute = torch.cuda.current_stream()
        compute.wait_event(item.fetched_event)
        tensor = item.fetched
        tensor.record_stream(compute)
        item.unpacked, item.fetched, item.host = tensor, None, None  # host block back to the pool
        return tensor
