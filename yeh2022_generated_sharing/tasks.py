"""Synthetic benchmarks from Yeh et al. (AISTATS 2022).

This module intentionally contains only data generation and analytic sharing
oracles.  It does *not* depend on the existing ``pattern`` experiments, so a
learner may use it with either a directly optimised assignment matrix or a
generator of assignment matrices.

The paper specifies the targets and split sizes, but leaves a few details
implicit (notably the exact cross-correlation boundary convention and whether
``N(0, 0.1)`` denotes variance or standard deviation).  We make the choices
explicit in the public specs below rather than silently treating them as part
of an algorithmic result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


Device = str | torch.device


@dataclass(frozen=True, slots=True)
class Split:
    """A vectorised, immutable train/validation/test split.

    ``x`` has leading dimension equal to the number of examples.  ``y`` has
    the corresponding leading dimension and is a scalar target for the sum
    benchmark and a signal target for the other two benchmarks.
    """

    x: torch.Tensor
    y: torch.Tensor

    def __post_init__(self) -> None:
        if self.x.ndim < 2 or self.y.ndim < 1 or self.x.size(0) != self.y.size(0):
            raise ValueError("x and y must have compatible non-empty batch dimensions")

    @property
    def size(self) -> int:
        return int(self.x.size(0))


@dataclass(frozen=True, slots=True)
class Splits:
    """The fixed three-way split used by a benchmark."""

    train: Split
    validation: Split
    test: Split


@dataclass(frozen=True, slots=True)
class Benchmark:
    """Samples plus a categorical parameter-sharing oracle.

    ``oracle_categories`` uses zero for an inactive coefficient and positive
    integers for tied coefficients.  The numerical values of positive labels
    are arbitrary; only equality of labels defines sharing.  ``oracle_weight``
    is supplied when the target has an exact linear ground-truth map.
    """

    splits: Splits
    oracle_categories: torch.Tensor
    oracle_weight: torch.Tensor | None
    name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_groups(self) -> int:
        labels = self.oracle_categories[self.oracle_categories > 0]
        return int(labels.max().item()) if labels.numel() else 0


@dataclass(frozen=True, slots=True)
class SumNumbersSpec:
    """Configuration for Sec. 5.2's sum-of-numbers benchmark.

    The paper uses sequences of numbers uniformly drawn from ``{1, ..., 10}``,
    train/validation/test sizes 100/150/100000, and describes additive uniform
    noise as ``[0.5, 0.5]``.  The latter interval is degenerate as printed, so
    ``label_noise_half_width=0.5`` implements the natural symmetric reading
    ``[-0.5, 0.5]``; set it to zero for noiseless targets.
    Test targets are noiseless by default, matching the benchmark protocol.

    Paper input is textual tokens followed by a learned embedding.  This data
    utility exposes the token IDs/numbers directly, leaving embeddings to the
    model implementation.
    """

    sequence_length: int = 10
    train_size: int = 100
    validation_size: int = 150
    test_size: int = 100_000
    number_low: int = 1
    number_high: int = 10
    label_noise_half_width: float = 0.5
    noiseless_test: bool = True
    alternating: bool = False
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        _validate_sizes(self.train_size, self.validation_size, self.test_size)
        if self.sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        if self.number_low > self.number_high:
            raise ValueError("number_low must not exceed number_high")
        if self.label_noise_half_width < 0:
            raise ValueError("label_noise_half_width must be non-negative")


@dataclass(frozen=True, slots=True)
class CrossCorrelationSpec:
    """Configuration for Sec. 5.3's 1D cross-correlation benchmark.

    We use *valid* cross-correlation: for input length ``K`` and kernel length
    ``G``, the target has length ``K-G+1`` and no samples are padded.  Thus
    ``y[k] = sum_j x[k+j] g[j]`` exactly matches Eq. (27) wherever it is
    defined.  The article discusses boundary padding qualitatively but does
    not state its exact convention; choosing valid correlation keeps the
    resulting Toeplitz oracle unambiguous.

    ``noise_std`` interprets the paper's ``N(0, 0.1)`` as a standard deviation
    (not variance).  Test targets are noiseless by default, as stated in the
    paper.
    """

    input_length: int = 15
    kernel_length: int = 5
    train_size: int = 50
    validation_size: int = 100
    test_size: int = 10_000
    noise_std: float = 0.1
    noiseless_test: bool = True
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        _validate_sizes(self.train_size, self.validation_size, self.test_size)
        if self.kernel_length < 1 or self.kernel_length > self.input_length:
            raise ValueError("kernel_length must lie in [1, input_length]")
        if self.noise_std < 0:
            raise ValueError("noise_std must be non-negative")

    @property
    def output_length(self) -> int:
        return self.input_length - self.kernel_length + 1


@dataclass(frozen=True, slots=True)
class UnitStepDenoisingSpec:
    """Configuration for Appendix D.1's step-signal denoising benchmark.

    A clean signal is ``s * 1[k >= t] + b`` with the paper's uniform ranges.
    The article writes ``t ~ unif{0, K}``; this implementation includes both
    endpoints, so ``t=K`` is a constant signal.  The desired linear map is not
    supplied analytically--the oracle is only its unconstrained Toeplitz
    sharing topology.  This is appropriate because the finite-data optimal
    denoiser depends on the signal/noise distribution.
    """

    signal_length: int = 15
    train_size: int = 50
    validation_size: int = 100
    test_size: int = 10_000
    scale_low: float = 1.0
    scale_high: float = 50.0
    bias_low: float = -5.0
    bias_high: float = 5.0
    noise_std: float = 1.0
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        _validate_sizes(self.train_size, self.validation_size, self.test_size)
        if self.signal_length < 1:
            raise ValueError("signal_length must be positive")
        if self.scale_low > self.scale_high or self.bias_low > self.bias_high:
            raise ValueError("uniform lower bounds must not exceed upper bounds")
        if self.noise_std < 0:
            raise ValueError("noise_std must be non-negative")


def _validate_sizes(*sizes: int) -> None:
    if any(not isinstance(size, int) or size < 1 for size in sizes):
        raise ValueError("all split sizes must be positive integers")


def _generator(seed: int, device: Device) -> torch.Generator:
    """Make a local generator without mutating PyTorch's global RNG state."""
    return torch.Generator(device=torch.device(device)).manual_seed(int(seed))


def _split_generators(seed: int, device: Device) -> tuple[torch.Generator, torch.Generator, torch.Generator]:
    # Different fixed offsets make a split stable when another split's size is
    # changed, a useful property for repeatable hyperparameter studies.
    return tuple(_generator(seed + offset, device) for offset in (0, 1_000_003, 2_000_003))  # type: ignore[return-value]


def sum_oracle_categories(sequence_length: int, *, alternating: bool = False, device: Device = "cpu") -> torch.Tensor:
    """Return position-wise oracle sharing labels for the sum task.

    For the alternating task, zero-indexed even positions have coefficient
    ``+1`` and odd positions coefficient ``-1`` (the convention in Eq. 64),
    so they form two sharing groups.  There are no inactive parameters here.
    """
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    if not alternating:
        return torch.ones(sequence_length, dtype=torch.long, device=device)
    return (torch.arange(sequence_length, device=device) % 2 + 1).to(torch.long)


def make_sum_numbers(spec: SumNumbersSpec = SumNumbersSpec(), *, seed: int = 0, device: Device = "cpu") -> Benchmark:
    """Generate standard or alternating sum-of-numbers splits and oracle."""
    device = torch.device(device)
    signs = torch.ones(spec.sequence_length, dtype=spec.dtype, device=device)
    if spec.alternating:
        signs[1::2] = -1

    def sample(n: int, generator: torch.Generator, *, noisy: bool) -> Split:
        x = torch.randint(
            spec.number_low,
            spec.number_high + 1,
            (n, spec.sequence_length),
            generator=generator,
            dtype=torch.int64,
            device=device,
        ).to(spec.dtype)
        y = x @ signs
        if noisy and spec.label_noise_half_width:
            y = y + (2.0 * torch.rand(n, generator=generator, device=device, dtype=spec.dtype) - 1.0) * spec.label_noise_half_width
        return Split(x=x, y=y)

    train_g, validation_g, test_g = _split_generators(seed, device)
    return Benchmark(
        splits=Splits(
            sample(spec.train_size, train_g, noisy=True),
            sample(spec.validation_size, validation_g, noisy=True),
            sample(spec.test_size, test_g, noisy=not spec.noiseless_test),
        ),
        oracle_categories=sum_oracle_categories(spec.sequence_length, alternating=spec.alternating, device=device),
        oracle_weight=signs,
        name="alternating_sum_of_numbers" if spec.alternating else "sum_of_numbers",
        metadata={
            "sequence_length": spec.sequence_length,
            "alternating": spec.alternating,
            "label_noise_half_width": spec.label_noise_half_width,
            "noiseless_test": spec.noiseless_test,
        },
    )


def valid_cross_correlation_weights(input_length: int, kernel: torch.Tensor) -> torch.Tensor:
    """Return the ``[K-G+1, K]`` valid-correlation Toeplitz weight matrix."""
    if kernel.ndim != 1 or kernel.numel() < 1 or kernel.numel() > input_length:
        raise ValueError("kernel must be a non-empty vector no longer than input_length")
    output_length = input_length - kernel.numel() + 1
    weight = torch.zeros(output_length, input_length, dtype=kernel.dtype, device=kernel.device)
    starts = torch.arange(output_length, device=kernel.device)[:, None]
    offsets = torch.arange(kernel.numel(), device=kernel.device)[None, :]
    weight[starts, starts + offsets] = kernel
    return weight


def valid_cross_correlation_categories(input_length: int, kernel_length: int, *, device: Device = "cpu") -> torch.Tensor:
    """Return Toeplitz sharing labels, with zero for invalid/inactive entries."""
    if kernel_length < 1 or kernel_length > input_length:
        raise ValueError("kernel_length must lie in [1, input_length]")
    output_length = input_length - kernel_length + 1
    categories = torch.zeros((output_length, input_length), dtype=torch.long, device=device)
    starts = torch.arange(output_length, device=device)[:, None]
    offsets = torch.arange(kernel_length, device=device)[None, :]
    categories[starts, starts + offsets] = offsets + 1
    return categories


def make_cross_correlation(
    spec: CrossCorrelationSpec = CrossCorrelationSpec(), *, seed: int = 0, device: Device = "cpu"
) -> Benchmark:
    """Generate valid cross-correlation splits and its exact Toeplitz oracle."""
    device = torch.device(device)
    # Yeh et al. fix g to odd positive integers, increasing by two.
    kernel = torch.arange(1, 2 * spec.kernel_length, 2, dtype=spec.dtype, device=device)
    weight = valid_cross_correlation_weights(spec.input_length, kernel)

    def sample(n: int, generator: torch.Generator, *, noisy: bool) -> Split:
        x = torch.randn((n, spec.input_length), generator=generator, dtype=spec.dtype, device=device)
        y = x @ weight.T
        if noisy and spec.noise_std:
            y = y + torch.randn(y.shape, generator=generator, dtype=spec.dtype, device=device) * spec.noise_std
        return Split(x=x, y=y)

    train_g, validation_g, test_g = _split_generators(seed, device)
    return Benchmark(
        splits=Splits(
            sample(spec.train_size, train_g, noisy=True),
            sample(spec.validation_size, validation_g, noisy=True),
            sample(spec.test_size, test_g, noisy=not spec.noiseless_test),
        ),
        oracle_categories=valid_cross_correlation_categories(spec.input_length, spec.kernel_length, device=device),
        oracle_weight=weight,
        name="valid_cross_correlation",
        metadata={
            "input_length": spec.input_length,
            "output_length": spec.output_length,
            "kernel_length": spec.kernel_length,
            "noise_std": spec.noise_std,
            "noiseless_test": spec.noiseless_test,
        },
    )


def toeplitz_categories(size: int, *, device: Device = "cpu") -> torch.Tensor:
    """Return full square Toeplitz-sharing labels for a linear ``size`` map.

    A label is determined by the diagonal offset ``input - output``.  Labels
    run from one through ``2*size-1``; unlike valid correlation, every entry
    is active, so no zero/inactive group occurs.
    """
    if size < 1:
        raise ValueError("size must be positive")
    row = torch.arange(size, device=device)[:, None]
    column = torch.arange(size, device=device)[None, :]
    return (column - row + size).to(torch.long)


def make_unit_step_denoising(
    spec: UnitStepDenoisingSpec = UnitStepDenoisingSpec(), *, seed: int = 0, device: Device = "cpu"
) -> Benchmark:
    """Generate the appendix unit-step denoising splits and Toeplitz oracle."""
    device = torch.device(device)
    positions = torch.arange(spec.signal_length, device=device).view(1, -1)

    def sample(n: int, generator: torch.Generator) -> Split:
        scale = torch.empty((n, 1), dtype=spec.dtype, device=device).uniform_(spec.scale_low, spec.scale_high, generator=generator)
        bias = torch.empty((n, 1), dtype=spec.dtype, device=device).uniform_(spec.bias_low, spec.bias_high, generator=generator)
        # torch.randint upper bound is exclusive: K+1 implements t in {0, ..., K}.
        transition = torch.randint(0, spec.signal_length + 1, (n, 1), generator=generator, device=device)
        clean = scale * (positions >= transition).to(spec.dtype) + bias
        noisy = clean
        if spec.noise_std:
            noisy = clean + torch.randn(clean.shape, generator=generator, dtype=spec.dtype, device=device) * spec.noise_std
        return Split(x=noisy, y=clean)

    train_g, validation_g, test_g = _split_generators(seed, device)
    return Benchmark(
        splits=Splits(sample(spec.train_size, train_g), sample(spec.validation_size, validation_g), sample(spec.test_size, test_g)),
        oracle_categories=toeplitz_categories(spec.signal_length, device=device),
        oracle_weight=None,
        name="unit_step_denoising",
        metadata={
            "signal_length": spec.signal_length,
            "noise_std": spec.noise_std,
        },
    )


__all__ = [
    "Benchmark",
    "CrossCorrelationSpec",
    "Split",
    "Splits",
    "SumNumbersSpec",
    "UnitStepDenoisingSpec",
    "make_cross_correlation",
    "make_sum_numbers",
    "make_unit_step_denoising",
    "sum_oracle_categories",
    "toeplitz_categories",
    "valid_cross_correlation_categories",
    "valid_cross_correlation_weights",
]
