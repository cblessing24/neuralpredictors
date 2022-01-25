from typing import Optional
from unittest.mock import MagicMock
import pytest

from torch.nn.parameter import Parameter
from torch import Tensor
import torch

from neuralpredictors.layers.readouts.base import Readout, Reduction


class MyReadout(Readout):
    def __init__(self) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        self.bias = Parameter(torch.tensor([1.0, 2.0, 3.0]))  # type: ignore[attr-defined]


@pytest.fixture
def readout() -> MyReadout:
    return MyReadout()


def test_initialize_raises_not_implemented_error(readout: MyReadout) -> None:
    with pytest.raises(NotImplementedError):
        readout.initialize()


def test_regularizer_raises_not_implemented_error(readout: MyReadout) -> None:
    with pytest.raises(NotImplementedError):
        readout.regularizer()


def test_apply_reduction_method_calls_resolve_reduction_method(readout: MyReadout) -> None:
    readout.resolve_reduction_method = MagicMock(return_value="sum")  # type: ignore[assignment]
    readout.apply_reduction(torch.tensor(1.0), reduction="sum", average=False)  # type: ignore[attr-defined]
    readout.resolve_reduction_method.assert_called_once_with(reduction="sum", average=False)


@pytest.mark.parametrize(
    "reduction,expected", [("sum", torch.tensor(3.0)), ("mean", torch.tensor(1.5)), (None, torch.tensor([1.0, 2.0]))]  # type: ignore[attr-defined]
)
def test_correct_reduction_is_applied(readout: MyReadout, reduction: Reduction, expected: Tensor) -> None:
    readout.resolve_reduction_method = MagicMock(return_value=reduction)  # type: ignore[assignment]
    out = readout.apply_reduction(torch.tensor([1.0, 2.0]))  # type: ignore[attr-defined]
    assert torch.equal(out, expected)  # type: ignore[attr-defined]


def test_invalid_reduction_raises_value_error(readout: MyReadout) -> None:
    with pytest.raises(ValueError):
        readout.apply_reduction(torch.tensor(1.0), reduction="median")  # type: ignore[arg-type,attr-defined]


@pytest.mark.parametrize(
    "reduction,average,expected",
    [("sum", None, "sum"), ("mean", None, "mean"), (None, True, "mean"), (None, False, "sum"), (None, None, "mean")],
)
def test_resolve_reduction_method_returns_correct_reduction(
    readout: MyReadout, reduction: Reduction, average: Optional[bool], expected: Reduction
) -> None:
    assert readout.resolve_reduction_method(reduction, average) == expected


@pytest.mark.parametrize("reduction,average", [("sum", True), ("sum", False), ("mean", True), ("mean", False)])
def test_resolve_reduction_method_raises_value_error_for_invalid_argument_combinations(
    readout: MyReadout, reduction: Reduction, average: Optional[bool]
) -> None:
    with pytest.raises(ValueError):
        readout.resolve_reduction_method(reduction, average)


@pytest.mark.parametrize("average", [True, False])
def test_resolve_reduction_method_raises_deprecation_warning_if_average_is_passed(
    readout: MyReadout, average: Optional[bool]
) -> None:
    with pytest.deprecated_call():
        readout.resolve_reduction_method(average=average)


@pytest.mark.parametrize("feature_reg_weight,gamma_readout,expected", [(1.2, None, 1.2), (1.2, 3.5, 3.5)])
def test_resolve_deprecated_gamma_readout_returns_correct_value(
    readout: MyReadout, feature_reg_weight: float, gamma_readout: Optional[float], expected: float
) -> None:
    assert readout.resolve_deprecated_gamma_readout(feature_reg_weight, gamma_readout) == pytest.approx(expected)


@pytest.mark.parametrize("feature_reg_weight,gamma_readout", [(1.2, 3.5)])
def test_resolve_deprecated_gamma_readout_raises_deprecation_warning_if_gamma_readout_is_passed(
    readout: MyReadout, feature_reg_weight: float, gamma_readout: Optional[float]
) -> None:
    with pytest.deprecated_call():
        readout.resolve_deprecated_gamma_readout(feature_reg_weight, gamma_readout)


def test_if_initialize_bias_raises_warning_if_mean_activity_is_not_passed(readout: MyReadout) -> None:
    with pytest.warns(UserWarning):
        readout.initialize_bias()


def test_if_bias_is_initialized_with_0_if_mean_activity_is_passed(readout: MyReadout) -> None:
    readout.initialize_bias()
    assert readout.bias.data == pytest.approx(torch.tensor([0.0, 0.0, 0.0]))  # type: ignore[attr-defined]


def test_if_bias_is_initialized_with_mean_activity_if_passed(readout: MyReadout) -> None:
    readout.initialize_bias(torch.tensor([4.0, 5.0, 6.0]))  # type: ignore[attr-defined]
    assert readout.bias.data == pytest.approx(torch.tensor([4.0, 5.0, 6.0]))  # type: ignore[attr-defined]


def test_repr(readout: MyReadout) -> None:
    assert repr(readout) == "MyReadout() [MyReadout]\n"
