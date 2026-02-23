from abc import ABCMeta, abstractmethod
from functools import lru_cache
from typing import List, Optional, Tuple, Type, Union

import math
import torch
from torch import Tensor

from alphagen_qlib.stock_data import StockData, FeatureType


class OutOfDataRangeError(IndexError):
    pass


class Expression(metaclass=ABCMeta):
    @abstractmethod
    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor: ...

    def __repr__(self) -> str: return str(self)

    def __add__(self, other: Union["Expression", float]) -> "Add":
        if isinstance(other, Expression):
            return Add(self, other)
        else:
            return Add(self, Constant(other))

    def __radd__(self, other: float) -> "Add": return Add(Constant(other), self)

    def __sub__(self, other: Union["Expression", float]) -> "Sub":
        if isinstance(other, Expression):
            return Sub(self, other)
        else:
            return Sub(self, Constant(other))

    def __rsub__(self, other: float) -> "Sub": return Sub(Constant(other), self)

    def __mul__(self, other: Union["Expression", float]) -> "Mul":
        if isinstance(other, Expression):
            return Mul(self, other)
        else:
            return Mul(self, Constant(other))

    def __rmul__(self, other: float) -> "Mul": return Mul(Constant(other), self)

    def __truediv__(self, other: Union["Expression", float]) -> "Div":
        if isinstance(other, Expression):
            return Div(self, other)
        else:
            return Div(self, Constant(other))

    def __rtruediv__(self, other: float) -> "Div": return Div(Constant(other), self)

    def __pow__(self, other: Union["Expression", float]) -> "Pow":
        if isinstance(other, Expression):
            return Pow(self, other)
        else:
            return Pow(self, Constant(other))

    def __rpow__(self, other: float) -> "Pow": return Pow(Constant(other), self)

    def __pos__(self) -> "Expression": return self
    def __neg__(self) -> "Sub": return Sub(Constant(0), self)
    def __abs__(self) -> "Abs": return Abs(self)

    @property
    def is_featured(self): raise NotImplementedError


class Feature(Expression):
    def __init__(self, feature: FeatureType) -> None:
        self._feature = feature

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        assert period.step == 1 or period.step is None
        if (period.start < -data.max_backtrack_days or
                period.stop - 1 > data.max_future_days):
            raise OutOfDataRangeError()
        start = period.start + data.max_backtrack_days
        stop = period.stop + data.max_backtrack_days + data.n_days - 1
        return data.data[start:stop, int(self._feature), :]

    def __str__(self) -> str: return '$' + self._feature.name.lower()

    @property
    def is_featured(self): return True


class Constant(Expression):
    def __init__(self, value: float) -> None:
        self._value = value

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        assert period.step == 1 or period.step is None
        if (period.start < -data.max_backtrack_days or
                period.stop - 1 > data.max_future_days):
            raise OutOfDataRangeError()
        device = data.data.device
        dtype = data.data.dtype
        days = period.stop - period.start - 1 + data.n_days
        return torch.full(size=(days, data.n_stocks),
                          fill_value=self._value, dtype=dtype, device=device)

    # def __str__(self) -> str: return f'Constant({str(self._value)})'
    def __str__(self) -> str: return f'{str(self._value)}'

    @property
    def is_featured(self): return False


class DeltaTime(Expression):
    # This is not something that should be in the final expression
    # It is only here for simplicity in the implementation of the tree builder
    def __init__(self, delta_time: int) -> None:
        self._delta_time = delta_time

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        assert False, "Should not call evaluate on delta time"

    def __str__(self) -> str: return str(self._delta_time)

    @property
    def is_featured(self): return False


# Operator base classes

class Operator(Expression):
    @classmethod
    @abstractmethod
    def n_args(cls) -> int: ...

    @classmethod
    @abstractmethod
    def category_type(cls) -> Type['Operator']: ...


class UnaryOperator(Operator):
    def __init__(self, operand: Union[Expression, float]) -> None:
        self._operand = operand if isinstance(operand, Expression) else Constant(operand)

    @classmethod
    def n_args(cls) -> int: return 1

    @classmethod
    def category_type(cls) -> Type['Operator']: return UnaryOperator

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        return self._apply(self._operand.evaluate(data, period))

    @abstractmethod
    def _apply(self, operand: Tensor) -> Tensor: ...

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._operand})"

    @property
    def is_featured(self): return self._operand.is_featured


class BinaryOperator(Operator):
    def __init__(self, lhs: Union[Expression, float], rhs: Union[Expression, float]) -> None:
        self._lhs = lhs if isinstance(lhs, Expression) else Constant(lhs)
        self._rhs = rhs if isinstance(rhs, Expression) else Constant(rhs)

    @classmethod
    def n_args(cls) -> int: return 2

    @classmethod
    def category_type(cls) -> Type['Operator']: return BinaryOperator

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        return self._apply(self._lhs.evaluate(data, period), self._rhs.evaluate(data, period))

    @abstractmethod
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: ...

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._lhs},{self._rhs})"

    @property
    def is_featured(self): return self._lhs.is_featured or self._rhs.is_featured


class RollingOperator(Operator):
    def __init__(self, operand: Union[Expression, float], delta_time: Union[int, DeltaTime]) -> None:
        self._operand = operand if isinstance(operand, Expression) else Constant(operand)
        if isinstance(delta_time, DeltaTime):
            delta_time = delta_time._delta_time
        self._delta_time = delta_time

    @classmethod
    def n_args(cls) -> int: return 2

    @classmethod
    def category_type(cls) -> Type['Operator']: return RollingOperator

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        start = period.start - self._delta_time + 1
        stop = period.stop
        # L: period length (requested time window length)
        # W: window length (dt for rolling)
        # S: stock count
        values = self._operand.evaluate(data, slice(start, stop))   # (L+W-1, S)
        values = values.unfold(0, self._delta_time, 1)              # (L, S, W)
        return self._apply(values)                                  # (L, S)

    @abstractmethod
    def _apply(self, operand: Tensor) -> Tensor: ...

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._operand},{self._delta_time})"

    @property
    def is_featured(self): return self._operand.is_featured


class PairRollingOperator(Operator):
    def __init__(self,
                 lhs: Expression, rhs: Expression,
                 delta_time: Union[int, DeltaTime]) -> None:
        self._lhs = lhs if isinstance(lhs, Expression) else Constant(lhs)
        self._rhs = rhs if isinstance(rhs, Expression) else Constant(rhs)
        if isinstance(delta_time, DeltaTime):
            delta_time = delta_time._delta_time
        self._delta_time = delta_time

    @classmethod
    def n_args(cls) -> int: return 3

    @classmethod
    def category_type(cls) -> Type['Operator']: return PairRollingOperator

    def _unfold_one(self, expr: Expression,
                    data: StockData, period: slice = slice(0, 1)) -> Tensor:
        start = period.start - self._delta_time + 1
        stop = period.stop
        # L: period length (requested time window length)
        # W: window length (dt for rolling)
        # S: stock count
        values = expr.evaluate(data, slice(start, stop))            # (L+W-1, S)
        return values.unfold(0, self._delta_time, 1)                # (L, S, W)

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        lhs = self._unfold_one(self._lhs, data, period)
        rhs = self._unfold_one(self._rhs, data, period)
        return self._apply(lhs, rhs)                                # (L, S)

    @abstractmethod
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: ...

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._lhs},{self._rhs},{self._delta_time})"

    @property
    def is_featured(self): return self._lhs.is_featured or self._rhs.is_featured


# Operator implementations

class Abs(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.abs()

    
class SLog1p(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.sign() * operand.abs().log1p()
    
class Inv(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return 1/operand


class Sign(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.sign()


class Log(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.log()


class Rank(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        nan_mask = operand.isnan()
        n = (~nan_mask).sum(dim=1, keepdim=True)
        rank = operand.argsort().argsort() / n
        rank[nan_mask] = torch.nan
        return rank


class Add(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs + rhs


class Sub(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs - rhs


class Mul(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs * rhs


class Div(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs / rhs


class Pow(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs ** rhs


class Greater(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs.max(rhs)

    @property
    def is_featured(self):
        return self._lhs.is_featured or self._rhs.is_featured


class Less(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs.min(rhs)

    @property
    def is_featured(self):
        return self._lhs.is_featured or self._rhs.is_featured

class GetGreater(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return torch.maximum(lhs, rhs)

    @property
    def is_featured(self):
        return self._lhs.is_featured or self._rhs.is_featured

class GetLess(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return torch.minimum(lhs, rhs)

    @property
    def is_featured(self):
        return self._lhs.is_featured or self._rhs.is_featured

class Ref(RollingOperator):
    # Ref is not *really* a rolling operator, in that other rolling operators
    # deal with the values in (-dt, 0], while Ref only deal with the values
    # at -dt. Nonetheless, it should be classified as rolling since it modifies
    # the time window.

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        start = period.start - self._delta_time
        stop = period.stop - self._delta_time
        return self._operand.evaluate(data, slice(start, stop))

    def _apply(self, operand: Tensor) -> Tensor:
        # This is just for fulfilling the RollingOperator interface
        ...


class TsMean(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.mean(dim=-1)


class TsSum(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.sum(dim=-1)


class TsStd(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.std(dim=-1)

class TsIr(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.mean(dim=-1) / operand.std(dim=-1)

class TsMinMaxDiff(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.max(dim=-1)[0] - operand.min(dim=-1)[0]
class TsMaxDiff(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand[..., -1] - operand.max(dim=-1)[0]
class TsMinDiff(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand[..., -1] - operand.min(dim=-1)[0]

class TsVar(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.var(dim=-1)


class TsSkew(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        # skew = m3 / m2^(3/2)
        central = operand - operand.mean(dim=-1, keepdim=True)
        m3 = (central ** 3).mean(dim=-1)
        m2 = (central ** 2).mean(dim=-1)
        return m3 / m2 ** 1.5


class TsKurt(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        # kurt = m4 / var^2 - 3
        central = operand - operand.mean(dim=-1, keepdim=True)
        m4 = (central ** 4).mean(dim=-1)
        var = operand.var(dim=-1)
        return m4 / var ** 2 - 3


class TsMax(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.max(dim=-1)[0]


class TsMin(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.min(dim=-1)[0]


class TsMed(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.median(dim=-1)[0]


class TsMad(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        central = operand - operand.mean(dim=-1, keepdim=True)
        return central.abs().mean(dim=-1)


class TsRank(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        n = operand.shape[-1]
        last = operand[:, :, -1, None]
        left = (last < operand).count_nonzero(dim=-1)
        right = (last <= operand).count_nonzero(dim=-1)
        result = (right + left + (right > left)) / (2 * n)
        return result


class TsDelta(RollingOperator):
    # Delta is not *really* a rolling operator, in that other rolling operators
    # deal with the values in (-dt, 0], while Delta only deal with the values
    # at -dt and 0. Nonetheless, it should be classified as rolling since it
    # modifies the time window.

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        start = period.start - self._delta_time
        stop = period.stop
        values = self._operand.evaluate(data, slice(start, stop))
        return values[self._delta_time:] - values[:-self._delta_time]

    def _apply(self, operand: Tensor) -> Tensor:
        # This is just for fulfilling the RollingOperator interface
        ...


class TsDiv(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        n = operand.shape[-1]
        return operand[:, :, -1] / operand.mean(dim=-1)


class TsPctChange(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return (operand[:, :, -1] - operand[:, :, 0]) / operand[:, :, 0]


class TsWMA(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        n = operand.shape[-1]
        weights = torch.arange(n, dtype=operand.dtype, device=operand.device)
        weights /= weights.sum()
        return (weights * operand).sum(dim=-1)


class TsEMA(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        n = operand.shape[-1]
        alpha = 1 - 2 / (1 + n)
        power = torch.arange(n, 0, -1, dtype=operand.dtype, device=operand.device)
        weights = alpha ** power
        weights /= weights.sum()
        return (weights * operand).sum(dim=-1)


class TsCov(PairRollingOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        n = lhs.shape[-1]
        clhs = lhs - lhs.mean(dim=-1, keepdim=True)
        crhs = rhs - rhs.mean(dim=-1, keepdim=True)
        return (clhs * crhs).sum(dim=-1) / (n - 1)


class TsCorr(PairRollingOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        clhs = lhs - lhs.mean(dim=-1, keepdim=True)
        crhs = rhs - rhs.mean(dim=-1, keepdim=True)
        ncov = (clhs * crhs).sum(dim=-1)
        nlvar = (clhs ** 2).sum(dim=-1)
        nrvar = (crhs ** 2).sum(dim=-1)
        stdmul = (nlvar * nrvar).sqrt()
        stdmul[(nlvar < 1e-6) | (nrvar < 1e-6)] = 1
        return ncov / stdmul

Operators: List[Type[Expression]] = [
    # Unary
    Abs, SLog1p, Inv, Sign, Log, Rank,
    # Binary
    Add, Sub, Mul, Div, Pow, Greater, Less, GetGreater, GetLess,
    # Rolling
    Ref, TsMean, TsSum, TsStd, TsIr, TsMinMaxDiff, TsMaxDiff, TsMinDiff, TsVar, TsSkew, TsKurt, TsMax, TsMin,
    TsMed, TsMad, TsRank, TsDelta, TsDiv, TsPctChange, TsWMA, TsEMA,
    # Pair rolling
    TsCov, TsCorr
]

def expression_edit_distance(expression_a: Expression, expression_b: Expression) -> int:
    """
    Compute the minimum number of edit operations required to turn expression_a into expression_b.

    The cost model follows the rules defined for expression modifications:
    * Changing a feature or constant leaf to another leaf costs 1.
    * Switching between operators of the same category (unary, binary, rolling, pair rolling)
      while keeping operands costs 1.
    * Adding or removing unary / rolling operators costs 1.
    * Adding or removing binary / pair rolling operators costs 1 plus the cost of building
      the additional operand from scratch (token length of that operand).
    * Building a fresh expression costs its token length.
    """

    commutative_types: Tuple[Type[Expression], ...] = (Add, Mul, TsCov, TsCorr)

    def _is_leaf(expr: Expression) -> bool:
        return isinstance(expr, (Feature, Constant))

    def _is_commutative(expr: Expression) -> bool:
        return isinstance(expr, commutative_types)

    def _swap_penalty(expr: Expression) -> int:
        return 0 if _is_commutative(expr) else 1


    def _operand(expr: Expression) -> Expression:
        if isinstance(expr, (UnaryOperator, RollingOperator)):
            return expr._operand  # type: ignore[attr-defined]
        raise TypeError(f"Unsupported operand extraction for {type(expr).__name__}")

    def _lhs_rhs(expr: Expression) -> Tuple[Expression, Expression]:
        if isinstance(expr, (BinaryOperator, PairRollingOperator)):
            return expr._lhs, expr._rhs  # type: ignore[attr-defined]
        raise TypeError(f"Unsupported lhs/rhs extraction for {type(expr).__name__}")

    @lru_cache(maxsize=None)
    def _length(expr: Optional[Expression]) -> int:
        if expr is None:
            return 0
        if isinstance(expr, (Feature, Constant)):
            return 1
        if isinstance(expr, PairRollingOperator):
            return 2 + _length(expr._lhs) + _length(expr._rhs)
        if isinstance(expr, RollingOperator):
            return 2 + _length(expr._operand)
        if isinstance(expr, BinaryOperator):
            return 1 + _length(expr._lhs) + _length(expr._rhs)
        if isinstance(expr, UnaryOperator):
            return 1 + _length(expr._operand)
        raise TypeError(f"Unsupported expression type for length: {type(expr).__name__}")

    def _align_children(lhs_a: Expression, rhs_a: Expression,
                        lhs_b: Expression, rhs_b: Expression,
                        swap_penalty_a: int, swap_penalty_b: int) -> int:
        direct_cost = _distance(lhs_a, lhs_b) + _distance(rhs_a, rhs_b)
        swap_penalty = 0 if (swap_penalty_a == 0 and swap_penalty_b == 0) else 1
        swap_cost = _distance(lhs_a, rhs_b) + _distance(rhs_a, lhs_b) + swap_penalty
        return min(direct_cost, swap_cost)

    @lru_cache(maxsize=None)
    def _distance(expr_a: Optional[Expression], expr_b: Optional[Expression]) -> int:
        if expr_a is expr_b:
            return 0
        if expr_a is None:
            return _length(expr_b)
        if expr_b is None:
            return _length(expr_a)

        if isinstance(expr_a, Feature) and isinstance(expr_b, Feature):
            return 0 if expr_a._feature == expr_b._feature else 1
        if isinstance(expr_a, Constant) and isinstance(expr_b, Constant):
            return 0 if math.isclose(expr_a._value, expr_b._value,
                                     rel_tol=1e-12, abs_tol=1e-12) else 1
        if (_is_leaf(expr_a) and _is_leaf(expr_b) and
                not isinstance(expr_a, type(expr_b))):
            return 1

        candidates: List[int] = []

        if isinstance(expr_a, UnaryOperator) and isinstance(expr_b, UnaryOperator):
            base = 0 if type(expr_a) is type(expr_b) else 1
            candidates.append(base + _distance(expr_a._operand, expr_b._operand))

        if isinstance(expr_a, RollingOperator) and isinstance(expr_b, RollingOperator):
            base = 0 if type(expr_a) is type(expr_b) else 1
            delta_cost = 0
            # Rolling delta_time differences (except Ref) do not affect the distance.
            if isinstance(expr_a, Ref) and isinstance(expr_b, Ref):
                delta_cost = 0 if expr_a._delta_time == expr_b._delta_time else 1
            candidates.append(base + delta_cost + _distance(expr_a._operand, expr_b._operand))

        if isinstance(expr_a, BinaryOperator) and isinstance(expr_b, BinaryOperator):
            base = 0 if type(expr_a) is type(expr_b) else 1
            child_cost = _align_children(
                expr_a._lhs, expr_a._rhs,
                expr_b._lhs, expr_b._rhs,
                _swap_penalty(expr_a),
                _swap_penalty(expr_b),
            )
            candidates.append(base + child_cost)

        if isinstance(expr_a, PairRollingOperator) and isinstance(expr_b, PairRollingOperator):
            base = 0 if type(expr_a) is type(expr_b) else 1
            child_cost = _align_children(
                expr_a._lhs, expr_a._rhs,
                expr_b._lhs, expr_b._rhs,
                _swap_penalty(expr_a),
                _swap_penalty(expr_b),
            )
            candidates.append(base + child_cost)

        if ((isinstance(expr_a, UnaryOperator) and isinstance(expr_b, RollingOperator)) or
                (isinstance(expr_a, RollingOperator) and isinstance(expr_b, UnaryOperator))):
            operand_cost = _distance(_operand(expr_a), _operand(expr_b))
            candidates.append(1 + operand_cost)
        if ((isinstance(expr_a, BinaryOperator) and isinstance(expr_b, PairRollingOperator)) or
                (isinstance(expr_a, PairRollingOperator) and isinstance(expr_b, BinaryOperator))):
            lhs_a, rhs_a = _lhs_rhs(expr_a)
            lhs_b, rhs_b = _lhs_rhs(expr_b)
            child_cost = _align_children(
                lhs_a, rhs_a,
                lhs_b, rhs_b,
                _swap_penalty(expr_a),
                _swap_penalty(expr_b),
            )
            candidates.append(1 + child_cost)
        if isinstance(expr_a, UnaryOperator):
            candidates.append(1 + _distance(expr_a._operand, expr_b))
        if isinstance(expr_a, RollingOperator):
            candidates.append(1 + _distance(expr_a._operand, expr_b))
        if isinstance(expr_a, BinaryOperator):
            candidates.append(1 + _length(expr_a._rhs) + _distance(expr_a._lhs, expr_b))
            candidates.append(1 + _length(expr_a._lhs) + _distance(expr_a._rhs, expr_b))
        if isinstance(expr_a, PairRollingOperator):
            candidates.append(1 + _length(expr_a._rhs) + _distance(expr_a._lhs, expr_b))
            candidates.append(1 + _length(expr_a._lhs) + _distance(expr_a._rhs, expr_b))

        if isinstance(expr_b, UnaryOperator):
            candidates.append(1 + _distance(expr_a, expr_b._operand))
        if isinstance(expr_b, RollingOperator):
            candidates.append(1 + _distance(expr_a, expr_b._operand))
        if isinstance(expr_b, BinaryOperator):
            candidates.append(1 + _distance(expr_a, expr_b._lhs) + _length(expr_b._rhs))
            candidates.append(1 + _distance(expr_a, expr_b._rhs) + _length(expr_b._lhs))
        if isinstance(expr_b, PairRollingOperator):
            candidates.append(1 + _distance(expr_a, expr_b._lhs) + _length(expr_b._rhs))
            candidates.append(1 + _distance(expr_a, expr_b._rhs) + _length(expr_b._lhs))

        # Fallback: drop to rebuilding both expressions from scratch.
        candidates.append(_length(expr_a) + _length(expr_b))
        return min(candidates)

    return _distance(expression_a, expression_b)
