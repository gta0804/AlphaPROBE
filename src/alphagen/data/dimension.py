"""
Utilities for validating factor expressions and computing their dimensionality.

The implementation reuses the parsing utilities from :mod:`alphagen.data.tree`
to turn an infix string into the expression tree defined in
``alphagen.data.expression``.  Dimensionality is then evaluated recursively
according to the user-specified rules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union

from alphagen.data.expression import (
    Abs,
    Add,
    Constant,
    Div,
    Expression,
    Feature,
    Greater,
    Inv,
    Less,
    GetGreater,
    GetLess,
    Log,
    Mul,
    Pow,
    Rank,
    Ref,
    SLog1p,
    Sign,
    Sub,
    TsCov,
    TsCorr,
    TsDelta,
    TsDiv,
    TsEMA,
    TsIr,
    TsKurt,
    TsMad,
    TsMax,
    TsMaxDiff,
    TsMean,
    TsMed,
    TsMin,
    TsMinDiff,
    TsMinMaxDiff,
    TsPctChange,
    TsRank,
    TsSkew,
    TsStd,
    TsSum,
    TsVar,
    TsWMA,
)
from alphagen.data.tree import ExpressionParser, InvalidExpressionException

_TOLERANCE = 1e-9


class DimensionError(Exception):
    """Base class for all dimension related errors."""


class InvalidDimensionExpression(DimensionError):
    """Raised when the input expression cannot be parsed or is structurally invalid."""


class DimensionConstraintError(DimensionError):
    """Raised when a dimension constraint is violated while evaluating the expression."""


@dataclass(frozen=True)
class DimensionResult:
    """Container for a dimensionality value and the normalized expression string."""

    dimension: float
    normalized_expression: str


def _is_close(lhs: float, rhs: float) -> bool:
    return math.isclose(lhs, rhs, abs_tol=_TOLERANCE, rel_tol=_TOLERANCE)


def _require_zero(value: float, context: str) -> False:
    if not _is_close(value, 0.0):
        return False
    return True

def _require_equal(lhs: float, rhs: float, context: str) -> bool:
    if not _is_close(lhs, rhs):
        return False
        # raise DimensionConstraintError(
        #     f"{context} requires operands with the same dimension, got {lhs} and {rhs}."
        # )
    return True


class DimensionCalculator:
    """Computes the dimensionality for factor expressions."""

    def __init__(self) -> None:
        self._parser = ExpressionParser()

    def validate(self, expression: str) -> Expression:
        """
        Validate that an expression can be parsed.

        The process mirrors the conversion to Reverse Polish Notation used in
        :mod:`alphagen.data.tree`.  Any parsing or structural error is surfaced
        as :class:`InvalidDimensionExpression`.
        """
        try:
            parsed = self._parser.parse(expression)
        except (InvalidExpressionException, ValueError) as exc:
            raise InvalidDimensionExpression(str(exc)) from exc
        return parsed

    def dimension(self, expression: Union[str, Expression]) -> DimensionResult:
        """
        Compute the dimensionality of an expression.

        Parameters
        ----------
        expression:
            Either an expression string (which will be validated and parsed) or
            an already constructed :class:`Expression`.
        """
        if isinstance(expression, str):
            expr_obj = self.validate(expression)
        elif isinstance(expression, Expression):
            expr_obj = expression
        else:
            raise TypeError(f"Unsupported expression type: {type(expression)!r}")

        dim = self._dimension(expr_obj)
        return DimensionResult(dimension=dim, normalized_expression=str(expr_obj))

    # pylint: disable=too-many-return-statements,too-many-branches
    def _dimension(self, expr: Expression) -> float:
        if isinstance(expr, Feature):
            return 1.0

        if isinstance(expr, Constant):
            return 0.0

        if isinstance(expr, Abs):
            return self._dimension(expr._operand)  # type: ignore[attr-defined]

        if isinstance(expr, SLog1p):
            return self._dimension(expr._operand)  # type: ignore[attr-defined]

        if isinstance(expr, Log):
            operand_dim = self._dimension(expr._operand)  # type: ignore[attr-defined]
            return operand_dim

        if isinstance(expr, Sign):
            return 0.0

        if isinstance(expr, Inv):
            return -self._dimension(expr._operand)  # type: ignore[attr-defined]

        if isinstance(expr, Rank):
            # _require_zero(self._dimension(expr._operand), "Rank")  # type: ignore[attr-defined]
            return 0.0

        if isinstance(expr, Add):
            lhs = self._dimension(expr._lhs)  # type: ignore[attr-defined]
            rhs = self._dimension(expr._rhs)  # type: ignore[attr-defined]
            return max(lhs, rhs)

        if isinstance(expr, Sub):
            lhs = self._dimension(expr._lhs)  # type: ignore[attr-defined]
            rhs = self._dimension(expr._rhs)  # type: ignore[attr-defined]
            return max(lhs, rhs)

        if isinstance(expr, Mul):
            lhs = self._dimension(expr._lhs)  # type: ignore[attr-defined]
            rhs = self._dimension(expr._rhs)  # type: ignore[attr-defined]
            return lhs + rhs

        if isinstance(expr, Div):
            lhs = self._dimension(expr._lhs)  # type: ignore[attr-defined]
            rhs = self._dimension(expr._rhs)  # type: ignore[attr-defined]
            return lhs - rhs

        if isinstance(expr, Pow):
            base_dim = self._dimension(expr._lhs)  # type: ignore[attr-defined]
            exponent = expr._rhs  # type: ignore[attr-defined]
            if not isinstance(exponent, Constant):
                raise DimensionConstraintError("Pow exponent must be a constant scalar.")
            exponent_value = float(exponent._value)
            return base_dim * exponent_value

        if isinstance(expr, Greater):
            lhs = self._dimension(expr._lhs)  # type: ignore[attr-defined]
            rhs = self._dimension(expr._rhs)  # type: ignore[attr-defined]
            # if not isinstance(expr._lhs, Constant) and not isinstance(expr._rhs, Constant):
            #     _require_equal(lhs, rhs, "Greater")
            # _require_equal(lhs, rhs, "Greater")
            return  0

        if isinstance(expr, Less):
            lhs = self._dimension(expr._lhs)  # type: ignore[attr-defined]
            rhs = self._dimension(expr._rhs)  # type: ignore[attr-defined]
            # if not isinstance(expr._lhs, Constant) and not isinstance(expr._rhs, Constant):
            #     _require_equal(lhs, rhs, "Less")
            return 0
        
        if isinstance(expr, GetGreater):
            lhs = self._dimension(expr._lhs)  # type: ignore[attr-defined]
            rhs = self._dimension(expr._rhs)  # type: ignore[attr-defined]
            # if not isinstance(expr._lhs, Constant) and not isinstance(expr._rhs, Constant):
            #     _require_equal(lhs, rhs, "GetGreater")
            return  max(lhs, rhs)  # Return the dimension of the operands
        
        if isinstance(expr, GetLess):
            lhs = self._dimension(expr._lhs)  # type: ignore[attr-defined]
            rhs = self._dimension(expr._rhs)  # type: ignore[attr-defined]
            # if not isinstance(expr._lhs, Constant) and not isinstance(expr._rhs, Constant):
            #     _require_equal(lhs, rhs, "GetLess")
            return  max(lhs, rhs)  # Return the dimension of the operands

        if isinstance(expr, Ref):
            return self._dimension(expr._operand)  # type: ignore[attr-defined]

        if isinstance(expr, TsMean):
            return self._dimension(expr._operand)  # type: ignore[attr-defined]

        if isinstance(expr, TsSum):
            return self._dimension(expr._operand)  # type: ignore[attr-defined]

        if isinstance(expr, TsStd):
            return self._dimension(expr._operand)  # type: ignore[attr-defined]

        if isinstance(expr, TsIr):
            return 0.0

        if isinstance(expr, TsMinMaxDiff):
            return self._dimension(expr._operand)  # type: ignore[attr-defined]

        if isinstance(expr, TsMaxDiff):
            return self._dimension(expr._operand)  # type: ignore[attr-defined]

        if isinstance(expr, TsMinDiff):
            return self._dimension(expr._operand)  # type: ignore[attr-defined]

        if isinstance(expr, TsVar):
            operand_dim = self._dimension(expr._operand)  # type: ignore[attr-defined]
            return 2.0 * operand_dim

        if isinstance(expr, TsSkew):
            return 0.0

        if isinstance(expr, TsKurt):
            return 0.0

        if isinstance(expr, TsMax):
            return self._dimension(expr._operand)  # type: ignore[attr-defined]

        if isinstance(expr, TsMin):
            return self._dimension(expr._operand)  # type: ignore[attr-defined]

        if isinstance(expr, TsMed):
            return self._dimension(expr._operand)  # type: ignore[attr-defined]

        if isinstance(expr, TsMad):
            return self._dimension(expr._operand)  # type: ignore[attr-defined]

        if isinstance(expr, TsRank):
            return 0.0

        if isinstance(expr, TsDelta):
            return self._dimension(expr._operand)  # type: ignore[attr-defined]

        if isinstance(expr, TsDiv):
            return 0.0

        if isinstance(expr, TsPctChange):
            return 0.0

        if isinstance(expr, TsWMA):
            return self._dimension(expr._operand)  # type: ignore[attr-defined]

        if isinstance(expr, TsEMA):
            return self._dimension(expr._operand)  # type: ignore[attr-defined]

        if isinstance(expr, TsCov):
            lhs = self._dimension(expr._lhs)  # type: ignore[attr-defined]
            rhs = self._dimension(expr._rhs)  # type: ignore[attr-defined]
            # _require_equal(lhs, rhs, "TsCov")
            return 0.0

        if isinstance(expr, TsCorr):
            lhs = self._dimension(expr._lhs)  # type: ignore[attr-defined]
            rhs = self._dimension(expr._rhs)  # type: ignore[attr-defined]
            # _require_equal(lhs, rhs, "TsCorr")
            return 0.0

        raise DimensionError(f"Unsupported expression type: {type(expr).__name__}")


def compute_dimension(expression: Union[str, Expression]) -> DimensionResult:
    """
    Convenience wrapper around :class:`DimensionCalculator`.
    """
    calculator = DimensionCalculator()
    return calculator.dimension(expression)

