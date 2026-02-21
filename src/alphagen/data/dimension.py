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

expressions = [
           "TsPctChange(Sub(TsVar(Add(2.0,$low),40),Pow(Pow($volume,Abs(Add(-0.01,$vwap))),-5.0)),50)",
        "Add(0.5,Abs(TsDelta(TsMinDiff(TsPctChange($vwap,40),20),40)))",
        "Sub(TsPctChange(Add(-0.5,$high),20),-2.0)",
        "Div(-2.0,TsSum(Log($open),40))",
        "TsSkew(Mul(-0.01,Rank($close)),40)",
        "Div(Log(TsEMA(Add(Inv(Add(-5.0,$open)),-0.01),10)),-0.5)",
        "Div(Div(-0.5,$volume),-30.0)",
        "TsKurt(Sub(10.0,$open),20)",
        "Mul(Greater(-30.0,$close),TsDelta($open,30))",
        "Mul(Less(2.0,Add($vwap,30.0)),Rank(Div(TsSkew(Greater($open,Sign(Sign(Pow(-30.0,$high)))),50),30.0)))",
        "Sub(TsMax(Abs(Sign(TsMad($volume,10))),10),Sub(Add(2.0,Pow($low,-0.01)),1.0))",
        "TsSkew(Mul(-5.0,$high),20)",
        "Sub(10.0,Mul(Sub(Greater($close,-10.0),Pow(TsDiv($low,40),30.0)),-10.0))",
        "Div(-0.01,TsWMA(TsDiv($volume,30),40))",
        "TsCorr(Sub(-0.5,$open),$close,20)",
        "TsDiv($open,10)",
        "TsSkew($high,10)",
        "Mul(Inv(Div(TsSum($vwap,40),$high)),TsMax(TsMaxDiff(Mul($close,10.0),40),50))",
        "TsSum(TsKurt(Greater($open,$open),10),20)",
        "Add(2.0,Mul(TsCov($open,$volume,20),30.0))",
        "TsMaxDiff(Div(Add(-30.0,Abs(TsMed($high,40))),-1.0),50)",
        "Mul(Greater(2.0,TsSum($close,30)),$volume)",
        "TsSkew(Pow(Abs(TsMin($low,20)),-0.5),30)",
        "TsVar(Mul(5.0,SLog1p($volume)),10)",
        "TsPctChange(Less(TsDiv($open,40),Add(Abs(Inv(Abs($low))),-0.01)),40)",
        "Add(-10.0,TsIr($volume,50))",
        "Sub(Less(Add(TsMinDiff($volume,40),$open),-0.01),Div($close,Add(-30.0,$low)))",
        "TsKurt(Add($vwap,TsWMA($open,40)),50)",
        "TsMad(TsSum(Mul($high,Sign(TsMinMaxDiff($volume,20))),20),20)",
        "Mul(2.0,Div(Abs(Inv(TsMinMaxDiff(TsIr(Add($high,Div($volume,Greater($high,-1.0))),10),20))),-0.5))",
        "Mul(Less(Add(Greater(TsStd($open,50),-2.0),Div(Less($high,-0.01),$high)),0.01),$volume)",
        "Add(TsSkew(Sub(Greater(Less($vwap,-5.0),$open),TsKurt($volume,20)),50),-2.0)",
        "TsKurt(TsRank($volume,20),40)",
        "Sub(Div(TsEMA(Mul(Mul(Greater(Sub($close,Ref($close,1)),0.0),Pow(Sub(Div($close,Ref($close,1)),1.0),2.0)),$volume),5),TsEMA($volume,5)),Div(TsEMA(Mul(Mul(Greater(Sub(Ref($close,1),$close),0.0),Pow(Sub(Div(Ref($close,1),$close),1.0),2.0)),$volume),5),TsEMA($volume,5)))"
]
# expressions = [        "Mul(Greater(-30.0,$vwap),Mul(Div(TsDelta($low,20),-1.0),TsSkew(Div(TsMad($open,30),-30.0),30)))",
#         "Abs(Sub(1.0,TsWMA(TsVar($low,10),50)))",
#         "TsPctChange(Less(TsDiv($open,40),Add(Abs(Inv(Abs($low))),-0.01)),40)",
#         "Div(Div(Less(TsMed(Div(0.5,TsMin($volume,10)),20),Greater(Log(Add(-5.0,$open)),2.0)),$volume),-0.5)",
#         "TsEMA(TsSkew(TsMinDiff($vwap,40),10),30)",
#         "TsKurt($open,40)",
#         "TsMinDiff(TsMed(TsMinMaxDiff(TsMinDiff($high,20),10),10),20)",
#         "Div(-0.01,Div(TsStd(Log($volume),40),-10.0))",
#         "Div(Add(TsKurt($open,20),-30.0),0.01)",
#         "Sub(1.0,SLog1p(Div(Less(Greater($low,-5.0),$close),Less(Div($close,-1.0),$close))))",
#         "TsMad(TsIr($volume,40),30)",
#         "Mul(Less(2.0,Add($vwap,30.0)),Rank(Div(TsSkew(Greater($open,Sign(Sign(Pow(-30.0,$high)))),50),30.0)))",
#         "Sign(Add(-2.0,Pow($vwap,TsDelta($open,20))))",
#         "Div(30.0,TsCorr($high,Mul(-1.0,$open),20))",
#         "Div(-0.01,TsWMA(TsDiv($volume,30),40))",
#         "TsDiv($open,10)",
#         "Div(Less(Div(TsCorr($open,$volume,10),-5.0),$low),0.01)",
#         "Add(TsMin(Sub(2.0,TsRank($volume,20)),20),Pow(Div($high,$close),2.0))",
#         "TsIr(Div(Mul(-0.5,Add($vwap,$high)),$low),30)",
#         "TsPctChange(Add(-2.0,Log(TsMax(Log(Add($vwap,Add($close,-1.0))),30))),10)",
#         "Add(2.0,Mul(TsCov($open,$volume,20),30.0))",
#         "Sub(-10.0,TsMean(Div(TsPctChange($volume,10),Add(Sign(TsDiv($open,50)),Less($high,0.01))),40))",
#         "Mul(Greater(2.0,TsSum($close,30)),$volume)",
#         "Mul(30.0,Log(Add(TsDelta($volume,40),Div(TsStd($high,10),-1.0))))",
#         "Add(1.0,TsDiv(Rank($vwap),40))",
#         "Sub(2.0,TsKurt(Inv($volume),50))",
#         "TsPctChange(Pow(30.0,$close),50)",
#         "Less($low,Div($low,Mul(Abs(Mul($open,-0.01)),-0.01)))",
#         "TsMinDiff(Sign(Add($high,-2.0)),10)",
#         "Add(0.01,Log(Add(TsDiv(SLog1p($close),30),-1.0)))",
#         "TsSkew(Inv(Pow(Pow(Add(Sign($low),-0.01),Log($high)),30.0)),20)",
#         "Add(TsMinMaxDiff($open,30),Div(-10.0,Abs(Log(TsCorr(TsDelta(Log($volume),30),$volume,30)))))",
#         "TsIr(Sub(-2.0,Div(TsMinDiff($high,40),-0.01)),10)",
#         "Pow(Add(0.01,Mul(TsDelta($vwap,20),Div($vwap,Mul($open,1.0)))),-0.5)",
#         "Ref(TsMinMaxDiff(Log($close),20),50)",
#         "TsSkew(TsDelta(Pow(Div($close,0.5),-2.0),40),20)",
#         "Mul(Mul(Sub(Div(Ref($close, 5), $close), 1.0),Pow(Div(TsSum($volume, 5), Mul($volume, 5.0)), 0.5)),Div(Sub($high, $low), Ref($close, 1)))"]
for expr in expressions:
    res = compute_dimension(expr)
    print(f"{expr} dimension: {res.dimension}")

