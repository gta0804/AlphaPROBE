
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Set

from alphagen.data.tree import ExpressionParser, InvalidExpressionException

from openai import OpenAI
from utils.prompt import PROMPT_COMPARE, PROMPT_HEAD

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

_UNARY_OPERATORS = (Abs, SLog1p, Inv, Sign, Log, Rank)
_BINARY_OPERATORS = (Add, Sub, Mul, Div, Pow, Greater, Less, GetGreater, GetLess)
_ROLLING_OPERATORS = (
    Ref,
    TsMean,
    TsSum,
    TsStd,
    TsIr,
    TsMinMaxDiff,
    TsMaxDiff,
    TsMinDiff,
    TsVar,
    TsSkew,
    TsKurt,
    TsMax,
    TsMin,
    TsMed,
    TsMad,
    TsRank,
    TsDelta,
    TsDiv,
    TsPctChange,
    TsWMA,
    TsEMA,
)
_PAIR_ROLLING_OPERATORS = (TsCov, TsCorr)

_TOKEN_KIND_ORDER = {
    'OP': 0,
    'DT': 1,
    'FEATURE': 2,
    'CONST': 3,
}


class ExpressionBSTError(Exception):
    """Base class for BST specific errors."""


class InvalidExpressionInput(ExpressionBSTError):
    """Raised when the supplied expression cannot be parsed."""

@dataclass(frozen=True)
class ExpressionPayload:
    expression: Expression
    length: int
    tokens: Tuple[str, ...]
    source: str


@dataclass
class ExpressionNode:
    payload: ExpressionPayload
    parent: Optional["ExpressionNode"] = None
    children: Set[str]  = field(default_factory=set)
    topic: Optional[str] = None
    description: Optional[str] = None
    depth: int = 0
    ic: float = 0.0
    icir: float = 0.0
    times: int = 0
    test_ic: float = 0.0
    test_icir: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "expression": self.payload.source,
            "length": self.payload.length,
            "tokens": list(self.payload.tokens),
            "topic": self.topic,
            "description": self.description,
            "depth": self.depth,
            "ic": self.ic,
            "icir": self.icir,
            "times": self.times,
            "test_ic": self.test_ic,
            "test_icir": self.test_icir,
            "children": list(self.children),
        }  

class ExpressionKnowledgeGraph():
    def __init__(self, depth_limit: int = 5, expression_size_limit: int = 50) -> None:
        self._nodes: Dict[str, ExpressionNode] = {}
        self._node_records: Dict[str, ExpressionNode] = {}
        # self._splitable_nodes: Set[str] = set()
        self._size = 0
        # self._optimization_targets:Set[str] = set()
        self._parser = ExpressionParser()
        self._depth_limit = depth_limit
        self._expression_size_limit = expression_size_limit
    

    def build_expression_node(self, expression: str, topic: str, explanation: str, length_threshold: int, ic: float = None, icir: float = None) -> Optional[ExpressionNode]:
        if self.search(expression.strip()) is not None:
            print(f"Expression {expression} already exists in the graph, not building.")
            return self.search(expression.strip())
        payload = self._build_payload(expression)
        if payload is None:
            print(f"Failed to build payload for expression {expression}.")
            return None
        if payload.length > length_threshold:
            print(f"Expression {expression} exceeds length threshold {length_threshold}, not building.")
            return None
        expression_node = ExpressionNode(payload=payload, parent=None, topic=topic, description=explanation, ic=ic if ic is not None else 0.0, icir=icir if icir is not None else 0.0)
        return expression_node
      
    def search(self, expression: str) -> Optional[ExpressionNode]:
        return self._nodes.get(expression.strip())
    
    def path_to_root(self, node: ExpressionNode) -> List[ExpressionNode]:
        path = []
        current = node
        while current is not None:
            path.append(current)
            current = current.parent
        path.reverse()
        return path
    
    # def insert_splitable_node(self, node: str):
    #     if node.strip() in self._splitable_nodes:
    #         print(f"Node {node} already in splitable nodes")
    #         return
    #     try:
    #         expression_rep = node.strip().replace("%d", "10")
    #         expr_obj = self._parser.parse(expression_rep)
    #     except (InvalidExpressionException, ValueError) as exc:
    #         print(f"error parsing expression in splitable {node}: {exc}")
    #         return
    #     self._splitable_nodes.add(node.strip())
    

    def insert(self, expression_node: ExpressionNode, parent_node: Optional[ExpressionNode]) -> bool:
        if self.search(expression_node.payload.source) is not None:
            print(f"Expression {expression_node.payload.source} already exists in the graph, not inserting.")
            return False
        if parent_node is not None:
            expression_node.parent = parent_node
            parent_node.children.add(expression_node.payload.source)
            expression_node.depth = parent_node.depth + 1
        self._node_records[expression_node.payload.source] = expression_node
        return True
    

    # def insert(self, expression: str, topic: str, explanation: str, ic: float, parent_node: ExpressionNode) -> bool:
    #     if self.search(expression) is not None:
    #         print(f"Expression {expression} already exists in the graph, not inserting.")
    #         return False
    #     payload = self._build_payload(expression)
    #     if payload.length > self._expression_size_limit:
    #         print(f"Expression {expression} exceeds size limit {self._expression_size_limit}, not inserting.")
    #         return False
    #     expression_node = ExpressionNode(payload=payload, parent=parent_node, topic=topic, description=explanation, depth=1 if parent_node is None else parent_node.depth + 1, ic=ic)
    #     self._nodes[expression.strip()] = expression_node
    #     self._size += 1
    #     expression_node.parent = parent_node
    #     # if expression_node.depth < self._depth_limit:
    #     #     self._optimization_targets.add(expression_node.payload.source)
    #     # if parent_node is not None:
    #     #     self._optimization_targets.discard(parent_node.payload.source)
    #     return True
     
    def _build_payload(self, expression: str) -> Optional[ExpressionPayload]:
        try:
            expression_rep = expression.strip().replace("%d", "10")
            expr_obj = self._parser.parse(expression_rep)
            source = expression
        except (InvalidExpressionException, ValueError) as exc:
            print(f"error parsing expression {expression}: {exc}")
            return None
            # raise InvalidExpressionInput(str(exc) + f" in expression: {expression}") from exc

        tokens = self._tokenize(expr_obj)
        return ExpressionPayload(
            expression=expr_obj,
            length=len(tokens),
            tokens=tokens,
            source=source.strip(),
        )

    def _tokenize(self, expr: Expression) -> Tuple[str, ...]:
        if isinstance(expr, Feature):
            return (f"FEATURE:{expr._feature.name}",)  # type: ignore[attr-defined]
        if isinstance(expr, Constant):
            return (f"CONST:{self._format_constant(expr._value)}",)  # type: ignore[attr-defined]
        if isinstance(expr, _UNARY_OPERATORS):
            operand = getattr(expr, "_operand")
            return (f"OP:{type(expr).__name__}",) + self._tokenize(operand)
        if isinstance(expr, _BINARY_OPERATORS):
            lhs = getattr(expr, "_lhs")
            rhs = getattr(expr, "_rhs")
            return (f"OP:{type(expr).__name__}",) + self._tokenize(lhs) + self._tokenize(rhs)
        if isinstance(expr, _ROLLING_OPERATORS):
            operand = getattr(expr, "_operand")
            delta = getattr(expr, "_delta_time")
            return (f"OP:{type(expr).__name__}", f"DT:{delta}") + self._tokenize(operand)
        if isinstance(expr, _PAIR_ROLLING_OPERATORS):
            lhs = getattr(expr, "_lhs")
            rhs = getattr(expr, "_rhs")
            delta = getattr(expr, "_delta_time")
            return (
                f"OP:{type(expr).__name__}",
                f"DT:{delta}",
            ) + self._tokenize(lhs) + self._tokenize(rhs)
        raise TypeError(f"Unsupported expression type for tokenization: {type(expr).__name__}")

    @staticmethod
    def _format_constant(value: float) -> str:
        return format(value, ".12g")

    def iter_nodes(self) -> List[ExpressionNode]:
        return list(self._node_records.values())
