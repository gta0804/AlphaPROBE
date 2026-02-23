"""
Binary search tree utilities for managing factor expressions.

Each node stores an :class:`Expression` instance and the tree enforces a
consistent ordering based on expression *complexity* (the total number of
operators and operands) and structural tokens. When two expressions share the
same complexity, a large language model is consulted to check for semantic
equivalence before the structural ordering is applied.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

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
from utils.llm import OpenAIModel

__all__ = ["ExpressionBST", "ExpressionNode", "ExpressionPayload"]

logger = logging.getLogger(__name__)

_DEFAULT_API_KEY = os.getenv("ALPHAGEN_BST_API_KEY", "")
_DEFAULT_BASE_URL = os.getenv("ALPHAGEN_BST_BASE_URL", "")
_DEFAULT_MODEL_NAME = os.getenv("ALPHAGEN_BST_MODEL", "")
_DEFAULT_STOP_WORDS: List[str] = []
_DEFAULT_MAX_NEW_TOKENS = int(os.getenv("ALPHAGEN_BST_MAX_NEW_TOKENS", "1000"))

_UNARY_OPERATORS = (Abs, SLog1p, Inv, Sign, Log, Rank)
_BINARY_OPERATORS = (Add, Sub, Mul, Div, Pow, Greater, Less)
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
    normalized: str
    length: int
    tokens: Tuple[str, ...]
    source: str


@dataclass
class ExpressionNode:
    payload: ExpressionPayload
    parent: Optional["ExpressionNode"] = None
    left: Optional["ExpressionNode"] = None
    right: Optional["ExpressionNode"] = None
    description: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "expression": self.payload.normalized,
            "length": self.payload.length,
            "tokens": list(self.payload.tokens),
            "description": self.description,
        }


class ExpressionBST:
    """Binary search tree ordered by expression length and structure."""

    def __init__(
        self,
        description: str,
        *,
        llm_client: Optional[OpenAI] = None,
        llm_model: Optional[OpenAIModel] = None,
        parser: Optional[ExpressionParser] = None,
    ) -> None:
        self.description = description
        self.root: Optional[ExpressionNode] = None
        self._size = 0
        self._parser = parser or ExpressionParser()
        self._llm_client = llm_client
        self._llm_model = llm_model
        self._equivalence_cache: Dict[Tuple[str, str], bool] = {}
        if (self._llm_client is None) != (self._llm_model is None):
            raise ValueError("llm_client and llm_model must be provided together.")

    def __len__(self) -> int:
        return self._size

    def is_empty(self) -> bool:
        return self.root is None

    def insert(self, expression: Union[str, Expression], description: Optional[str] = None) -> Tuple[ExpressionNode, bool]:
        payload = self._build_payload(expression)
        if self.root is None:
            self.root = ExpressionNode(payload=payload, description=description)
            self._size = 1
            return self.root, True

        current = self.root
        parent: Optional[ExpressionNode] = None
        last_cmp = 0
        while current is not None:
            parent = current
            last_cmp = self._compare_payloads(payload, current.payload)
            if last_cmp == 0:
                return current, False
            current = current.left if last_cmp < 0 else current.right

        assert parent is not None
        new_node = ExpressionNode(payload=payload, parent=parent, description=description)
        if last_cmp < 0:
            parent.left = new_node
        else:
            parent.right = new_node
        self._size += 1
        return new_node, True

    def search(self, expression: Union[str, Expression]) -> Optional[ExpressionNode]:
        payload = self._build_payload(expression)
        current = self.root
        while current is not None:
            cmp_result = self._compare_payloads(payload, current.payload)
            if cmp_result == 0:
                return current
            current = current.left if cmp_result < 0 else current.right
        return None

    def delete(self, expression: Union[str, Expression]) -> bool:
        node = self.search(expression)
        if node is None:
            return False
        self._delete_node(node)
        self._size -= 1
        return True

    def path_to(self, expression: Union[str, Expression]) -> List[str]:
        payload = self._build_payload(expression)
        path: List[str] = []
        current = self.root
        while current is not None:
            path.append(current.payload.normalized)
            cmp_result = self._compare_payloads(payload, current.payload)
            if cmp_result == 0:
                return path
            current = current.left if cmp_result < 0 else current.right
        return []

    def structured(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "size": self._size,
            "root": self._node_to_dict(self.root),
        }

    def format_tree(self) -> str:
        if self.root is None:
            return f"{self.description}: <empty>"
        lines = [f"{self.description} (size={self._size})", self.root.payload.normalized]
        children = [child for child in (self.root.left, self.root.right) if child is not None]
        for index, child in enumerate(children):
            child_is_last = index == len(children) - 1
            self._append_tree_lines(child, prefix="", is_last=child_is_last, lines=lines)
        return "\n".join(lines)

    def clear(self) -> None:
        self.root = None
        self._size = 0
        self._equivalence_cache.clear()

    def _delete_node(self, node: ExpressionNode) -> None:
        if node.left is None:
            self._transplant(node, node.right)
        elif node.right is None:
            self._transplant(node, node.left)
        else:
            successor = self._minimum(node.right)
            if successor.parent is not node:
                self._transplant(successor, successor.right)
                successor.right = node.right
                if successor.right is not None:
                    successor.right.parent = successor
            self._transplant(node, successor)
            successor.left = node.left
            if successor.left is not None:
                successor.left.parent = successor

    def _transplant(self, u: ExpressionNode, v: Optional[ExpressionNode]) -> None:
        if u.parent is None:
            self.root = v
        elif u is u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        if v is not None:
            v.parent = u.parent

    def _minimum(self, node: ExpressionNode) -> ExpressionNode:
        current = node
        while current.left is not None:
            current = current.left
        return current

    def _node_to_dict(self, node: Optional[ExpressionNode]) -> Optional[Dict[str, Any]]:
        if node is None:
            return None
        return {
            "node": node.as_dict(),
            "left": self._node_to_dict(node.left),
            "right": self._node_to_dict(node.right),
        }

    def _append_tree_lines(
        self,
        node: ExpressionNode,
        *,
        prefix: str,
        is_last: bool,
        lines: List[str],
    ) -> None:
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{node.payload.normalized}")
        children = [child for child in (node.left, node.right) if child is not None]
        next_prefix = prefix + ("    " if is_last else "│   ")
        for index, child in enumerate(children):
            child_is_last = index == len(children) - 1
            self._append_tree_lines(
                child,
                prefix=next_prefix,
                is_last=child_is_last,
                lines=lines,
            )

    def _build_payload(self, expression: Union[str, Expression]) -> ExpressionPayload:
        try:
            if isinstance(expression, Expression):
                expr_obj = expression
                source = str(expression)
            else:
                expr_obj = self._parser.parse(expression)
                source = expression
        except (InvalidExpressionException, ValueError) as exc:
            raise InvalidExpressionInput(str(exc)) from exc

        normalized = str(expr_obj)
        tokens = self._tokenize(expr_obj)
        return ExpressionPayload(
            expression=expr_obj,
            normalized=normalized,
            length=len(tokens),
            tokens=tokens,
            source=source.strip(),
        )

    def _compare_payloads(self, lhs: ExpressionPayload, rhs: ExpressionPayload) -> int:
        if lhs.length != rhs.length:
            return -1 if lhs.length < rhs.length else 1

        if self._expressions_equivalent(lhs, rhs):
            return 0

        token_cmp = self._compare_tokens(lhs.tokens, rhs.tokens)
        if token_cmp != 0:
            return token_cmp

        if lhs.normalized == rhs.normalized:
            return 0
        return -1 if lhs.normalized < rhs.normalized else 1

    def _expressions_equivalent(self, lhs: ExpressionPayload, rhs: ExpressionPayload) -> bool:
        cache_key = tuple(sorted((lhs.normalized, rhs.normalized)))
        if cache_key in self._equivalence_cache:
            return self._equivalence_cache[cache_key]

        if lhs.normalized == rhs.normalized:
            self._equivalence_cache[cache_key] = True
            return True

        self._ensure_llm_client()
        if self._llm_client is None or self._llm_model is None:
            self._equivalence_cache[cache_key] = False
            return False

        system_prompt = PROMPT_HEAD
        user_prompt = PROMPT_COMPARE.format(expr1=lhs.source, expr2=rhs.source)

        try:
            response_text, _ = self._llm_model.chat_generate(
                self._llm_client,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("LLM equivalence check failed: %s", exc)
            result = False
        else:
            result = self._parse_equivalence_response(response_text)

        self._equivalence_cache[cache_key] = result
        return result

    def _parse_equivalence_response(self, response: str) -> bool:
        text = response.strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                value = parsed.get("equivalent")
            else:
                value = parsed
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lowered = value.lower()
                if lowered == "true":
                    return True
                if lowered == "false":
                    return False
        except json.JSONDecodeError:
            lowered = text.lower()
            if lowered in {"true", "false"}:
                return lowered == "true"
        logger.warning("Unexpected LLM response: %s", response)
        return False

    def _ensure_llm_client(self) -> None:
        if self._llm_client is not None and self._llm_model is not None:
            return
        if not _DEFAULT_API_KEY:
            logger.warning("LLM API key is not configured; equivalence checks will always fail.")
            self._llm_client = None
            self._llm_model = None
            return
        self._llm_client = OpenAI(api_key=_DEFAULT_API_KEY, base_url=_DEFAULT_BASE_URL)
        self._llm_model = OpenAIModel(
            model_name=_DEFAULT_MODEL_NAME,
            stop_words=_DEFAULT_STOP_WORDS,
            max_new_tokens=_DEFAULT_MAX_NEW_TOKENS,
        )

    def _compare_tokens(self, lhs_tokens: Tuple[str, ...], rhs_tokens: Tuple[str, ...]) -> int:
        for l_token, r_token in zip(lhs_tokens, rhs_tokens):
            if l_token == r_token:
                continue
            l_kind, l_value = self._split_token(l_token)
            r_kind, r_value = self._split_token(r_token)
            if l_kind != r_kind:
                l_rank = _TOKEN_KIND_ORDER.get(l_kind, 99)
                r_rank = _TOKEN_KIND_ORDER.get(r_kind, 99)
                if l_rank != r_rank:
                    return -1 if l_rank < r_rank else 1
                return -1 if l_kind < r_kind else 1
            if l_kind in {"OP", "FEATURE"}:
                return -1 if l_value < r_value else 1
            if l_kind in {"CONST", "DT"}:
                if l_value < r_value:
                    return -1
                if l_value > r_value:
                    return 1
                continue
            return -1 if l_value < r_value else 1
        if len(lhs_tokens) == len(rhs_tokens):
            return 0
        return -1 if len(lhs_tokens) < len(rhs_tokens) else 1

    @staticmethod
    def _split_token(token: str) -> Tuple[str, Any]:
        try:
            kind, value = token.split(":", 1)
        except ValueError:
            return token, token
        if kind in {"CONST", "DT"}:
            try:
                return kind, float(value)
            except ValueError:
                return kind, value
        return kind, value

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
