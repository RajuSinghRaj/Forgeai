"""Registry of domain-independent tools that generated agents can select and invoke."""

from __future__ import annotations

import ast
import csv
import operator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from forgeai.retrieval.loaders import SUPPORTED_EXTENSIONS
from forgeai.retrieval.vector_store import ChromaVectorStore

_BINARY_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_SAFE_MATH_FUNCS: dict[str, Callable[..., float]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
}


def _safe_eval_math_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval_math_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left = _safe_eval_math_node(node.left)
        right = _safe_eval_math_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent magnitude exceeds safe limit (100).")
        return float(_BINARY_OPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return float(_UNARY_OPS[type(node.op)](_safe_eval_math_node(node.operand)))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn = _SAFE_MATH_FUNCS.get(node.func.id)
        if fn is None or node.keywords:
            raise ValueError(f"Unsupported math function: {node.func.id}")
        args = [_safe_eval_math_node(arg) for arg in node.args]
        return float(fn(*args))
    if isinstance(node, ast.List):
        # Allowed solely as argument to safe aggregation functions like sum([...]).
        raise ValueError("Use comma-separated arguments for min/max instead of nested lists.")
    raise ValueError(f"Disallowed expression element: {type(node).__name__}")


def _evaluate_math_expression(expression: str) -> float:
    cleaned = expression.strip()
    if not cleaned or len(cleaned) > 200:
        raise ValueError("Expression must be 1..200 characters.")
    tree = ast.parse(cleaned, mode="eval")
    return _safe_eval_math_node(tree)


@dataclass(frozen=True)
class ToolDefinition:
    """Metadata and callable implementation for a registered agent tool."""

    name: str
    description: str
    parameters_schema: dict[str, Any]
    handler: Callable[..., Any] = field(repr=False)


class ToolRegistry:
    """Lookup and execution container for domain-independent agent tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        if not tool.name.strip():
            raise ValueError("Tool name must not be empty.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition:
        if name not in self._tools:
            available = ", ".join(sorted(self._tools)) or "(none)"
            raise KeyError(f"Unknown tool '{name}'. Available tools: {available}")
        return self._tools[name]

    def has(self, name: str) -> bool:
        return name in self._tools

    def list_specs(self) -> list[dict[str, Any]]:
        """Return JSON-serializable tool specifications for prompt construction."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters_schema": tool.parameters_schema,
            }
            for tool in self._tools.values()
        ]

    def execute(self, name: str, **kwargs: Any) -> Any:
        tool = self.get(name)
        return tool.handler(**kwargs)


def build_default_registry(
    *,
    vector_store: ChromaVectorStore | None = None,
    documents_dir: Path | None = None,
) -> ToolRegistry:
    """Register standard domain-independent tools for generated agents."""
    registry = ToolRegistry()

    def retrieve_documents(query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if vector_store is None:
            raise RuntimeError("No vector_store configured for retrieve_documents.")
        results = vector_store.query(query, top_k=top_k)
        return [item.model_dump() for item in results]

    def inspect_csv_records(
        file_name: str,
        column: str | None = None,
        contains: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, str]]:
        if documents_dir is None:
            raise RuntimeError("No documents_dir configured for inspect_csv_records.")
        root = Path(documents_dir).resolve()
        target = (root / file_name).resolve()
        if not str(target).startswith(str(root)):
            raise ValueError("file_name must stay within documents_dir.")
        if not target.exists() or target.suffix.lower() != ".csv":
            raise FileNotFoundError(f"CSV file not found: {file_name}")

        matches: list[dict[str, str]] = []
        needle = (contains or "").strip().lower()
        with target.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row_num, row in enumerate(reader, start=1):
                if column and needle:
                    cell = (row.get(column) or "").lower()
                    if needle not in cell:
                        continue
                elif needle:
                    if not any(needle in (val or "").lower() for val in row.values()):
                        continue
                record = {"_row": str(row_num)}
                record.update({k: (v or "") for k, v in row.items() if k is not None})
                matches.append(record)
                if len(matches) >= limit:
                    break
        return matches

    def calculate_math(expression: str) -> dict[str, Any]:
        value = _evaluate_math_expression(expression)
        return {"expression": expression.strip(), "result": value}

    def list_corpus_documents() -> list[dict[str, Any]]:
        if documents_dir is None:
            raise RuntimeError("No documents_dir configured for list_corpus_documents.")
        root = Path(documents_dir).resolve()
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"Document folder not found: {root}")
        entries: list[dict[str, Any]] = []
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            ext = path.suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue
            entries.append(
                {
                    "file_name": path.relative_to(root).as_posix(),
                    "file_type": ext,
                    "size_bytes": path.stat().st_size,
                }
            )
        return entries

    registry.register(
        ToolDefinition(
            name="retrieve_documents",
            description="Semantic search across indexed source documents returning cited chunks.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                "required": ["query"],
            },
            handler=retrieve_documents,
        )
    )
    registry.register(
        ToolDefinition(
            name="inspect_csv_records",
            description="Filter and inspect structured rows from a CSV document in the corpus.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "file_name": {"type": "string"},
                    "column": {"type": "string"},
                    "contains": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                "required": ["file_name"],
            },
            handler=inspect_csv_records,
        )
    )
    registry.register(
        ToolDefinition(
            name="calculate_math",
            description="Safely evaluate arithmetic expressions (totals, percentages, differences, min/max, round) without arbitrary code execution.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                },
                "required": ["expression"],
            },
            handler=calculate_math,
        )
    )
    registry.register(
        ToolDefinition(
            name="list_corpus_documents",
            description="List available source document filenames, extensions, and file sizes in the corpus.",
            parameters_schema={
                "type": "object",
                "properties": {},
            },
            handler=list_corpus_documents,
        )
    )
    return registry
