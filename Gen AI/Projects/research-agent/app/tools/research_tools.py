import ast
import operator
import time
from typing import Callable, TypeVar

from app.rag.retriever import search_documents
from ddgs import DDGS
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool


# =============================================================
# RETRY CONFIGURATION
# =============================================================

T = TypeVar("T")

MAX_TOOL_ATTEMPTS = 3
RETRY_BASE_DELAY_SECONDS = 1
WEB_SEARCH_TIMEOUT_SECONDS = 10


def _run_with_retries(
    operation: Callable[[], T],
    *,
    max_attempts: int = MAX_TOOL_ATTEMPTS,
) -> T:
    """
    Retry an operation that raises an exception.

    Delays are exponential: 1 second, then 2 seconds by default.
    Input validation and deterministic calculation errors are handled
    outside this helper and are not retried.
    """
    for attempt in range(max_attempts):
        try:
            return operation()
        except Exception:
            if attempt == max_attempts - 1:
                raise

            delay = RETRY_BASE_DELAY_SECONDS * (2 ** attempt)
            time.sleep(delay)

    raise RuntimeError("Retry operation ended unexpectedly.")


# =============================================================
# TOOL INPUT SCHEMAS
# =============================================================

class WebSearchInput(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=500,
        description="Specific web search query",
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of search results to return.",
    )


class CalculatorInput(BaseModel):
    expression: str = Field(
        min_length=1,
        max_length=100,
        description=(
            "Arithmetic expression using numbers and "
            "basic mathematical operators."
        ),
    )


class DocumentSearchInput(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=500,
        description="Question or keywords to search in indexed documents.",
    )
    max_results: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Maximum number of document chunks to return.",
    )


# =============================================================
# WEB SEARCH
# =============================================================

def web_search(query: str, max_results: int = 5) -> str:
    """Search the web with retries for temporary provider failures."""

    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if not 1 <= max_results <= 20:
        raise ValueError("max_results must be between 1 and 20.")

    def perform_search():
        with DDGS(timeout=WEB_SEARCH_TIMEOUT_SECONDS) as search_client:
            return list(
                search_client.text(
                    query,
                    max_results=max_results,
                )
            )

    try:
        results = _run_with_retries(perform_search)

        if not results:
            return (
                f"No web search results were found for '{query}'. "
                "Try a more specific or alternative search query."
            )

        formatted_results = []

        for result in results:
            formatted_results.append(
                {
                    "title": result.get("title", ""),
                    "url": result.get("href", ""),
                    "snippet": result.get("body", ""),
                }
            )

        return str(formatted_results)

    except Exception as exc:
        return (
            "WEB_SEARCH_FAILED: The external search provider could not "
            "complete the request after "
            f"{MAX_TOOL_ATTEMPTS} attempts. "
            f"Error type: {type(exc).__name__}. "
            "Do not repeatedly call the same search with identical "
            "arguments. If the question can be answered reliably "
            "from existing knowledge, answer it and disclose that "
            "live web verification was unavailable."
        )


# =============================================================
# SAFE CALCULATOR
# =============================================================

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

MAX_ABS_RESULT = 1e15


def safe_calculate(expression: str):
    """Evaluate a restricted arithmetic expression."""

    if len(expression) > 100:
        raise ValueError("Expression is too long.")

    tree = ast.parse(expression, mode="eval")

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)

        if isinstance(node, ast.Constant):
            if type(node.value) not in (int, float):
                raise ValueError("Only numeric values are allowed.")

            if abs(node.value) > MAX_ABS_RESULT:
                raise ValueError("Number exceeds the allowed limit.")

            return node.value

        if isinstance(node, ast.BinOp):
            operation = OPERATORS.get(type(node.op))

            if operation is None:
                raise ValueError("Unsupported arithmetic operator.")

            if isinstance(node.op, ast.Pow):
                base = evaluate(node.left)
                exponent = evaluate(node.right)

                if abs(exponent) > 10:
                    raise ValueError("Exponent is too large.")

                result = operation(base, exponent)
            else:
                result = operation(
                    evaluate(node.left),
                    evaluate(node.right),
                )

            if (
                isinstance(result, (int, float))
                and abs(result) > MAX_ABS_RESULT
            ):
                raise ValueError("Result exceeds the allowed limit.")

            return result

        if isinstance(node, ast.UnaryOp):
            operation = OPERATORS.get(type(node.op))

            if operation is None:
                raise ValueError("Unsupported unary operator.")

            return operation(evaluate(node.operand))

        raise ValueError("Unsupported expression.")

    return evaluate(tree)


def calculator(expression: str) -> str:
    """Calculate a basic arithmetic expression without retrying deterministic errors."""

    try:
        return str(safe_calculate(expression))
    except (
        ValueError,
        TypeError,
        ZeroDivisionError,
        OverflowError,
        SyntaxError,
    ) as exc:
        return f"Calculation error: {exc}"


# =============================================================
# DOCUMENT SEARCH / RAG
# =============================================================

def document_search(query: str, max_results: int = 4) -> str:
    """Search local documents with retries for temporary retrieval failures."""

    if not query or not query.strip():
        raise ValueError("Document search query cannot be empty.")

    if not 1 <= max_results <= 10:
        raise ValueError("max_results must be between 1 and 10.")

    try:
        matches = _run_with_retries(
            lambda: search_documents(
                query=query,
                k=max_results,
            )
        )

        if not matches:
            return (
                "NO_RELEVANT_DOCUMENTS: No matching document "
                "chunks were found. Do not claim the documents "
                "support an answer."
            )

        formatted = []

        for match in matches:
            item = {
                "source": match.get("source", "unknown"),
                "page": match.get("page"),
                "content": match.get("content", ""),
            }

            if match.get("relevance") is not None:
                item["relevance"] = match["relevance"]
            elif match.get("distance") is not None:
                item["distance"] = match["distance"]

            formatted.append(item)

        return str(formatted)

    except Exception as exc:
        return (
            "RAG_SEARCH_FAILED: Local document search failed after "
            f"{MAX_TOOL_ATTEMPTS} attempts. "
            f"Error type: {type(exc).__name__}: {exc}"
        )


# =============================================================
# LANGCHAIN TOOL REGISTRATION
# =============================================================

web_search_tool = StructuredTool.from_function(
    func=web_search,
    name="web_search",
    description=(
        "Search the web for relevant information, technical "
        "documentation, research topics, and source URLs. "
        "Use this when external evidence is required."
    ),
    args_schema=WebSearchInput,
)

calculator_tool = StructuredTool.from_function(
    func=calculator,
    name="calculator",
    description=(
        "Perform arithmetic calculation accurately. "
        "Use this instead of estimating numerical results."
    ),
    args_schema=CalculatorInput,
)

document_search_tool = StructuredTool.from_function(
    func=document_search,
    name="document_search",
    description=(
        "Search the locally indexed PDF and TXT knowledge base. "
        "Use this when the user asks about information that may "
        "be contained in the user's documents. Returns relevant "
        "text chunks and source metadata."
    ),
    args_schema=DocumentSearchInput,
)


tools = [
    web_search_tool,
    calculator_tool,
    document_search_tool,
]

# Map each tool name to its corresponding tool object.
TOOLS_BY_NAME = {
    tool.name: tool
    for tool in tools
}
import ast
import operator
import time
from typing import Callable, TypeVar

from app.rag.retriever import search_documents
from ddgs import DDGS
from pydantic import BaseModel, ConfigDict, Field
from langchain_core.tools import StructuredTool


# =============================================================
# RETRY CONFIGURATION
# =============================================================

T = TypeVar("T")

MAX_TOOL_ATTEMPTS = 3
RETRY_BASE_DELAY_SECONDS = 1
WEB_SEARCH_TIMEOUT_SECONDS = 10


def _run_with_retries(
    operation: Callable[[], T],
    *,
    max_attempts: int = MAX_TOOL_ATTEMPTS,
) -> T:
    """
    Retry an operation that raises an exception.

    Delays are exponential: 1 second, then 2 seconds by default.
    Input validation and deterministic calculation errors are handled
    outside this helper and are not retried.
    """
    for attempt in range(max_attempts):
        try:
            return operation()
        except Exception:
            if attempt == max_attempts - 1:
                raise

            delay = RETRY_BASE_DELAY_SECONDS * (2 ** attempt)
            time.sleep(delay)

    raise RuntimeError("Retry operation ended unexpectedly.")


# =============================================================
# TOOL INPUT SCHEMAS
# =============================================================

class WebSearchInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    query: str = Field(
        strict=True,
        min_length=1,
        max_length=500,
        description="Specific web search query",
    )
    max_results: int = Field(
        default=5,
        strict=True,
        ge=1,
        le=20,
        description="Number of search results to return.",
    )


class CalculatorInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    expression: str = Field(
        strict=True,
        min_length=1,
        max_length=100,
        description=(
            "Arithmetic expression using numbers and "
            "basic mathematical operators."
        ),
    )


class DocumentSearchInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    query: str = Field(
        strict=True,
        min_length=1,
        max_length=500,
        description="Question or keywords to search in indexed documents.",
    )
    max_results: int = Field(
        default=4,
        strict=True,
        ge=1,
        le=10,
        description="Maximum number of document chunks to return.",
    )


# =============================================================
# WEB SEARCH
# =============================================================

def web_search(query: str, max_results: int = 5) -> str:
    """Search the web with retries for temporary provider failures."""

    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if not 1 <= max_results <= 20:
        raise ValueError("max_results must be between 1 and 20.")

    def perform_search():
        with DDGS(timeout=WEB_SEARCH_TIMEOUT_SECONDS) as search_client:
            return list(
                search_client.text(
                    query,
                    max_results=max_results,
                )
            )

    try:
        results = _run_with_retries(perform_search)

        if not isinstance(results, list):
            raise TypeError("Web search returned an unexpected response type.")

        if not results:
            return (
                f"No web search results were found for '{query}'. "
                "Try a more specific or alternative search query."
            )

        formatted_results = []

        for result in results:
            if not isinstance(result, dict):
                raise TypeError("Web search returned a malformed result item.")

            title = result.get("title", "")
            url = result.get("href", "")
            snippet = result.get("body", "")

            if not all(isinstance(value, str) for value in (title, url, snippet)):
                raise TypeError("Web search result fields must be strings.")

            formatted_results.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                }
            )

        return str(formatted_results)

    except Exception as exc:
        return (
            "WEB_SEARCH_FAILED: The external search provider could not "
            "complete the request after "
            f"{MAX_TOOL_ATTEMPTS} attempts. "
            f"Error type: {type(exc).__name__}. "
            "Do not repeatedly call the same search with identical "
            "arguments. If the question can be answered reliably "
            "from existing knowledge, answer it and disclose that "
            "live web verification was unavailable."
        )


# =============================================================
# SAFE CALCULATOR
# =============================================================

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

MAX_ABS_RESULT = 1e15


def safe_calculate(expression: str):
    """Evaluate a restricted arithmetic expression."""

    if len(expression) > 100:
        raise ValueError("Expression is too long.")

    tree = ast.parse(expression, mode="eval")

    if sum(1 for _ in ast.walk(tree)) > 50:
        raise ValueError("Expression is too complex.")

    def evaluate(node, depth=0):
        if depth > 10:
            raise ValueError("Expression nesting is too deep.")

        if isinstance(node, ast.Expression):
            return evaluate(node.body, depth + 1)

        if isinstance(node, ast.Constant):
            if type(node.value) not in (int, float):
                raise ValueError("Only numeric values are allowed.")

            if abs(node.value) > MAX_ABS_RESULT:
                raise ValueError("Number exceeds the allowed limit.")

            return node.value

        if isinstance(node, ast.BinOp):
            operation = OPERATORS.get(type(node.op))

            if operation is None:
                raise ValueError("Unsupported arithmetic operator.")

            if isinstance(node.op, ast.Pow):
                base = evaluate(node.left, depth + 1)
                exponent = evaluate(node.right, depth + 1)

                if abs(exponent) > 10:
                    raise ValueError("Exponent is too large.")

                result = operation(base, exponent)
            else:
                result = operation(
                    evaluate(node.left, depth + 1),
                    evaluate(node.right, depth + 1),
                )

            if (
                isinstance(result, (int, float))
                and abs(result) > MAX_ABS_RESULT
            ):
                raise ValueError("Result exceeds the allowed limit.")

            return result

        if isinstance(node, ast.UnaryOp):
            operation = OPERATORS.get(type(node.op))

            if operation is None:
                raise ValueError("Unsupported unary operator.")

            return operation(evaluate(node.operand, depth + 1))

        raise ValueError("Unsupported expression.")

    return evaluate(tree)


def calculator(expression: str) -> str:
    """Calculate a basic arithmetic expression without retrying deterministic errors."""

    try:
        return str(safe_calculate(expression))
    except (
        ValueError,
        TypeError,
        ZeroDivisionError,
        OverflowError,
        SyntaxError,
    ) as exc:
        return f"Calculation error: {exc}"


# =============================================================
# DOCUMENT SEARCH / RAG
# =============================================================

def document_search(query: str, max_results: int = 4) -> str:
    """Search local documents with retries for temporary retrieval failures."""

    if not query or not query.strip():
        raise ValueError("Document search query cannot be empty.")

    if not 1 <= max_results <= 10:
        raise ValueError("max_results must be between 1 and 10.")

    try:
        matches = _run_with_retries(
            lambda: search_documents(
                query=query,
                k=max_results,
            )
        )

        if not isinstance(matches, list):
            raise TypeError("Document retrieval returned an unexpected response type.")

        if not matches:
            return (
                "NO_RELEVANT_DOCUMENTS: No matching document "
                "chunks were found. Do not claim the documents "
                "support an answer."
            )

        formatted = []

        for match in matches:
            if not isinstance(match, dict):
                raise TypeError("Document retrieval returned a malformed result item.")

            content = match.get("content", "")
            source = match.get("source", "unknown")

            if not isinstance(content, str) or not isinstance(source, str):
                raise TypeError("Document result content and source must be strings.")

            item = {
                "source": source,
                "page": match.get("page"),
                "content": content,
            }

            if match.get("relevance") is not None:
                item["relevance"] = match["relevance"]
            elif match.get("distance") is not None:
                item["distance"] = match["distance"]

            formatted.append(item)

        return str(formatted)

    except Exception as exc:
        return (
            "RAG_SEARCH_FAILED: Local document search failed after "
            f"{MAX_TOOL_ATTEMPTS} attempts. "
            f"Error type: {type(exc).__name__}: {exc}"
        )


# =============================================================
# LANGCHAIN TOOL REGISTRATION
# =============================================================

web_search_tool = StructuredTool.from_function(
    func=web_search,
    name="web_search",
    description=(
        "Search the web for relevant information, technical "
        "documentation, research topics, and source URLs. "
        "Use this when external evidence is required."
    ),
    args_schema=WebSearchInput,
)

calculator_tool = StructuredTool.from_function(
    func=calculator,
    name="calculator",
    description=(
        "Perform arithmetic calculation accurately. "
        "Use this instead of estimating numerical results."
    ),
    args_schema=CalculatorInput,
)

document_search_tool = StructuredTool.from_function(
    func=document_search,
    name="document_search",
    description=(
        "Search the locally indexed PDF and TXT knowledge base. "
        "Use this when the user asks about information that may "
        "be contained in the user's documents. Returns relevant "
        "text chunks and source metadata."
    ),
    args_schema=DocumentSearchInput,
)


tools = [
    web_search_tool,
    calculator_tool,
    document_search_tool,
]

# Map each tool name to its corresponding tool object.
TOOLS_BY_NAME = {
    tool.name: tool
    for tool in tools
}
