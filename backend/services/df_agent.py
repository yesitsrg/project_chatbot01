# services/df_agent.py

"""
DataFrame Agent with bounded iterations and safety checks.

Safety features:
- max_iterations parameter (default 5) prevents infinite loops
- JOIN pattern detection for early rejection
- Timeout mechanism (future enhancement)
- Graceful fallback on agent failure
"""

from typing import Any, Dict
import pandas as pd
import re

from core import get_logger
from services.llm_factory import get_llm

logger = get_logger(__name__)


def _detect_unsupported_pattern(question: str) -> str:
    """
    Detect query patterns that are known to fail with pandas agent.
    Returns a rejection message if detected, else empty string.
    
    Unsupported patterns:
    - Multi-table JOINs
    - Cross-tab operations without explicit schema
    - Recursive aggregations
    """
    q_lower = question.lower()
    
    # Pattern 1: JOIN-like queries
    join_keywords = ["join", "merge", "combine", "cross", "link", "match"]
    if any(kw in q_lower for kw in join_keywords):
        if "with" in q_lower or "from" in q_lower or "and" in q_lower:
            return "This query requires multi-table operations which are not supported. Please ask about specific students or metrics."
    
    # Pattern 2: Complex aggregation chaining
    if q_lower.count("and") > 3 or q_lower.count("or") > 3:
        return "This query is too complex. Please break it into simpler questions."
    
    return ""


def run_df_agent(
    question: str,
    df: pd.DataFrame,
    df_name: str,
    max_iterations: int = 5,
) -> Dict[str, Any]:
    """
    Use a LangChain pandas dataframe agent to answer a question over a dataframe.
    
    Args:
        question: User question
        df: Input dataframe
        df_name: Name of the dataframe (for logging)
        max_iterations: Maximum agent iterations (default 5, prevents infinite loops)
    
    Returns:
        {"answer": str, "raw": optional raw agent output}
    """
    from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

    if df.empty:
        return {"answer": f"No data available for {df_name}.", "raw": None}

    # SAFETY CHECK 1: Detect unsupported patterns
    rejection_msg = _detect_unsupported_pattern(question)
    if rejection_msg:
        logger.warning(f"DF_AGENT unsupported pattern detected: {question}")
        return {"answer": rejection_msg, "raw": None}

    llm = get_llm()

    # SAFETY CHECK 2: Create agent with bounded iterations
    try:
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=False,
            allow_dangerous_code=True,
            max_iterations=max_iterations,  # CRITICAL: Prevents infinite loops
        )
    except Exception as e:
        logger.error(f"Failed to create agent for {df_name}: {e}")
        return {
            "answer": f"Could not initialize query agent for {df_name}.",
            "raw": None,
        }

    prompt = (
        f"You are a data analysis assistant working on the {df_name} table. "
        f"Use pandas operations correctly and answer strictly from the data.\n\n"
        f"Columns: {list(df.columns)}\n\n"
        f"Question: {question}"
    )

    logger.info(f"DF_AGENT {df_name} question={question} max_iterations={max_iterations}")
    
    try:
        result = agent.invoke(prompt)
    except Exception as e:
        logger.error(f"DF_AGENT failed for {df_name}: {e}", exc_info=True)
        return {
            "answer": f"I was unable to process this question for {df_name}. The query may be too complex.",
            "raw": None,
        }

    # Parse result
    if isinstance(result, str):
        return {"answer": result, "raw": None}
    if isinstance(result, dict) and "output" in result:
        return {"answer": result["output"], "raw": result}
    return {"answer": str(result), "raw": result}
