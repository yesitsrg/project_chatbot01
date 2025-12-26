#!/usr/bin/env python3
"""
TRANSCRIPT TOOL - Universal CSV Agent with Dynamic Context
Production-ready solution for ALL transcript queries.
Location: backend/services/tools/transcript_tool.py
"""

import os
import logging
import re
from typing import Dict, Any, Optional
import pandas as pd

from models.schemas import ToolResult
from services.data_views import get_transcript_df
from config import get_settings

logger = logging.getLogger(__name__)


# ============================================================================
# LANGCHAIN IMPORTS - Version 1.x Compatible
# ============================================================================

AGENT_AVAILABLE = False

try:
    from langchain_groq import ChatGroq
    from langchain_ollama import ChatOllama
    from langchain_experimental.agents import create_pandas_dataframe_agent
    
    AGENT_AVAILABLE = True
    logger.info("LangChain imports successful - Agent mode enabled")
    
except Exception as e:
    logger.warning(f"LangChain import failed: {e}")
    AGENT_AVAILABLE = False


# ============================================================================
# DYNAMIC CONTEXT BUILDER
# ============================================================================

def _build_dynamic_context(df: pd.DataFrame) -> Dict[str, Any]:
    """Extract key metadata from DataFrame for dynamic prompt injection"""
    
    context = {
        'total_rows': len(df),
        'total_students': df['Student Name'].nunique() if 'Student Name' in df.columns else 0,
        'total_courses': df['Course Title'].dropna().nunique() if 'Course Title' in df.columns else 0,
        'sample_students': list(df['Student Name'].unique()[:8]) if 'Student Name' in df.columns else [],
        'sample_courses': list(df['Course Title'].dropna().unique()[:8]) if 'Course Title' in df.columns else [],
        'available_terms': list(df['Term'].dropna().unique()) if 'Term' in df.columns else [],
        'gpa_range': {
            'min': df[df['GPA'] > 0]['GPA'].min() if 'GPA' in df.columns and len(df[df['GPA'] > 0]) > 0 else 0.0,
            'max': df['GPA'].max() if 'GPA' in df.columns else 0.0
        },
        'columns': list(df.columns)
    }
    
    return context


# ============================================================================
# QUERY PREPROCESSOR
# ============================================================================

def preprocess_query(query: str, df_columns: list) -> Dict[str, Any]:
    """Analyze query and detect patterns"""
    query_lower = query.lower()
    hints = []
    patterns = {}
    
    # Detect partial names
    name_patterns = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', query)
    if name_patterns and any(kw in query_lower for kw in ['gpa', 'courses', 'enrolled', 'hours', 'student']):
        patterns['has_student_name'] = True
        patterns['name_fragments'] = name_patterns
        hints.append("FUZZY_NAME_MATCH")
    
    # Detect term queries
    term_match = re.search(r'(\d{4}-\d{4}\s+(?:Fall|Spring|Summer))|((?:Fall|Spring|Summer)\s+\d{4})', query, re.IGNORECASE)
    if term_match:
        patterns['term'] = term_match.group(0)
        hints.append("TERM_SPECIFIC")
    
    # Detect GPA queries
    if 'gpa' in query_lower or 'grade' in query_lower:
        patterns['asking_gpa'] = True
        hints.append("GPA_QUERY")
    
    # Detect course queries
    if any(kw in query_lower for kw in ['course', 'enrolled', 'class']):
        patterns['asking_courses'] = True
        hints.append("COURSE_QUERY")
    
    # Detect aggregation
    if any(kw in query_lower for kw in ['how many', 'count', 'list all', 'display', 'show all']):
        patterns['is_aggregation'] = True
        hints.append("AGGREGATION")
    
    # Detect ranking
    if any(kw in query_lower for kw in ['highest', 'lowest', 'top', 'best', 'worst']):
        patterns['is_ranking'] = True
        hints.append("RANKING")
    
    # Detect tabular output
    if any(kw in query_lower for kw in ['table', 'tabular', 'format']):
        patterns['wants_table'] = True
        hints.append("TABULAR_OUTPUT")
    
    # Detect average/mean
    if any(kw in query_lower for kw in ['average', 'mean', 'avg']):
        patterns['wants_average'] = True
        hints.append("AVERAGE_CALCULATION")
    
    return {
        'original_query': query,
        'hints': hints,
        'patterns': patterns,
        'columns_available': df_columns
    }


# ============================================================================
# ENHANCED AGENT PREFIX WITH DYNAMIC CONTEXT
# ============================================================================

AGENT_PREFIX_TEMPLATE = """
You are an EXPERT data analyst working with student transcript data.

================================================================================
DATA CONTEXT (LIVE FROM CURRENT DATASET)
================================================================================

Dataset Size: {total_rows} enrollment records
Unique Students: {total_students}
Unique Courses: {total_courses}

Sample Students in Data:
{sample_students_formatted}

Sample Courses in Data:
{sample_courses_formatted}

Available Terms:
{available_terms_formatted}

GPA Range: {gpa_min:.2f} - {gpa_max:.2f}

Available Columns:
{columns_list}

================================================================================
CRITICAL DATA STRUCTURE
================================================================================

- Each row = ONE COURSE ENROLLMENT
- Students appear in MULTIPLE rows (one per course)
- You MUST aggregate correctly to avoid double-counting

================================================================================
YOUR PROBLEM-SOLVING APPROACH
================================================================================

STEP 1: Understand what is being asked
STEP 2: Write ONE code statement to get the answer
STEP 3: Execute it ONCE
STEP 4: Format the result and respond immediately

================================================================================
KEY TECHNIQUES
================================================================================

FUZZY MATCHING (always use .str.contains):
  df[df['Student Name'].str.contains('Leslie', case=False, na=False)]
  df[df['Student Name'].str.contains('Trista.*Barrett', case=False, regex=True, na=False)]
  df[df['Course Title'].str.contains('Business.*Math', case=False, regex=True, na=False)]

TERM FILTERING (flexible matching):
  df[df['Term'].str.contains('Fall', case=False, na=False) & 
     df['Term'].str.contains('2024', case=False, na=False)]

GPA HANDLING (filter and aggregate):
  Career GPA: df[df['GPA'] > 0]['GPA'].max()
  Average: df[df['GPA'] > 0].groupby('Student Name')['GPA'].max().mean()

COUNTING:
  Students: df['Student Name'].nunique()
  Courses: df['Course Title'].dropna().nunique()

RANKING:
  df[df['GPA'] > 0].groupby('Student Name')['GPA'].max().sort_values(ascending=False).head(5)

================================================================================
EXAMPLES (5 KEY PATTERNS)
================================================================================

Q: "What is Trista Barrett's GPA?"
Code:
student_rows = df[df['Student Name'].str.contains('Trista.*Barrett', case=False, regex=True, na=False)]
gpa_values = student_rows[student_rows['GPA'] > 0]['GPA']
gpa_values.max() if len(gpa_values) > 0 else 0.0

Q: "What courses is Leslie enrolled in?"
Code:
courses = df[df['Student Name'].str.contains('Leslie', case=False, na=False)]['Course Title'].dropna().unique().tolist()
student_name = df[df['Student Name'].str.contains('Leslie', case=False, na=False)]['Student Name'].iloc[0]
f"{{student_name}} is enrolled in:\\n" + "\\n".join([f"- {{c}}" for c in courses])

Q: "Top 5 students by GPA"
Code:
df[df['GPA'] > 0].groupby('Student Name')['GPA'].max().reset_index().sort_values('GPA', ascending=False).head(5)

Q: "How many courses did Trista take in Fall 2024?"
Code:
student_filter = df['Student Name'].str.contains('Trista', case=False, na=False)
term_filter = df['Term'].str.contains('Fall', case=False, na=False) & df['Term'].str.contains('2024', case=False, na=False)
df[student_filter & term_filter]['Course Title'].dropna().nunique()

Q: "Average GPA for Fall 2024?"
Code:
term_rows = df[df['Term'].str.contains('Fall', case=False, na=False) & df['Term'].str.contains('2024', case=False, na=False)]
term_rows[term_rows['GPA'] > 0].groupby('Student Name')['GPA'].max().mean()

================================================================================
QUERY HINTS (for this specific query):
================================================================================
{query_hints}

================================================================================
IMMEDIATE RESPONSE PROTOCOL (CRITICAL!)
================================================================================

After ONE successful code execution that returns valid data:

1. Take the result exactly as returned
2. Format it nicely if needed
3. Return your final answer IMMEDIATELY
4. Do NOT re-execute the code to verify

Example:
- Code executes: ['Biology', 'Math', 'English']
- You see the result
- Format: "Courses: Biology, Math, English"
- RESPOND and STOP

Do NOT:
- Run the same code again
- Try to verify the result
- Second-guess the execution

The python_repl_ast output IS the source of truth.
If code returns data, that data is CORRECT and FINAL.

================================================================================
EXECUTION RULES
================================================================================

1. Write concise code (1-3 lines when possible)
2. Execute ONCE
3. Accept the first valid result
4. Format and respond immediately
5. STOP

Only execute again if:
- First attempt had a syntax error
- Result was empty/null and you need to try a different approach

FORMAT GUIDELINES:
- For lists: Use bullet points (\\n- item)
- For single values: Include context ("GPA: 3.5")
- For rankings: Number the items (1. Name - GPA)
- For tables: Use DataFrame.to_markdown()

================================================================================
KEY REMINDERS
================================================================================

ALWAYS:
- Use fuzzy matching (.str.contains)
- Filter GPA > 0 before calculations
- Use .dropna() for text columns
- Group by student for aggregations
- Accept first valid execution result

NEVER:
- Use exact string matching (==)
- Average GPA column directly
- Re-execute unnecessarily
- Ignore valid execution results

Now solve the query: Write ONE code statement, execute it ONCE, format result, respond.
"""


# ============================================================================
# LLM CREATION
# ============================================================================

def _create_llm():
    """Create LLM using settings (Groq or Ollama)."""
    settings = get_settings()
    
    groq_key = settings.groq_api_key or os.getenv("GROQ_API_KEY")
    if groq_key:
        logger.info("[TranscriptTool] Using Groq LLM")
        try:
            return ChatGroq(
                groq_api_key=groq_key,
                model_name=settings.groq_model,
                temperature=0,
                max_tokens=settings.llm_max_tokens
            )
        except Exception as e:
            logger.warning(f"[TranscriptTool] Groq failed: {e}")
    
    logger.info("[TranscriptTool] Using Ollama LLM")
    return ChatOllama(
        base_url=settings.ollama_base_url.rstrip("/"),
        model=settings.ollama_model,
        temperature=0
    )


# ============================================================================
# MAIN ANSWER FUNCTION
# ============================================================================

def answer(query: str, params: Dict[str, Any] = None) -> ToolResult:
    """
    Universal CSV Agent for ALL transcript queries with dynamic context.
    """
    logger.info("="*80)
    logger.info(f"[TranscriptTool] QUERY: '{query}'")
    
    # Load transcript data
    df = get_transcript_df()
    
    if df is None or df.empty:
        logger.error("[TranscriptTool] No transcript data available")
        return ToolResult(
            data={},
            explanation="No transcript data available. Please upload transcript files.",
            confidence=0.0,
            format_hint="text",
            citations=[]
        )
    
    logger.info(f"[TranscriptTool] DataFrame: {len(df)} rows, {len(df.columns)} columns")
    logger.info(f"[TranscriptTool] Unique students: {df['Student Name'].nunique() if 'Student Name' in df.columns else 'N/A'}")
    
    if not AGENT_AVAILABLE:
        logger.warning("[TranscriptTool] LangChain not available, using fallback")
        return _fallback_simple_query(df, query)
    
    try:
        # STEP 1: Extract dynamic context from live data
        context = _build_dynamic_context(df)
        
        # STEP 2: Preprocess query
        preprocessed = preprocess_query(query, df.columns.tolist())
        
        logger.info(f"[TranscriptTool] Query hints: {preprocessed['hints']}")
        
        # STEP 3: Build dynamic prompt with actual data context
        agent_prefix = AGENT_PREFIX_TEMPLATE.format(
            total_rows=context['total_rows'],
            total_students=context['total_students'],
            total_courses=context['total_courses'],
            sample_students_formatted='\n'.join(f"  - {s}" for s in context['sample_students']),
            sample_courses_formatted='\n'.join(f"  - {c}" for c in context['sample_courses']),
            available_terms_formatted='\n'.join(f"  - {t}" for t in context['available_terms']),
            gpa_min=context['gpa_range']['min'],
            gpa_max=context['gpa_range']['max'],
            columns_list='\n'.join(f"  - {col}" for col in context['columns']),
            query_hints='\n'.join(f"  - {hint}" for hint in preprocessed['hints']) if preprocessed['hints'] else "  - None"
        )
        
        # STEP 4: Create LLM
        llm = _create_llm()
        
        logger.info("[TranscriptTool] Creating pandas agent with dynamic context...")
        
        # STEP 5: Create agent with strict stopping
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            prefix=agent_prefix,
            verbose=True,
            agent_type="openai-tools",
            allow_dangerous_code=True,
            max_iterations=5,  # Reduced to force faster stopping
            early_stopping_method="force"  # Force stop after max iterations
        )
        
        logger.info(f"[TranscriptTool] Executing agent...")
        
        # STEP 6: Execute
        result = agent.invoke({"input": query})
        
        answer_text = result.get("output", "Unable to process query")
        
        logger.info(f"[TranscriptTool] Success")
        logger.info(f"[TranscriptTool] Answer: {answer_text[:200]}...")
        
        return ToolResult(
            data={
                "query": query,
                "preprocessing": preprocessed,
                "context": context
            },
            explanation=answer_text,
            confidence=0.88,
            format_hint="text",
            citations=["merged_transcripts.csv"]
        )
        
    except Exception as e:
        logger.error(f"[TranscriptTool] Agent failed: {e}", exc_info=True)
        return _fallback_simple_query(df, query)


# ============================================================================
# ENHANCED FALLBACK
# ============================================================================

def _fallback_simple_query(df, query: str) -> ToolResult:
    """Enhanced pandas fallback - handles common query types without LangChain"""
    query_lower = query.lower()
    
    name_col = 'Student Name'
    gpa_col = 'GPA'
    course_col = 'Course Title'
    
    # 1. SINGLE STUDENT GPA
    if 'gpa' in query_lower:
        name_match = re.search(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', query)
        if name_match:
            search_name = name_match.group(1)
            matches = df[df[name_col].str.contains(search_name.split()[0], case=False, na=False)]
            
            if len(matches) > 0:
                student_name = matches[name_col].iloc[0]
                gpa_values = matches[matches[gpa_col] > 0][gpa_col]
                career_gpa = gpa_values.max() if len(gpa_values) > 0 else 0.0
                
                return ToolResult(
                    data={"student": student_name, "gpa": float(career_gpa)},
                    explanation=f"**GPA for {student_name}:**\nCareer GPA: {career_gpa:.2f}",
                    confidence=0.90,
                    format_hint="text",
                    citations=["merged_transcripts.csv"]
                )
    
    # 2. COUNT STUDENTS
    if "how many student" in query_lower:
        count = df[name_col].nunique()
        return ToolResult(
            data={"count": count},
            explanation=f"There are **{count}** unique students.",
            confidence=0.95,
            format_hint="text",
            citations=["merged_transcripts.csv"]
        )
    
    # 3. LIST/RANK STUDENTS
    if any(kw in query_lower for kw in ['list', 'top', 'highest', 'rank', 'show me']):
        student_gpas = df[df[gpa_col] > 0].groupby(name_col)[gpa_col].max().reset_index()
        student_gpas = student_gpas.sort_values(gpa_col, ascending=False)
        
        top_n = 5 if 'top 5' in query_lower else 10
        
        result_text = f"**Top {min(top_n, len(student_gpas))} Students by GPA:**\n\n"
        for i, (_, row) in enumerate(student_gpas.head(top_n).iterrows(), 1):
            result_text += f"{i}. {row[name_col]}: GPA {row[gpa_col]:.2f}\n"
        
        return ToolResult(
            data={"students": student_gpas.head(top_n).to_dict('records')},
            explanation=result_text,
            confidence=0.90,
            format_hint="text",
            citations=["merged_transcripts.csv"]
        )
    
    # 4. COURSE ENROLLMENT
    if 'course' in query_lower and 'enrolled' in query_lower:
        name_match = re.search(r'\b([A-Z][a-z]+)\b', query)
        if name_match:
            search_name = name_match.group(1)
            matches = df[df[name_col].str.contains(search_name, case=False, na=False)]
            
            if len(matches) > 0:
                student_name = matches[name_col].iloc[0]
                courses = matches[course_col].dropna().unique()
                
                result_text = f"**Courses enrolled by {student_name}:**\n\n"
                for i, course in enumerate(courses, 1):
                    result_text += f"{i}. {course}\n"
                
                return ToolResult(
                    data={"student": student_name, "courses": list(courses)},
                    explanation=result_text,
                    confidence=0.90,
                    format_hint="text",
                    citations=["merged_transcripts.csv"]
                )
    
    return ToolResult(
        data={},
        explanation="Unable to process query. Please ensure LangChain is installed for advanced queries.",
        confidence=0.0,
        format_hint="text",
        citations=[]
    )
