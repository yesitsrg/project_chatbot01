#!/usr/bin/env python3
"""
TRANSCRIPT TOOL - Universal CSV Agent with Dynamic Context + Smart Course Filtering
Production-ready solution for ALL transcript queries with intelligent data quality handling.
Location: backend/services/tools/transcript_tool.py
"""

import os
import logging
import re
from typing import Dict, Any, Optional, List
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
# SMART COURSE FILTERING (NEW - Handles Bad Data)
# ============================================================================

def _filter_courses_with_llm(raw_courses: list, student_name: str, llm) -> list:
    """
    Use LLM to intelligently filter course list and remove junk data (dates, empty values).
    
    Args:
        raw_courses: Raw list from Course Title + Course Number columns
        student_name: Student name for context
        llm: LLM instance for filtering
    
    Returns:
        List of clean, valid course names/titles/codes
    """
    if not raw_courses or len(raw_courses) == 0:
        return []
    
    # Quick heuristic filter first (cheap)
    candidates = []
    for course in raw_courses:
        course_str = str(course).strip()
        
        # Skip obvious junk
        if not course_str or course_str == ' ':
            continue
        if len(course_str) < 3:  # Too short
            continue
        if re.match(r'^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$', course_str):  # Date pattern
            continue
        
        candidates.append(course_str)
    
    # If empty after filtering, return empty
    if not candidates:
        logger.warning(f"[CourseFilter] No valid courses found for {student_name}")
        return []
    
    # If <= 3 candidates and they look reasonable, skip LLM (optimization)
    if len(candidates) <= 3 and all(len(c) > 8 for c in candidates):
        logger.info(f"[CourseFilter] {len(candidates)} courses look valid, skipping LLM")
        return candidates
    
    # Use LLM for intelligent filtering
    logger.info(f"[CourseFilter] Sending {len(candidates)} candidates to LLM for filtering")
    
    filter_prompt = f"""
You are a data quality filter for student course records.

Student: {student_name}

Raw course data (may contain course titles, course codes, dates, or junk):
{candidates}

Task: Return ONLY the items that are actual courses (course titles, course names, or course codes).

Rules:
- Include: Course titles (e.g., "Introduction to Biology", "Business Mathematics")
- Include: Course codes (e.g., "MATH 101", "ENG-1010", "CS101")
- Exclude: Dates (e.g., "12/16/2024", "12-11-2024")
- Exclude: Empty strings or whitespace
- Exclude: Random numbers or gibberish

Output format: Return a Python list of valid courses ONLY.
Example: ["Introduction to Biology", "Business Mathematics", "MATH 101"]

If no valid courses found, return empty list: []

Respond with ONLY the Python list, nothing else.
"""
    
    try:
        from langchain_core.messages import HumanMessage
        
        response = llm.invoke([HumanMessage(content=filter_prompt)])
        response_text = response.content.strip()
        
        # Try to parse as Python list
        import ast
        try:
            filtered_courses = ast.literal_eval(response_text)
            if isinstance(filtered_courses, list):
                logger.info(f"[CourseFilter] LLM filtered to {len(filtered_courses)} valid courses")
                return filtered_courses
        except:
            pass
        
        # Fallback: extract items from response
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]
        filtered_courses = []
        for line in lines:
            clean = line.strip('- "\'[]')
            if clean and not re.match(r'^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$', clean):
                filtered_courses.append(clean)
        
        logger.info(f"[CourseFilter] LLM filtered to {len(filtered_courses)} valid courses (fallback parsing)")
        return filtered_courses
        
    except Exception as e:
        logger.error(f"[CourseFilter] LLM filtering failed: {e}")
        # Fallback: return candidates with basic filtering
        return [c for c in candidates if len(c) > 8]


# ============================================================================
# ENHANCED AGENT PREFIX WITH DYNAMIC CONTEXT
# ============================================================================

AGENT_PREFIX_TEMPLATE = """
You are an EXPERT data analyst working with student transcript data.

DATA: {total_rows} records | {total_students} students | {total_courses} courses | GPA range: {gpa_min:.2f}-{gpa_max:.2f}

Sample Students: {sample_students_formatted}
Sample Courses: {sample_courses_formatted}
Terms: {available_terms_formatted}
Columns: {columns_list}

STRUCTURE: Each row = ONE course enrollment. Students appear multiple times. Aggregate correctly.

================================================================================
QUERY: "{original_query}"
ANSWER ABOUT: {query_topic}
================================================================================

TECHNIQUES:

FUZZY MATCH (always use):
  df['Student Name'].str.contains('Leslie', case=False, na=False)
  df['Course Title'].str.contains('Business.*Math', case=False, regex=True, na=False)

TERM FILTER:
  df['Term'].str.contains('Fall', case=False, na=False) & df['Term'].str.contains('2024', na=False)

GPA (always filter > 0):
  Career: df[df['GPA'] > 0]['GPA'].max()
  Average: df[df['GPA'] > 0].groupby('Student Name')['GPA'].max().mean()

RANK:
  df[df['GPA'] > 0].groupby('Student Name')['GPA'].max().sort_values(ascending=False).head(5)

COURSE QUERIES (check both columns):
  courses_title = df[filter]['Course Title'].dropna().unique().tolist()
  courses_number = df[filter]['Course Number'].dropna().unique().tolist() if 'Course Number' in df.columns else []
  all_courses = list(set(courses_title + courses_number))

================================================================================
EXAMPLES
================================================================================

Q: "What is Trista Barrett's GPA?"
Code: df[df['Student Name'].str.contains('Trista.*Barrett', case=False, regex=True, na=False)][df['GPA'] > 0]['GPA'].max()

Q: "What courses is Leslie enrolled in?"
Code:
filter = df['Student Name'].str.contains('Leslie', case=False, na=False)
list(set(df[filter]['Course Title'].dropna().unique().tolist() + df[filter].get('Course Number', pd.Series()).dropna().unique().tolist()))

Q: "Top 5 students by GPA"
Code: df[df['GPA'] > 0].groupby('Student Name')['GPA'].max().sort_values(ascending=False).head(5)

================================================================================
EXECUTION PROTOCOL (CRITICAL - DO NOT VIOLATE)
================================================================================
1. Write ONE aggregated code statement that gets the full answer
2. Execute EXACTLY ONCE
3. Take result AS-IS
4. Format and respond immediately
5. STOP - DO NOT make additional tool calls

VIOLATION EXAMPLES (NEVER DO THIS):
❌ Checking each student individually in a loop
❌ Re-running code to verify
❌ Making multiple queries when one would work

CORRECT APPROACH:
✅ Use groupby/sort/head to get answer in ONE execution
✅ Trust the pandas result
✅ Respond immediately

Example: "Highest GPA student"
WRONG: Check each student (10 calls) ❌
RIGHT: df.groupby('Student Name')['GPA'].max().idxmax() (1 call) ✅

If query asks for GPA → Return ONLY GPA
If query asks for courses → Return ONLY courses
DO NOT mix topics.

Query hints: {query_hints}

ALWAYS:
- Use .str.contains (fuzzy matching)
- Filter GPA > 0 before calculations
- Use .dropna() for text columns
- Group by student for aggregations
- Accept first valid result

NEVER:
- Use exact matching (==)
- Re-execute valid results
- Add information not requested

Now solve: Write code, execute ONCE, format, respond.
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
    Universal CSV Agent for ALL transcript queries with dynamic context + smart filtering.
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
        
        # Determine query topic for focused prompt
        query_lower = query.lower()
        if 'gpa' in query_lower and 'course' not in query_lower:
            query_topic = "GPA/grades ONLY"
        elif 'course' in query_lower:
            query_topic = "course enrollment ONLY"
        elif any(kw in query_lower for kw in ['rank', 'top', 'highest', 'display', 'list']):
            query_topic = "student rankings/lists ONLY"
        else:
            query_topic = "the specific question asked"

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
            query_hints='\n'.join(f"  - {hint}" for hint in preprocessed['hints']) if preprocessed['hints'] else "  - None",
            original_query=query,  # NEW
            query_topic=query_topic  # NEW
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
            max_iterations=5,
            early_stopping_method="force"
        )
        
        logger.info(f"[TranscriptTool] Executing agent...")
        
        # STEP 6: Execute
        result = agent.invoke({"input": query})
        
        answer_text = result.get("output", "Unable to process query")

        # DEBUG: Check post-processing trigger conditions
        logger.info(f"[POST-PROCESS CHECK] Original query: '{query}'")
        logger.info(f"[POST-PROCESS CHECK] 'course' in query: {'course' in query.lower()}")
        logger.info(f"[POST-PROCESS CHECK] 'enrolled' in query: {'enrolled' in query.lower()}")
        logger.info(f"[POST-PROCESS CHECK] Answer preview: {answer_text[:200]}")
        
        # STEP 7: POST-PROCESSING - Smart course filtering if output looks suspicious
        if 'course' in query.lower() and 'enrolled' in query.lower():
            # Check if answer contains date patterns (indicates bad data)
            if re.search(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', answer_text):
                logger.warning("[TranscriptTool] Detected dates in course list - applying smart LLM filter")
                
                # Extract student name from query
                name_match = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})', query)
                student_name = name_match.group(1) if name_match else "Student"
                
                # Re-query with smart filtering
                student_filter = df['Student Name'].str.contains(student_name.split()[0], case=False, na=False)
                student_rows = df[student_filter]
                
                if len(student_rows) > 0:
                    # Collect from both columns
                    courses_title = student_rows['Course Title'].dropna().unique().tolist()
                    courses_number = student_rows['Course Number'].dropna().unique().tolist() if 'Course Number' in df.columns else []
                    raw_courses = list(set(courses_title + courses_number))
                    
                    # Apply LLM filter
                    filtered_courses = _filter_courses_with_llm(raw_courses, student_name, llm)
                    
                    if filtered_courses:
                        actual_student_name = student_rows['Student Name'].iloc[0]
                        answer_text = f"{actual_student_name} is enrolled in:\n" + "\n".join([f"- {c}" for c in filtered_courses])
                        logger.info(f"[TranscriptTool] Applied smart filter: {len(filtered_courses)} valid courses")
                    else:
                        answer_text = f"No valid course enrollment records found for {student_name}."
        
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
