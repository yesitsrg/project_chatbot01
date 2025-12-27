#!/usr/bin/env python3
"""
backend/services/tools/transcript_tool.py
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
from services.llm_factory import get_llm


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
# SMART COURSE FILTERING (ENHANCED - Handles Bad Data)
# ============================================================================


def _filter_courses_with_llm(raw_courses: list, student_name: str, llm) -> list:
    """
    Use LLM to intelligently filter course list and remove junk data (dates, empty values, totals).

    Args:
        raw_courses: Raw list from Course Title + Course Number columns
        student_name: Student name for context
        llm: LLM instance for filtering

    Returns:
        List of clean, valid course names/titles/codes
    """
    if not raw_courses or len(raw_courses) == 0:
        logger.warning(f"[CourseFilter] Empty input for {student_name}")
        return []

    logger.info(f"[CourseFilter] Raw input ({len(raw_courses)} items): {raw_courses[:10]}")

    # Quick heuristic filter first (cheap)
    candidates = []
    for course in raw_courses:
        course_str = str(course).strip()

        # Skip obvious junk
        if not course_str or course_str == ' ' or course_str == 'nan':
            continue
        if len(course_str) < 3:
            continue
        if re.match(r'^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$', course_str):
            logger.debug(f"[CourseFilter] Skipping date: {course_str}")
            continue
        if 'Total' in course_str and ':' in course_str:
            logger.debug(f"[CourseFilter] Skipping total: {course_str}")
            continue

        candidates.append(course_str)

    logger.info(f"[CourseFilter] After heuristic filter: {len(candidates)} candidates")

    if not candidates:
        logger.warning(f"[CourseFilter] No valid courses after heuristic for {student_name}")
        return []

    # If <= 3 candidates and they look reasonable, skip LLM
    if len(candidates) <= 3 and all(len(c) > 8 for c in candidates):
        logger.info(f"[CourseFilter] {len(candidates)} courses look valid, skipping LLM")
        return candidates

    # Use LLM for intelligent filtering
    logger.info(f"[CourseFilter] Invoking LLM to filter {len(candidates)} candidates")

    filter_prompt = f"""You are a data quality filter for student course records.

Student: {student_name}

Raw course data (may contain course titles, course codes, dates, or junk):
{candidates}

Task: Return ONLY the items that are actual courses (course titles, course names, or course codes).

Rules:
- Include: Course titles (e.g., "Introduction to Biology", "Business Mathematics")
- Include: Course codes (e.g., "MATH 101", "ENG-1010", "CS101")
- Exclude: Dates (e.g., "12/16/2024", "12-11-2024", "03-06-2025")
- Exclude: Empty strings, whitespace, or single spaces
- Exclude: Lines with "Total" or "Subterm"
- Exclude: Random numbers or gibberish

Output format: Return a Python list of valid courses ONLY.
Example: ["Introduction to Biology", "Business Mathematics", "MATH 101"]

If no valid courses found, return empty list: []

Respond with ONLY the Python list, nothing else."""

    try:
        from langchain_core.messages import HumanMessage

        response = llm.invoke([HumanMessage(content=filter_prompt)])
        response_text = response.content.strip()

        logger.info(f"[CourseFilter] LLM response: {response_text[:200]}")

        # Try to parse as Python list
        import ast
        try:
            filtered_courses = ast.literal_eval(response_text)
            if isinstance(filtered_courses, list):
                logger.info(f"[CourseFilter] LLM filtered to {len(filtered_courses)} valid courses")
                return filtered_courses
        except Exception as parse_err:
            logger.warning(f"[CourseFilter] ast.literal_eval failed: {parse_err}")

        # Fallback: extract items from response
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]
        filtered_courses = []
        for line in lines:
            clean = line.strip('- "\'[](),')
            if clean and not re.match(r'^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$', clean):
                filtered_courses.append(clean)

        logger.info(f"[CourseFilter] LLM filtered to {len(filtered_courses)} valid courses (fallback parsing)")
        return filtered_courses

    except Exception as e:
        logger.error(f"[CourseFilter] LLM filtering failed: {e}", exc_info=True)
        # Fallback: return candidates with basic filtering
        fallback = [c for c in candidates if len(c) > 8]
        logger.info(f"[CourseFilter] Using fallback: {len(fallback)} courses")
        return fallback


# ============================================================================
# ENHANCED AGENT PREFIX WITH OUTPUT FORMATTING & SPLIT-NAME MATCHING
# ============================================================================


AGENT_PREFIX_TEMPLATE = """
You are an EXPERT data analyst. A DataFrame 'df' is PRE-LOADED with student transcript data.

CRITICAL: DO NOT create new DataFrame. USE existing 'df' variable directly.

================================================================================
DATA: {total_rows} records | {total_students} students | {total_courses} courses | GPA: {gpa_min:.2f}-{gpa_max:.2f}
================================================================================

Sample Students: {sample_students_formatted}
Sample Courses: {sample_courses_formatted}
Available Terms: {available_terms_formatted}
Columns: {columns_list}

STRUCTURE: Each row = ONE course enrollment. Students appear multiple times. Aggregate to avoid double-counting.

================================================================================
QUERY: "{original_query}"
TOPIC: {query_topic}
================================================================================

CORE TECHNIQUES:

1. NAME MATCHING - USE REGEX (handles middle names):
   df['Student Name'].str.contains('Trista.*Barrett', case=False, regex=True, na=False)

   Why: .* wildcard matches any middle name.

2. COURSE DATA - CHECK BOTH COLUMNS AND FILTER JUNK (CRITICAL):
   
   Use NEWLINES between statements (not semicolons):
   
   student_data = df[df['Student Name'].str.contains('Joshua.*Gaitan', case=False, regex=True, na=False)]
   courses = student_data['Course Title'].dropna().unique().tolist()
   
   clean_courses = []
   for c in courses:
       if len(str(c).strip()) > 5 and 'Total' not in str(c):
           clean_courses.append(str(c).strip())
   
   student_name = student_data['Student Name'].iloc[0]
   print(f"**{{student_name}}** is enrolled in:")
   for c in clean_courses:
       print(f"- {{c}}")
   
   CRITICAL: Each statement on new line. NO semicolons.

3. GPA CALCULATIONS (filter > 0):
   df[df['Student Name'].str.contains('Trista.*Barrett', case=False, regex=True, na=False) & (df['GPA'] > 0)]['GPA'].max()

4. RANKING:
   df[df['GPA'] > 0].groupby('Student Name')['GPA'].max().sort_values(ascending=False).head(5)

================================================================================
OUTPUT FORMATTING - PROFESSIONAL MARKDOWN (PHASE 3)
================================================================================

ALWAYS format output with Markdown for professional presentation:

For GPA queries:
WRONG: "4.0"
RIGHT: "The GPA for **Trista Denay Barrett** is **4.0**"

For course lists:
RIGHT:
"**Joshua Don Gaitan** is enrolled in:
- Introduction to Biology
- Elementary Algebra  
- College Writing I
- Biblical Literature"

For rankings:
RIGHT:
"Top 5 Students by GPA:

1. **Student Name** - GPA: **4.0**
2. **Student Name** - GPA: **3.8**
3. **Student Name** - GPA: **3.7**"

For counts:
RIGHT: "There are **15** students with GPA above 3.5"

For tables (when showing multiple students):
RIGHT:
| Student Name | GPA |
|--------------|-----|
| **Trista Barrett** | **4.0** |
| **Leslie Bright** | **3.8** |

FORMATTING RULES:
- Use **bold** for names, numbers, and key facts
- Use bullet points (-) for lists  
- Use tables when showing 3+ items with multiple attributes
- NO raw Python lists or DataFrames
- NO phrases like "According to" or "Based on"

================================================================================
CRITICAL: STOP AFTER SUCCESS (PHASE 4 FIX)
================================================================================

If your code executes successfully and produces the correct output:
1. Format the output with Markdown (bold, bullets, etc.)
2. Return your formatted answer IMMEDIATELY
3. Do NOT re-run the same code
4. Do NOT try alternative approaches

WRONG (repeating unnecessarily):
Observation: **Trista Barrett** is enrolled in:
- Course 1
- Course 2

Thought: Let me try again...  ← DO NOT DO THIS

RIGHT (stop after success):
Observation: **Trista Barrett** is enrolled in:
- Course 1
- Course 2

Thought: I have the complete answer formatted correctly.
Final Answer: **Trista Barrett** is enrolled in:
- Course 1
- Course 2

================================================================================
EXECUTION RULES
================================================================================

1. Use regex for name matching: 'First.*Last'
2. For courses: Check BOTH Course Title AND Course Number
3. FILTER OUT: dates (contain / or -), "Total" strings, empty values, short strings
4. Format output with Markdown (bold, bullets, tables)
5. Execute code ONCE, get result, format professionally, provide Final Answer, STOP

Query hints: {query_hints}

Query hints: {query_hints}

================================================================================
FINAL INSTRUCTION - READ CAREFULLY
================================================================================

1. Write ONE piece of Python code to answer the query
2. Execute it ONCE using python_repl_ast
3. If the output looks correct (list of items, number, name, etc.):
   - Format it with Markdown
   - Provide your Final Answer
   - STOP IMMEDIATELY - DO NOT RUN MORE CODE

For course queries: Filter out dates (12/16/2024) and "Total" strings BEFORE responding.

WRONG PATTERN (DO NOT DO THIS):
Observation: [correct output]
Thought: Let me try another approach...  ← STOP! Output was already correct

RIGHT PATTERN:
Observation: [correct output]
Thought: This is the correct answer, formatting now
Final Answer: [formatted output]

Begin solving now.
"""

# ============================================================================
# LLM CREATION
# ============================================================================


def _create_llm():
    """Create LLM using centralized factory with key rotation."""
    return get_llm(temperature=0)


# ============================================================================
# MAIN ANSWER FUNCTION
# ============================================================================


def answer(query: str, params: Dict[str, Any] = None) -> ToolResult:
    """
    Universal CSV Agent for ALL transcript queries with dynamic context + smart filtering.
    PHASE 4: Fixed agent iteration loops.
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

        # STEP 3: Build agent prefix
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
            original_query=query,
            query_topic=query_topic
        )

        # STEP 4: Create LLM
        llm = _create_llm()

        logger.info("[TranscriptTool] Creating pandas agent with optimized stopping...")

        # STEP 5: Create agent with STRICT stopping (PHASE 4 FIX)
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            prefix=agent_prefix,
            verbose=True,
            agent_type="openai-tools",
            allow_dangerous_code=True,
            max_iterations=3,  # REDUCED from 5 - force early stop
            max_execution_time=20,  # REDUCED from 30
            early_stopping_method="force"  # Force stop on first success
        )

        logger.info(f"[TranscriptTool] Executing agent with max 3 iterations...")

        # STEP 6: Execute with explicit stop instruction
        enhanced_query = f"""{query}

            IMPORTANT: 
            - Write Python code with NEWLINES (not semicolons)
            - Execute ONCE
            - Format output with Markdown
            - STOP after first execution"""

        result = agent.invoke({"input": enhanced_query})

        answer_text = result.get("output", "Unable to process query")
        # PHASE 4.1: Strip "Final Answer:" prefix if present
        # Strip "Final Answer:" prefix (appears in agent output)
        if "Final Answer:" in answer_text:
            # Split on "Final Answer:" and take the last part (the actual answer)
            parts = answer_text.split("Final Answer:")
            answer_text = parts[-1].strip()

        logger.info(f"[TranscriptTool] Agent output: {answer_text[:300]}")

        # STEP 7: POST-PROCESSING - Smart course filtering for course queries
        if 'course' in query.lower() and ('enrolled' in query.lower() or 'taken' in query.lower() or 'taking' in query.lower() or 'which' in query.lower()):

            # Detect if answer contains list-like pattern (Python list or dates)
            has_python_list = re.search(r'\[[\'"]', answer_text)  # Looks for ["..." or ['...
            has_dates = re.search(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', answer_text)
            has_totals = 'Total' in answer_text and ':' in answer_text

            logger.info(f"[POST-PROCESS] Course query - has_python_list={bool(has_python_list)}, has_dates={bool(has_dates)}, has_totals={bool(has_totals)}")

            if has_python_list or has_dates or has_totals:
                logger.warning("[TranscriptTool] Detected unformatted/junk course data - applying smart filter")

                # Extract student name from query (handle multiple patterns)
                name_match = re.search(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b', query)
                student_name = name_match.group(1) if name_match else "Student"

                logger.info(f"[POST-PROCESS] Extracted student name: '{student_name}'")

                # Use split-name matching
                name_parts = student_name.split()
                student_filter = df['Student Name'].str.contains(name_parts[0], case=False, na=False)
                for part in name_parts[1:]:
                    student_filter = student_filter & df['Student Name'].str.contains(part, case=False, na=False)

                student_rows = df[student_filter]

                logger.info(f"[POST-PROCESS] Found {len(student_rows)} rows for student")

                if len(student_rows) > 0:
                    # Collect from BOTH columns
                    courses_title = student_rows['Course Title'].dropna().unique().tolist()
                    courses_number = student_rows['Course Number'].dropna().unique().tolist() if 'Course Number' in df.columns else []

                    logger.info(f"[POST-PROCESS] From Course Title: {len(courses_title)} items")
                    logger.info(f"[POST-PROCESS] From Course Number: {len(courses_number)} items")

                    raw_courses = list(set(courses_title + courses_number))

                    logger.info(f"[POST-PROCESS] Combined raw courses: {len(raw_courses)} items")

                    # Apply LLM filter
                    filtered_courses = _filter_courses_with_llm(raw_courses, student_name, llm)

                    if filtered_courses:
                        # PHASE 4.1: Deduplicate courses (preserve order)
                        seen = set()
                        unique_courses = []
                        for course in filtered_courses:
                            if course not in seen:
                                seen.add(course)
                                unique_courses.append(course)
                        
                        actual_student_name = student_rows['Student Name'].iloc[0]
                        answer_text = f"**{actual_student_name}** is enrolled in:\n" + "\n".join([f"- {c}" for c in unique_courses])
                        logger.info(f"[TranscriptTool] Applied smart filter: {len(unique_courses)} unique courses (from {len(filtered_courses)} raw)")

                    else:
                        answer_text = f"No valid course enrollment records found for {student_name}."
                        logger.warning(f"[TranscriptTool] No valid courses after filtering")

        logger.info(f"[TranscriptTool] Final answer: {answer_text[:200]}...")

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
    course_num_col = 'Course Number'

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
                    explanation=f"The GPA for **{student_name}** is **{career_gpa:.2f}**",
                    confidence=0.90,
                    format_hint="text",
                    citations=["merged_transcripts.csv"]
                )

    # 2. COUNT STUDENTS
    if "how many student" in query_lower:
        count = df[name_col].nunique()
        return ToolResult(
            data={"count": count},
            explanation=f"There are **{count}** unique students in the transcript data.",
            confidence=0.95,
            format_hint="text",
            citations=["merged_transcripts.csv"]
        )

    # 3. LIST/RANK STUDENTS
    if any(kw in query_lower for kw in ['list', 'top', 'highest', 'rank', 'show me']):
        student_gpas = df[df[gpa_col] > 0].groupby(name_col)[gpa_col].max().reset_index()
        student_gpas = student_gpas.sort_values(gpa_col, ascending=False)

        top_n = 5 if 'top 5' in query_lower else 10

        result_text = f"### Top {min(top_n, len(student_gpas))} Students by GPA\n\n"
        for i, (_, row) in enumerate(student_gpas.head(top_n).iterrows(), 1):
            result_text += f"{i}. **{row[name_col]}** - GPA: **{row[gpa_col]:.2f}**\n"

        return ToolResult(
            data={"students": student_gpas.head(top_n).to_dict('records')},
            explanation=result_text,
            confidence=0.90,
            format_hint="text",
            citations=["merged_transcripts.csv"]
        )

    # 4. COURSE ENROLLMENT (with smart filtering)
    if 'course' in query_lower and ('enrolled' in query_lower or 'taken' in query_lower):
        name_match = re.search(r'\b([A-Z][a-z]+)\b', query)
        if name_match:
            search_name = name_match.group(1)
            matches = df[df[name_col].str.contains(search_name, case=False, na=False)]

            if len(matches) > 0:
                student_name = matches[name_col].iloc[0]

                # Get courses from BOTH columns
                courses_title = matches[course_col].dropna().unique().tolist()
                courses_number = matches[course_num_col].dropna().unique().tolist() if course_num_col in df.columns else []
                raw_courses = list(set(courses_title + courses_number))

                # Filter out dates and junk
                clean_courses = []
                for course in raw_courses:
                    course_str = str(course).strip()
                    if course_str and len(course_str) > 3 and not re.match(r'^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$', course_str):
                        clean_courses.append(course_str)

                if clean_courses:
                    result_text = f"**{student_name}** is enrolled in:\n\n"
                    for course in clean_courses:
                        result_text += f"- {course}\n"

                    return ToolResult(
                        data={"student": student_name, "courses": clean_courses},
                        explanation=result_text,
                        confidence=0.85,
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
