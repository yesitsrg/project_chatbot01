# services/tools/transcript_tool.py

"""
TranscriptTool - Student transcript Q&A with strict matching and aggregations.

Features:
- Single student queries (GPA, courses, academic info)
- Aggregate queries (top students, averages, counts)
- Strict name matching (exact > first+last > last-only)
- Bounded fallback to dataframe agent (max_iterations=5)
"""

from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import re

from core import get_logger
from services.data_views import get_transcript_df
from models.schemas import ToolResult
from services.df_agent import run_df_agent

logger = get_logger(__name__)


def _find_col(df: pd.DataFrame, target: str) -> Optional[str]:
    """Find a column whose normalized name matches target exactly or contains it."""
    cols = list(df.columns)
    target_l = target.strip().lower()

    # Exact match first
    for c in cols:
        if c.strip().lower() == target_l:
            return c
    # Partial match second
    for c in cols:
        cl = c.strip().lower()
        if target_l in cl:
            return c
    return None


def _find_student_name_col(df: pd.DataFrame) -> Optional[str]:
    """Find the student name column."""
    cols = list(df.columns)
    for c in cols:
        if c.strip().lower() == "student name":
            return c
    for c in cols:
        cl = c.strip().lower()
        if "student" in cl and "name" in cl:
            return c
    return None


def _find_gpa_col(df: pd.DataFrame) -> Optional[str]:
    return _find_col(df, "gpa")


def _find_quality_points_col(df: pd.DataFrame) -> Optional[str]:
    return _find_col(df, "quality points")


def _find_hours_gpa_col(df: pd.DataFrame) -> Optional[str]:
    return _find_col(df, "hours gpa")


def _compute_student_gpa(
    sdf: pd.DataFrame,
    gpa_col: Optional[str],
    qp_col: Optional[str],
    hrs_col: Optional[str],
) -> Tuple[Optional[float], str]:
    """
    Compute a student's GPA from all their transcript rows.

    Priority:
    1) If a non-null GPA value exists in gpa_col, use its mean.
    2) Else, if quality points & hours-gpa present, compute sum(QP)/sum(Hours).
    3) Else, return None.
    """
    # 1) Direct GPA column
    if gpa_col and gpa_col in sdf.columns:
        gpa_series = pd.to_numeric(sdf[gpa_col], errors="coerce")
        gpa_series = gpa_series.dropna()
        if not gpa_series.empty:
            return float(gpa_series.mean()), "from GPA column"

    # 2) Derived from quality points / hours
    if qp_col and hrs_col and qp_col in sdf.columns and hrs_col in sdf.columns:
        qp = pd.to_numeric(sdf[qp_col], errors="coerce")
        hrs = pd.to_numeric(sdf[hrs_col], errors="coerce")
        mask = (hrs > 0) & qp.notna()
        if mask.any():
            total_qp = float(qp[mask].sum())
            total_hrs = float(hrs[mask].sum())
            if total_hrs > 0:
                return total_qp / total_hrs, "from Quality Points / Hours GPA"

    return None, "no numeric GPA available"


def _normalize_name(s: str) -> str:
    """Normalize name for comparison."""
    return " ".join(s.strip().lower().split())


def _filter_student(df: pd.DataFrame, name_col: str, student_name: str) -> pd.DataFrame:
    """
    Robust student matching with precedence:
    1. Exact full-name match
    2. First+Last name match
    3. Last-name only match
    
    Returns empty df if no match found.
    """
    target = _normalize_name(student_name)

    # 1. Exact full-name contains
    mask = df[name_col].astype(str).apply(_normalize_name).str.contains(target, na=False, regex=False)
    sdf = df[mask]

    if not sdf.empty:
        return sdf

    # 2. Last-name only fallback
    parts = target.split()
    if len(parts) >= 2:
        last_name = parts[-1]
        mask = df[name_col].astype(str).str.lower().str.contains(last_name, na=False, regex=False)
        sdf = df[mask]
        if not sdf.empty:
            return sdf

    return sdf  # Empty


def _build_courses_table(sdf: pd.DataFrame) -> List[Dict[str, Any]]:
    """Extract and format course information."""
    cols = list(sdf.columns)
    course_num_col = None
    course_title_col = None
    term_col = None
    subterm_col = None
    grade_col = None

    for c in cols:
        cl = c.strip().lower()
        if "course number" == cl or ("course" in cl and "number" in cl):
            course_num_col = c
        if "course title" == cl or ("course" in cl and "title" in cl):
            course_title_col = c
        if cl == "term":
            term_col = c
        if "subterm" in cl:
            subterm_col = c
        if cl == "grade":
            grade_col = c

    subset_cols = [
        c for c in [course_num_col, course_title_col, term_col, subterm_col, grade_col] if c
    ]
    if not subset_cols:
        return []

    subset = sdf[subset_cols].drop_duplicates()
    return subset.to_dict("records")


def answer(question: str, params: Dict[str, Any] | None = None) -> ToolResult:
    """Main TranscriptTool entry point."""
    df = get_transcript_df()
    if df.empty:
        return ToolResult(
            data={},
            explanation="No transcript data available.",
            confidence=0.0,
            format_hint="text",
            citations=[],
        )

    params = params or {}
    student_name = params.get("student_name")
    format_hint = params.get("format_hint", "text")

    name_col = _find_student_name_col(df)
    gpa_col = _find_gpa_col(df)
    qp_col = _find_quality_points_col(df)
    hrs_col = _find_hours_gpa_col(df)

    if not name_col:
        return ToolResult(
            data={},
            explanation="Transcript data is loaded but no student name column could be identified.",
            confidence=0.4,
            format_hint="text",
            citations=[],
        )

    q_lower = question.lower()

    # =====================================================================
    # SINGLE-STUDENT QUESTIONS
    # =====================================================================
    if student_name:
        sdf = _filter_student(df, name_col, student_name)
        if sdf.empty:
            return ToolResult(
                data={},
                explanation=f"No transcript records found for '{student_name}'.",
                confidence=0.5,
                format_hint="text",
                citations=["merged_transcripts.csv"],
            )

        # Compute GPA from all rows for this student
        gpa_val, gpa_source = _compute_student_gpa(sdf, gpa_col, qp_col, hrs_col)
        display_name = str(sdf.iloc[0][name_col])
        courses = _build_courses_table(sdf)

        # Course-specific question
        if "course" in q_lower and ("which" in q_lower or "what" in q_lower):
            if not courses:
                explanation = f"Courses for {display_name} could not be determined from the transcript data."
            else:
                lines = [
                    f"- {c.get('Course Number', '')} {c.get('Course Title', '')} "
                    f"({c.get('Term', '')} {c.get('Subterm', '')})"
                    for c in courses
                ]
                explanation = f"Courses which {display_name} has enrolled:\n" + "\n".join(lines)
            return ToolResult(
                data={"student": display_name, "courses": courses},
                explanation=explanation,
                confidence=0.9,
                format_hint="text",
                citations=["merged_transcripts.csv"],
            )

        # GPA-specific question
        if "gpa" in q_lower:
            if gpa_val is None:
                explanation = f"GPA details for {display_name}: GPA is not available in the transcript data."
            else:
                explanation = f"GPA details for {display_name}: GPA = {gpa_val:.3f} ({gpa_source})."
            return ToolResult(
                data={"student": display_name, "gpa": gpa_val, "gpa_source": gpa_source},
                explanation=explanation,
                confidence=0.9,
                format_hint="text",
                citations=["merged_transcripts.csv"],
            )

        # Generic academic information
        if any(
            key in q_lower
            for key in [
                "academic information",
                "grades",
                "performance",
                "transcript",
                "enrollment history",
                "course completion",
            ]
        ):
            total_attempted = None
            total_earned = None
            hours_attempted_col = _find_col(df, "hours attempted")
            hours_earned_col = _find_col(df, "hours earned")
            if hours_attempted_col and hours_attempted_col in sdf.columns:
                total_attempted = float(
                    pd.to_numeric(sdf[hours_attempted_col], errors="coerce").fillna(0).sum()
                )
            if hours_earned_col and hours_earned_col in sdf.columns:
                total_earned = float(
                    pd.to_numeric(sdf[hours_earned_col], errors="coerce").fillna(0).sum()
                )

            explanation_parts = [f"Academic information for {display_name}:"]
            if gpa_val is not None:
                explanation_parts.append(f"- GPA: {gpa_val:.3f} ({gpa_source})")
            if total_attempted is not None:
                explanation_parts.append(f"- Credit hours attempted: {total_attempted:.1f}")
            if total_earned is not None:
                explanation_parts.append(f"- Credit hours earned: {total_earned:.1f}")
            if courses:
                explanation_parts.append(f"- Courses taken: {len(courses)}")

            explanation = "\n".join(explanation_parts)
            return ToolResult(
                data={
                    "student": display_name,
                    "gpa": gpa_val,
                    "courses": courses,
                    "hours_attempted": total_attempted,
                    "hours_earned": total_earned,
                },
                explanation=explanation,
                confidence=0.9,
                format_hint="text",
                citations=["merged_transcripts.csv"],
            )

        # Default per-student fallback
        explanation = f"Transcript records found for {display_name}."
        return ToolResult(
            data={"student": display_name, "rows": sdf.to_dict("records")},
            explanation=explanation,
            confidence=0.8,
            format_hint="table",
            citations=["merged_transcripts.csv"],
        )

    # =====================================================================
    # AGGREGATE QUESTIONS (no specific student_name)
    # =====================================================================

    # Precompute per-student GPA table for aggregate queries
    student_groups = df.groupby(name_col)

    def build_student_gpa_df() -> pd.DataFrame:
        records = []
        for sname, sdf in student_groups:
            g, src = _compute_student_gpa(sdf, gpa_col, qp_col, hrs_col)
            records.append({"Student Name": sname, "GPA": g, "gpa_source": src})
        return pd.DataFrame(records)

    # Sort students in descending order of GPA
    if "sort" in q_lower and "gpa" in q_lower and "descending" in q_lower:
        sg = build_student_gpa_df()
        sg = sg.dropna(subset=["GPA"]).sort_values("GPA", ascending=False)
        explanation = (
            "Students sorted in descending order of GPA (showing top 20):\n"
            + sg.head(20)[["Student Name", "GPA"]].to_markdown(index=False)
        )
        return ToolResult(
            data={"rows": sg.to_dict("records")},
            explanation=explanation,
            confidence=0.9,
            format_hint="table",
            citations=["merged_transcripts.csv"],
        )

    # Count students with GPA >= threshold
    if "how many" in q_lower and "gpa" in q_lower and ">=" in q_lower:
        m = re.search(r"gpa\s*>=\s*([0-9]+(\.[0-9]+)?)", q_lower)
        threshold = None
        if m:
            threshold = float(m.group(1))
        if threshold is not None:
            sg = build_student_gpa_df()
            mask = sg["GPA"].notna() & (sg["GPA"] >= threshold)
            filtered = sg[mask]
            count = len(filtered)
            explanation = f"{count} students have GPA >= {threshold}."
            return ToolResult(
                data={"threshold": threshold, "count": count, "rows": filtered.to_dict("records")},
                explanation=explanation,
                confidence=0.9,
                format_hint="table",
                citations=["merged_transcripts.csv"],
            )

    # Average GPA of students
    if "average gpa" in q_lower or "avg gpa" in q_lower:
        sg = build_student_gpa_df()
        valid = sg["GPA"].dropna()
        if valid.empty:
            explanation = "Average GPA cannot be computed because no numeric GPA values are available."
            return ToolResult(
                data={},
                explanation=explanation,
                confidence=0.5,
                format_hint="text",
                citations=["merged_transcripts.csv"],
            )
        avg_gpa = float(valid.mean())
        explanation = f"The average GPA across students is {avg_gpa:.3f}."
        return ToolResult(
            data={"average_gpa": avg_gpa},
            explanation=explanation,
            confidence=0.9,
            format_hint="text",
            citations=["merged_transcripts.csv"],
        )

    # Generic "how many students"
    if "how many" in q_lower and "student" in q_lower:
        count = df[name_col].dropna().nunique()
        explanation = f"There are {count} unique students in the transcript data."
        return ToolResult(
            data={"student_count": count},
            explanation=explanation,
            confidence=0.9,
            format_hint="text",
            citations=["merged_transcripts.csv"],
        )

    # =====================================================================
    # FALLBACK → DATAFRAME AGENT (WITH SAFETY BOUNDS)
    # =====================================================================
    # Only invoke if no explicit logic matched
    # df_agent now has max_iterations=5 to prevent infinite loops
    df_answer = run_df_agent(question, df, df_name="transcript records", max_iterations=5)
    return ToolResult(
        data={"raw": df_answer.get("raw")},
        explanation=df_answer["answer"],
        confidence=0.75,
        format_hint="text",
        citations=["merged_transcripts.csv"],
    )
