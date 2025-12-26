# services/orchestrator.py
# ENTERPRISE-GRADE HYBRID ORCHESTRATOR (4-LAYER)
# Fast Rules + LLM Descriptions + Feedback Loop + Safe Fallback
# BACKWARD COMPATIBLE with existing Orchestrator interface
# LAYER 2 LLM ENABLED via RAGPipeline

from typing import Dict, Any, List, Tuple, Optional
import json
import re
import logging
from datetime import datetime
from collections import defaultdict

from pydantic import BaseModel

from core import get_logger
from config import get_settings
from services.tools.registry import get_tool_descriptions
from services.tools.transcript_tool import answer as transcript_answer
from services.tools.payroll_tool import answer as payroll_answer
from services.tools.bor_planner_tool import answer as bor_answer
from services.tools.generic_rag_tool import answer as rag_answer
from models.schemas import ToolResult
from services.rag_pipeline import RAGPipeline

logger = get_logger(__name__)
settings = get_settings()

TOOL_MAP = {
    "TranscriptTool": transcript_answer,
    "PayrollTool": payroll_answer,
    "BorPlannerTool": bor_answer,
    "GenericRagTool": rag_answer,
}

# =============================================================================
# LAYER 1: FAST RULE-BASED ROUTING (< 1ms)
# =============================================================================

LAYER1_PATTERNS = {
"TranscriptTool": {
    "phrases": [
        # Count queries
        "how many students",
        "number of students",
        "total students",
        "student count",
        
        # Top/ranking queries
        "top students",
        "top student",
        "highest gpa",
        "lowest gpa",
        "best students",
        "rank students",
        "students by gpa",
        
        # Course queries
        "what courses",
        "which courses",
        "courses enrolled",
        "student courses",
        "enrolled in",
        
        # GPA queries
        "what is the gpa",
        "what is gpa",
        "gpa for",
        "gpa of",
        "student gpa",
        "career gpa",
        
        # Display/Show/List by metric
        "display students by gpa",
        "display student by gpa",
        "show students by gpa",
    ],
    
    "patterns": [  # ✅ CHANGED FROM "regex_patterns"
        # Count patterns
        r'how\s+many\s+students',
        r'number\s+of\s+students',
        
        # Top N patterns
        r'top\s+\d+\s*students',
        r'(?:highest|lowest|best|worst)\s+(?:gpa|students)',
        r'students.*?by.*?gpa',
        
        # Course patterns
        r'what\s+courses.*?(?:enrolled|taking)',
        r'courses.*?(?:for|of)\s+([A-Z][a-z]+)',
        
        # GPA patterns
        r'(?:gpa|grade).*?(?:for|of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
    ],
    
    "min_confidence": 0.75,
    
    "negative_patterns": [
        r'payroll',
        r'salary',
        r'HR',
        r'catalog',
        r'policy',
        r'handbook',
        r'fee',
        r'refund',
    ]
},
    "PayrollTool": {
        "phrases": {
            "pay period", "payroll period", "payroll number",
            "check date", "payroll calendar", "payroll schedule",
            "payroll no", "payroll p"
        },
        "patterns": [
            r"p0\d",  # p01, p02, etc
            r"period\s+\d+",
            r"p\d{2}"
        ],
        "confidence_threshold": 0.80
    },
    "BorPlannerTool": {
        "phrases": {
            "board of regents", "board meeting", "bor meeting",
            "board of regents meeting", "regents meeting",
            "finance committee", "governance committee",
            "acct-nls", "board planner"
        },
        "confidence_threshold": 0.85
    }
}

# ============================================================================
# ENTITY DETECTION HELPERS (Add before layer1_route)
# ============================================================================

def _has_student_name(query: str) -> bool:
    """
    Detect if query contains a student name (capitalized multi-word pattern).
    
    Examples:
        "What is the GPA of Leslie Nichole Bright?" → True
        "What courses is Leslie enrolled in?" → True (first name + context)
        "How many students enrolled?" → False
    """
    # Pattern 1: Full name (First Middle Last or First Last)
    pattern_full = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b'
    match = re.search(pattern_full, query)
    
    if match:
        candidate = match.group(1)
        # Filter out common false positives
        false_positives = {
            "Fall", "Spring", "Summer", "Winter",
            "Board Of Regents", "Din College", "Newman University",
            "Murray State", "Pay Period", "Payroll Calendar"
        }
        if candidate not in false_positives:
            logger.debug(f"[Entity Detection] Found full name: '{candidate}'")
            return True
    
    # Pattern 2: First name only with context keywords
    # "What courses is Leslie enrolled in?"
    # "Show me grades for John"
    if re.search(r'\b([A-Z][a-z]{3,})\s+(?:enrolled|taking|courses|grades?|gpa|transcript)', query):
        logger.debug(f"[Entity Detection] Found first name with context")
        return True
    
    if re.search(r'(?:courses?|grades?|gpa|transcript).*?(?:for|of)\s+([A-Z][a-z]{3,})\b', query):
        logger.debug(f"[Entity Detection] Found first name with context (reverse)")
        return True
    
    return False

def _has_payroll_period(query: str) -> bool:
    """
    Detect payroll period references.
    
    Examples:
        "check date for pay period 3" --> True
        "payroll p05" --> True
    """
    patterns = [
        r'pay\s*period\s*\d+',
        r'payroll\s*(?:period)?\s*\d+',
        r'\bp\d{2}\b',
        r'period\s*\d+.*check\s*date',
    ]
    return any(re.search(p, query.lower()) for p in patterns)

def layer1_route(query: str) -> Tuple[Optional[str], float]:
    """
    Layer 1: Fast rule-based routing with ENTITY DETECTION (< 1ms)
    
    Priority:
    1. Entity detection (student names, payroll periods) - HIGHEST
    2. Intent keywords (what is, explain, policy)
    3. Domain keywords (existing patterns)
    """
    q_norm = query.lower()
    
    # =========================================================================
    # PRIORITY 1: ENTITY DETECTION (New - Solves enrollment ambiguity)
    # =========================================================================
    
    if _has_student_name(query):
        logger.info("[Layer 1] Entity detected: STUDENT NAME --> TranscriptTool (confidence=0.95)")
        return ("TranscriptTool", 0.95)
    
    if _has_payroll_period(query):
        logger.info("[Layer 1] Entity detected: PAYROLL PERIOD --> PayrollTool (confidence=0.95)")
        return ("PayrollTool", 0.95)
    
    # =========================================================================
    # PRIORITY 2: INTENT DETECTION (New - Policy vs Data queries)
    # =========================================================================
    
    # Policy/informational intent
    policy_keywords = ["what is", "explain", "policy", "procedure", "how to", "process for"]
    if any(kw in q_norm for kw in policy_keywords):
        # Exclude if also has data query markers
        data_markers = ["how many", "list", "show", "display", "which students"]
        if not any(dm in q_norm for dm in data_markers):
            logger.info("[Layer 1] Intent detected: POLICY --> GenericRagTool (confidence=0.85)")
            return ("GenericRagTool", 0.85)
    
    # =========================================================================
    # PRIORITY 3: DOMAIN KEYWORD MATCHING (FIXED)
    # =========================================================================
    
    for tool, config in LAYER1_PATTERNS.items():
        match_score = 0
        total_possible = 0
        
        # 1. Check phrases
        phrases = config.get("phrases", [])
        if isinstance(phrases, set):
            phrases = list(phrases)
        
        for phrase in phrases:
            total_possible += 2  # Phrases worth 2 points
            if phrase.lower() in q_norm:
                match_score += 2
                logger.debug(f"[Layer 1 {tool}] ✅ Phrase: '{phrase}'")
        
        # 2. Check keywords
        keywords = config.get("keywords", [])
        if isinstance(keywords, set):
            keywords = list(keywords)
        
        for keyword in keywords:
            total_possible += 1  # Keywords worth 1 point
            if keyword.lower() in q_norm:
                match_score += 1
                logger.debug(f"[Layer 1 {tool}] ✅ Keyword: '{keyword}'")
        
        # 3. Check regex patterns
        patterns = config.get("patterns", [])
        for pattern in patterns:
            total_possible += 2  # Patterns worth 2 points
            try:
                if re.search(pattern, q_norm, re.IGNORECASE):
                    match_score += 2
                    logger.debug(f"[Layer 1 {tool}] ✅ Pattern: {pattern}")
            except Exception as e:
                logger.error(f"[Layer 1 {tool}] Bad regex: {pattern} - {e}")
        
        # Calculate confidence
        if total_possible > 0:
            confidence = match_score / total_possible
            threshold = config.get("min_confidence", config.get("confidence_threshold", 0.75))
            
            logger.debug(
                f"[Layer 1 {tool}] Score: {match_score}/{total_possible} = "
                f"{confidence:.2f} (need {threshold})"
            )
            
            if confidence >= threshold:
                logger.info(
                    f"[Layer 1] Match: {tool} (confidence={confidence:.2f})"
                )
                return (tool, confidence)
    
    logger.debug("[Layer 1] No high-confidence match. Falling to Layer 2")
    return (None, 0.0)

# =============================================================================
# LAYER 2: LLM WITH TOOL DESCRIPTIONS (100-200ms)
# =============================================================================

TOOL_DESCRIPTIONS = {
    "TranscriptTool": """
Tool: Student Transcript & Academic Records

Purpose: Answers questions about individual student academic records,
transcripts, performance, grades, courses, GPA, academic standing.

Handles:
- Student academic performance and grades
- Transcript records and course history
- GPA (cumulative and term-based)
- Courses taken, passed, failed
- Academic standing and progress
- Student enrollment status
- Student academic information

Example intent keywords:
- Student performance, academic records, GPA, transcript
- Courses, grades, academic standing
- Student information, enrollment status

Do NOT use for:
- Payroll/salary/payments --> PayrollTool
- Meeting schedules/calendars --> BorPlannerTool
- Policies/procedures/handbook --> GenericRagTool
""",

    "PayrollTool": """
Tool: Payroll Calendar & Pay Schedule

Purpose: Answers questions about payroll calendar, pay periods,
check dates, payroll schedules, payment processing information.

Handles:
- Payroll calendar and pay periods
- Check dates for specific pay periods
- Payroll schedule information
- Payment processing dates
- Payroll period details

Example intent keywords:
- Payroll, pay period, check date, payroll schedule
- Payment date, payroll calendar, payment schedule

Do NOT use for:
- Student records --> TranscriptTool
- Meeting schedules --> BorPlannerTool
- Policies --> GenericRagTool
""",

    "BorPlannerTool": """
Tool: Board of Regents Meeting Schedule

Purpose: Answers questions about Board of Regents meetings,
committee meetings, board schedules, governance events.

Handles:
- Board of Regents meeting schedules
- Committee meeting information
- Board calendar and dates
- Governance event schedules

Example intent keywords:
- Board of Regents, board meeting, BOR, committee meeting
- Finance committee, governance, ACCT-NLS

Do NOT use for:
- Student records --> TranscriptTool
- Payroll --> PayrollTool
- Policies --> GenericRagTool
""",

    "GenericRagTool": """
Tool: Institutional Policies, Handbook & General Information

Purpose: Answers questions about institutional policies, procedures,
student handbook, enrollment, benefits, housing, and all general
information NOT covered by specialized tools.

Handles:
- Student handbook and policies
- Enrollment procedures and process
- Housing and residence hall policies
- Institutional benefits and programs
- Academic calendar and break dates
- Health and wellness information
- Travel and reimbursement policies
- Code of conduct and conduct policies
- General institutional information

This is the FALLBACK tool for anything not in other tools.
Use when uncertain what else applies.

Example intent keywords:
- Policy, handbook, procedure, enrollment, benefits
- Housing, health, wellness, calendar, code of conduct
"""
}

LLM_ROUTING_PROMPT = """
You are a query router for a college information system.

User Query: "{query}"

Available Tools and Their Purpose:
{tool_descriptions}

Task: Determine which tool is MOST appropriate for this query.

Output JSON:
{{
    "tool": "ToolName or UNCERTAIN",
    "confidence": 0.75,
    "reasoning": "Brief explanation"
}}

Rules:
1. Be conservative: Only pick a tool if confident (>= 0.65)
2. If confidence < 0.65, respond with UNCERTAIN
3. If query needs multiple tools, pick the PRIMARY one
4. GenericRagTool is catch-all for everything else
5. Never make up tools - only use provided tools

Respond with ONLY the JSON."""

def layer2_route(query: str, rag_pipeline) -> Tuple[Optional[str], float]:
    """
    Layer 2: LLM-based routing with tool descriptions (100-200ms)
    
    Uses LLM via RAGPipeline to understand query intent based on tool purposes,
    not examples. Works with any phrasing.
    """
    
    tool_descs = "\n\n".join([
        f"### {tool}\n{desc}"
        for tool, desc in TOOL_DESCRIPTIONS.items()
    ])
    
    prompt = LLM_ROUTING_PROMPT.format(
        query=query,
        tool_descriptions=tool_descs
    )
    
    try:
        # Use RAGPipeline's LLM client to generate routing decision
        response = rag_pipeline._generate_answer(
            context="",
            question=prompt
        )
        
        result = json.loads(response)
        
        tool = result.get("tool")
        confidence = float(result.get("confidence", 0.0))
        
        if tool == "UNCERTAIN" or confidence < 0.5:
            logger.debug(
                f"[Layer 2] Uncertain (confidence={confidence:.2f})"
            )
            return (None, 0.0)
        
        logger.info(
            f"[Layer 2] {tool} (confidence={confidence:.2f})"
        )
        return (tool, confidence)
        
    except Exception as e:
        logger.error(f"[Layer 2] LLM routing failed: {e}")
        return (None, 0.0)


# =============================================================================
# LAYER 3: FEEDBACK TRACKING & LEARNING
# =============================================================================

class RoutingFeedback:
    """Track routing decisions to improve over time."""
    
    def __init__(self):
        self.routing_log = []
        self.accuracy_by_tool = {
            tool: {"correct": 0, "total": 0}
            for tool in TOOL_MAP.keys()
        }
        self.layer1_stats = {"hits": 0, "total": 0}
    
    def log_decision(
        self,
        query: str,
        routed_tool: str,
        confidence: float,
        layer: int,
        actual_tool: Optional[str] = None,
        user_satisfied: Optional[bool] = None
    ):
        """Log routing decision."""
        
        entry = {
            "timestamp": datetime.now(),
            "query": query,
            "routed_tool": routed_tool,
            "confidence": confidence,
            "layer": layer,
            "actual_tool": actual_tool,
            "user_satisfied": user_satisfied,
        }
        
        self.routing_log.append(entry)
        
        if layer == 1:
            self.layer1_stats["total"] += 1
            if actual_tool == routed_tool:
                self.layer1_stats["hits"] += 1
        
        if actual_tool:
            self.accuracy_by_tool[routed_tool]["total"] += 1
            if routed_tool == actual_tool:
                self.accuracy_by_tool[routed_tool]["correct"] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get routing performance metrics."""
        
        metrics = {
            "total_routed": len(self.routing_log),
            "layer1_coverage": (
                self.layer1_stats["hits"] / max(self.layer1_stats["total"], 1)
            ),
            "by_tool": {}
        }
        
        for tool, stats in self.accuracy_by_tool.items():
            if stats["total"] > 0:
                accuracy = stats["correct"] / stats["total"]
                metrics["by_tool"][tool] = {
                    "accuracy": accuracy,
                    "total": stats["total"]
                }
        
        return metrics


# =============================================================================
# PYDANTIC MODELS (Keep existing interface)
# =============================================================================

class OrchestratorRequest(BaseModel):
    query: str
    conversation_history: List[Dict[str, str]] = []


class OrchestratorResponse(BaseModel):
    answer: str
    tools_used: List[str]
    confidence: float
    sources: List[Dict[str, Any]]


# =============================================================================
# MAIN ORCHESTRATOR CLASS (BACKWARD COMPATIBLE)
# =============================================================================

class Orchestrator:
    """
    Enterprise-grade 4-layer hybrid orchestrator.
    
    BACKWARD COMPATIBLE with existing code that imports from 
    services.orchestrator import Orchestrator
    
    Architecture:
    - Layer 1: Fast rules (< 1ms) - obvious queries
    - Layer 2: LLM descriptions (100-200ms) - ambiguous queries
    - Layer 3: Feedback tracking - continuous improvement
    - Layer 4: Fallback - default to GenericRagTool
    
    This is a DIRECT REPLACEMENT for the previous Orchestrator.
    All existing code will work without changes.
    """
    
    def __init__(self, rag_pipeline: Optional[RAGPipeline] = None) -> None:
        """
        Initialize orchestrator with optional RAGPipeline for Layer 2.
        
        Args:
            rag_pipeline: RAGPipeline instance with LLM client for Layer 2 routing
                         If not provided, uses only Layer 1 + Layer 4 (no Layer 2)
        
        KEY CHANGE: Accepts rag_pipeline instead of llm_client
        This enables Layer 2 LLM routing when RAGPipeline is passed
        """
        self.rag_pipeline = rag_pipeline
        self.planner_pipeline = RAGPipeline() if not rag_pipeline else rag_pipeline
        self.feedback = RoutingFeedback()
        
        if self.rag_pipeline:
            logger.info("[Orchestrator] Initialized (4-layer hybrid routing WITH Layer 2 LLM ENABLED)")
        else:
            logger.info("[Orchestrator] Initialized (4-layer hybrid routing - Layer 2 disabled)")
    
    def handle_query(self, request: OrchestratorRequest) -> OrchestratorResponse:
        """
        Main orchestrator entry point.
        
        Routes query through 4-layer system and executes appropriate tool.
        
        Args:
            request: OrchestratorRequest with query and conversation history
            
        Returns:
            OrchestratorResponse with answer, tools used, and confidence
        """
        
        # =====================================================================
        # Routing Decision (Layers 1-4)
        # =====================================================================
        
        tool_name, confidence, routing_source = self._route_query(request.query)
        
        logger.info(
            f"[Orchestrator] Routed to {tool_name} "
            f"(confidence={confidence:.2f}, source={routing_source})"
        )
        
        # =====================================================================
        # Tool Execution
        # =====================================================================
        
        tool_results: List[ToolResult] = []
        tools_used: List[str] = []
        
        if tool_name in TOOL_MAP:
            try:
                # Get parameters from LLM planning
                tool_plan = self._plan_tools(request.query)
                params = (
                    tool_plan.get("tools", [{}])[0].get("parameters", {})
                    if tool_plan.get("tools") else {}
                )
                
                # Inside handle_query(), in the tool execution section
                # REPLACE the TranscriptTool parameter extraction with:

                if tool_name == "TranscriptTool":
                    # Extract student name using multiple patterns + case-insensitive
                    student_name = None
                    
                    # Pattern 1: "courses/grades/gpa for/of/is [Name]" (BEST - specific)
                    match = re.search(
                        r'(?:courses|grades?|gpa|transcript).*?(?:for|of|is)\s+([A-Z][a-z]{3,}(?:\s+[A-Z][a-z]+)*)\b',
                        request.query
                    )
                    if match:
                        student_name = match.group(1).strip()
                        logger.info(f"[Orchestrator] Pattern 1 matched: '{student_name}'")
                    
                    # Pattern 2: "of/for [Full Name]"
                    if not student_name:
                        match = re.search(
                            r'(?:of|for)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})',
                            request.query
                        )
                        if match:
                            student_name = match.group(1).strip()
                            logger.info(f"[Orchestrator] Pattern 2 matched: '{student_name}'")
                    
                    # Pattern 3: "which/student [Name] has/is/was/enrolled"
                    if not student_name:
                        match = re.search(
                            r'(?:which|student)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\s+(?:has|is|was|enrolled)',
                            request.query
                        )
                        if match:
                            student_name = match.group(1).strip()
                            logger.info(f"[Orchestrator] Pattern 3 matched: '{student_name}'")
                    
                    if student_name:
                        params["student_name"] = student_name
                        logger.info(f"[Orchestrator] Final extracted student_name: '{student_name}'")

                if tool_name == "PayrollTool":
                    q_lower = request.query.lower()
                    m_year = re.search(r"\b(20[0-9]{2})\b", q_lower)
                    if m_year:
                        params["year"] = int(m_year.group(1))
                    m_date = re.search(r"(\d{1,2}/\d{1,2}/\d{2,4})", q_lower)
                    if m_date:
                        params["check_date"] = m_date.group(1)
                    m_payroll = re.search(
                        r"(pay\s*period|payroll|p)\s*(\d+)",
                        q_lower
                    )
                    if m_payroll:
                        params["payroll_no"] = int(m_payroll.group(2))
                
                # Execute tool
                result = TOOL_MAP[tool_name](request.query, params)
                tool_results.append(result)
                tools_used.append(tool_name)
                
                # Log for feedback
                self.feedback.log_decision(
                    query=request.query,
                    routed_tool=tool_name,
                    confidence=confidence,
                    layer=1 if routing_source == "layer1" else 2
                )
                
            except Exception as e:
                logger.error(f"Tool execution failed: {e}", exc_info=True)
        
        # =====================================================================
        # Synthesize Response
        # =====================================================================
        
        # After tool_results and tools_used are populated
        if not tool_results:
            return OrchestratorResponse(
                answer="Unable to process query.",
                tools_used=[],
                confidence=0.0,
                sources=[],
            )

        final_answer = final_answer = "\n\n".join([tr.explanation if hasattr(tr, 'explanation') else tr.get('explanation', '') for tr in tool_results])
        avg_conf = sum(t.confidence for t in tool_results) / len(tool_results)

        # NEW: merge sources from all tools
        merged_sources: List[Dict[str, Any]] = []
        for tr in tool_results:
            if getattr(tr, "sources", None):
                merged_sources.extend(tr.sources)

        return OrchestratorResponse(
            answer=final_answer,
            tools_used=tools_used,
            confidence=avg_conf,
            sources=merged_sources,  # ← no longer []
        )

    
    def _route_query(self, query: str) -> Tuple[str, float, str]:
        """
        Hybrid routing through all layers.
        
        Layer 1: Fast rules (< 1ms)
        Layer 2: LLM with descriptions (100-200ms) - NOW ENABLED if rag_pipeline passed
        Layer 4: Safe fallback
        
        Returns: (tool_name, confidence, routing_source)
        """
        
        # Layer 1: Fast rules
        tool, confidence = layer1_route(query)
        if tool is not None:
            return (tool, confidence, "layer1_rule")
        
        # Layer 2: LLM with descriptions (ENABLED when rag_pipeline is available)
        if self.rag_pipeline:
            tool, confidence = layer2_route(query, self.rag_pipeline)
            if tool is not None:
                return (tool, confidence, "layer2_llm")
        else:
            logger.debug("[Routing] No RAGPipeline. Skipping Layer 2 (LLM disabled)")
        
        # Layer 4: Default fallback (safe, always works)
        logger.info("[Routing] Defaulting to GenericRagTool (safe fallback)")
        return ("GenericRagTool", 0.3, "layer4_default")
    
    def _plan_tools(self, query: str) -> Dict[str, Any]:
        """
        Get parameters for tool execution using LLM planning.
        """
        
        tools_desc = get_tool_descriptions()
        prompt = f"""
You are a tool parameter extractor.

Tools:
{tools_desc}

Query: "{query}"

Extract any relevant parameters as JSON:
{{
  "tools": [
    {{"name": "ToolName", "parameters": {{}}}}
  ]
}}
"""
        
        try:
            plan_str = self.planner_pipeline._generate_answer(
                context="",
                question=prompt
            )
            return json.loads(plan_str)
        except Exception as e:
            logger.warning(f"Parameter extraction failed: {e}")
            return {"tools": [{"name": "GenericRagTool", "parameters": {}}]}
    
    def get_routing_metrics(self) -> Dict[str, Any]:
        """
        Get routing performance metrics from Layer 3 feedback tracking.
        
        Returns:
            Dictionary with accuracy, coverage, and per-tool metrics
        """
        return self.feedback.get_metrics()
    
    def log_feedback(
        self,
        query: str,
        routed_tool: str,
        actual_tool: str,
        user_satisfied: bool
    ):
        """
        Log user feedback to improve routing over time.
        
        Args:
            query: Original user query
            routed_tool: Tool we routed to
            actual_tool: Correct tool (from user/system feedback)
            user_satisfied: Did user get correct answer?
        """
        self.feedback.log_decision(
            query=query,
            routed_tool=routed_tool,
            confidence=0.5,  # Feedback logged separately
            layer=0,  # Special marker for feedback
            actual_tool=actual_tool,
            user_satisfied=user_satisfied
        )
        logger.info(
            f"[Feedback] Logged: {routed_tool} vs {actual_tool} "
            f"(satisfied={user_satisfied})"
        )
