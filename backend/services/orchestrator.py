# services/orchestrator.py
# ENTERPRISE-GRADE SEMANTIC ROUTING ORCHESTRATOR (3-LAYER CONSERVATIVE)
# Semantic Router + Parallel Execution for Bias Protection
# BACKWARD COMPATIBLE with existing Orchestrator interface

from typing import Dict, Any, List, Tuple, Optional
import json
import re
import logging
import numpy as np
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

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
# SEMANTIC ROUTER - Embedded in Orchestrator
# =============================================================================

class SemanticRouter:
    """
    Semantic routing using embeddings for bias-free intent classification.
    Replaces brittle keyword matching with semantic similarity.
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize with tool descriptions and examples."""
        
        # Import here to avoid startup dependency if not used
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(embedding_model)
            logger.info(f"[SemanticRouter] Loaded model: {embedding_model}")
        except ImportError:
            logger.error(
                "[SemanticRouter] sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            self.model = None
            return
        
        # Tool descriptions with balanced examples
        self.tool_contexts = {
            "TranscriptTool": {
                "description": """
                Handles individual student academic records and performance data.
                Queries about specific student grades, courses taken by students,
                GPA calculations, academic standing, student enrollment in courses,
                credit hours earned, student performance, transcript data.
                """,
                "examples": [
                    "What is Trista Barrett's GPA?",
                    "Show me courses taken by Leslie Bright",
                    "Which students are enrolled in Business Math?",
                    "List top 5 students by GPA",
                    "How many credit hours did John complete?",
                    "Display students in tabular format sorted by GPA",
                    "What courses is Leslie enrolled in?",
                    "Show me all students with GPA above 3.5",
                    "What courses is Leslie enrolled in?",  # ← Already there
                    "What is Trista's GPA?",  # First name only
                    "Show courses for John",  # Short form
                    "Trista Barrett GPA",  # No question words
                    "Leslie courses enrolled",  # Minimal phrasing
                    "GPA for Arnoldo",  # Reverse pattern
                    "Courses taken by Blen",  # Name variations
                ]
            },
            
            "PayrollTool": {
                "description": """
                Handles payroll calendar, pay periods, check dates, payment schedules.
                Queries about when payments are made, specific pay period information,
                payroll deadlines, payment processing dates, salary schedules.
                """,
                "examples": [
                    "When is the check date for pay period 5?",
                    "What is the payroll schedule for 2024?",
                    "When do I get paid for p03?",
                    "Show me all pay periods in 2025",
                    "What date does payroll period 12 end?",
                    "When is the next payroll date?",
                ]
            },
            
            "BorPlannerTool": {
                "description": """
                Handles Board of Regents meetings, committee schedules, governance events.
                Queries about meeting dates, committee meetings, board calendars,
                governance event schedules, board agendas.
                """,
                "examples": [
                    "When is the next Board of Regents meeting?",
                    "Finance committee meeting schedule",
                    "Show me all BOR meetings for 2024",
                    "When does the audit committee meet?",
                    "What's on the agenda for the board meeting?",
                    "When is the Finance/Audit/Investment Committee meeting?",
                ]
            },
            
            "GenericRagTool": {
                "description": """
                Handles institutional policies, handbooks, procedures, general information.
                Academic calendar, housing rules, enrollment procedures (NOT student enrollment data),
                benefits information, health services, conduct policies, institutional calendars,
                general questions about policies and procedures.
                """,
                "examples": [
                    "What are the housing rules in the residence handbook?",
                    "What is the enrollment policy?",
                    "Show me the academic calendar for Fall 2024",
                    "What are the student conduct policies?",
                    "How do I register for classes?",
                    "What health insurance options are available?",
                    "What is the refund policy?",
                    "List the 2024-2025 academic calendar",
                ]
            }
        }
        
        # Pre-compute embeddings
        self.tool_embeddings = self._precompute_tool_embeddings()
        logger.info(f"[SemanticRouter] Ready with {len(self.tool_embeddings)} tools")
    
    def _precompute_tool_embeddings(self) -> Dict[str, np.ndarray]:
        """Pre-compute embeddings for all tool contexts."""
        if not self.model:
            return {}
        
        tool_embeddings = {}
        
        for tool_name, context in self.tool_contexts.items():
            # Combine description + examples
            full_context = (
                context["description"] + "\n\nExamples:\n" + 
                "\n".join(context["examples"])
            )
            
            # Embed
            embedding = self.model.encode(full_context, convert_to_tensor=False)
            tool_embeddings[tool_name] = embedding
            
        return tool_embeddings
    
    def get_top_k_candidates(
        self, 
        query: str, 
        k: int = 2
    ) -> List[Tuple[str, float]]:
        """
        Get top K candidate tools for a query.
        
        Args:
            query: User query
            k: Number of top candidates to return
        
        Returns:
            List of (tool_name, similarity_score) tuples, sorted descending
        """
        if not self.model:
            return []
        
        # Embed query
        query_embedding = self.model.encode(query, convert_to_tensor=False)
        
        # Compute similarities
        similarities = {}
        for tool_name, tool_embedding in self.tool_embeddings.items():
            similarity = self._cosine_similarity(query_embedding, tool_embedding)
            similarities[tool_name] = similarity
        
        # Sort and return top k
        sorted_tools = sorted(
            similarities.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_tools[:k]
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# =============================================================================
# FEEDBACK TRACKING
# =============================================================================

class RoutingFeedback:
    """Track routing decisions for monitoring."""
    
    def __init__(self):
        self.routing_log = []
        self.parallel_count = 0
        self.total_count = 0
    
    def log_decision(
        self,
        query: str,
        routed_tool: str,
        confidence: float,
        routing_source: str
    ):
        """Log routing decision."""
        entry = {
            "timestamp": datetime.now(),
            "query": query,
            "routed_tool": routed_tool,
            "confidence": confidence,
            "routing_source": routing_source,
        }
        self.routing_log.append(entry)
        self.total_count += 1
        
        if routing_source == "parallel_judge":
            self.parallel_count += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get routing performance metrics."""
        return {
            "total_routed": self.total_count,
            "parallel_executions": self.parallel_count,
            "parallel_rate": (
                self.parallel_count / max(self.total_count, 1)
            ),
        }


# =============================================================================
# PYDANTIC MODELS
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
# MAIN ORCHESTRATOR CLASS
# =============================================================================

class Orchestrator:
    """
    3-Layer Conservative Semantic Routing Orchestrator.
    
    Architecture:
    - Layer 0: Explicit format markers (optional, < 1ms)
    - Layer 1: Semantic router (candidate generator, 50-100ms)
    - Layer 2: Conservative decision + parallel execution (200-600ms when needed)
    
    Bias Protection: Parallel execution for close contests (margin < 0.20)
    """
    
    def __init__(self, rag_pipeline: Optional[RAGPipeline] = None) -> None:
        """Initialize orchestrator with semantic router."""
        self.rag_pipeline = rag_pipeline or RAGPipeline()
        self.planner_pipeline = self.rag_pipeline
        self.feedback = RoutingFeedback()
        self._cached_parallel_result = None
        
        # Initialize semantic router
        try:
            self.semantic_router = SemanticRouter()
            logger.info(
                "[Orchestrator] Initialized with Semantic Router "
                "+ Parallel Execution (3-layer conservative)"
            )
        except Exception as e:
            logger.error(f"[Orchestrator] Semantic router failed to init: {e}")
            self.semantic_router = None
    
    def handle_query(self, request: OrchestratorRequest) -> OrchestratorResponse:
        """Main orchestrator entry point."""
        
        # Route query
        tool_name, confidence, routing_source = self._route_query(request.query)
        
        logger.info(
            f"[Orchestrator] Routed to {tool_name} "
            f"(confidence={confidence:.2f}, source={routing_source})"
        )
        
        # Check if we have cached parallel result
        if self._cached_parallel_result:
            result = self._cached_parallel_result
            self._cached_parallel_result = None
            
            return OrchestratorResponse(
                answer=result.explanation,
                tools_used=[tool_name],
                confidence=result.confidence,
                sources=result.sources if hasattr(result, 'sources') else [],
            )
        
        # Execute tool
        tool_results: List[ToolResult] = []
        tools_used: List[str] = []
        
        if tool_name in TOOL_MAP:
            try:
                # Extract parameters
                params = {}
                
                if tool_name == "TranscriptTool":
                    # Extract student name
                    student_name = None
                    match = re.search(
                        r'(?:courses|grades?|gpa|transcript).*?(?:for|of|is)\s+([A-Z][a-z]{3,}(?:\s+[A-Z][a-z]+)*)\b',
                        request.query
                    )
                    if match:
                        student_name = match.group(1).strip()
                    
                    if not student_name:
                        match = re.search(
                            r'(?:of|for)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})',
                            request.query
                        )
                        if match:
                            student_name = match.group(1).strip()
                    
                    if student_name:
                        params["student_name"] = student_name
                
                elif tool_name == "PayrollTool":
                    q_lower = request.query.lower()
                    m_year = re.search(r"\b(20[0-9]{2})\b", q_lower)
                    if m_year:
                        params["year"] = int(m_year.group(1))
                    m_payroll = re.search(r"(pay\s*period|payroll|p)\s*(\d+)", q_lower)
                    if m_payroll:
                        params["payroll_no"] = int(m_payroll.group(2))
                
                # Execute
                result = TOOL_MAP[tool_name](request.query, params)
                tool_results.append(result)
                tools_used.append(tool_name)
                
                # Log feedback
                self.feedback.log_decision(
                    query=request.query,
                    routed_tool=tool_name,
                    confidence=confidence,
                    routing_source=routing_source
                )
                
            except Exception as e:
                logger.error(f"Tool execution failed: {e}", exc_info=True)
        
        # Synthesize response
        if not tool_results:
            return OrchestratorResponse(
                answer="Unable to process query.",
                tools_used=[],
                confidence=0.0,
                sources=[],
            )
        
        final_answer = "\n\n".join([
            tr.explanation if hasattr(tr, 'explanation') 
            else tr.get('explanation', '') 
            for tr in tool_results
        ])
        avg_conf = sum(t.confidence for t in tool_results) / len(tool_results)
        
        merged_sources: List[Dict[str, Any]] = []
        for tr in tool_results:
            if getattr(tr, "sources", None):
                merged_sources.extend(tr.sources)
        
        return OrchestratorResponse(
            answer=final_answer,
            tools_used=tools_used,
            confidence=avg_conf,
            sources=merged_sources,
        )
    
    def _route_query(self, query: str) -> Tuple[str, float, str]:
        """
        3-Layer Conservative Routing with bias protection.
        
        Returns: (tool_name, confidence, routing_source)
        """
        q_norm = query.lower()
        
        # =================================================================
        # LAYER 0: Explicit Format Markers (Optional, < 1ms)
        # =================================================================
        
        # Payroll code format (p01, p05, etc.)
        if re.search(r'\bp\d{2}\b', q_norm):
            logger.info("[Layer 0] Explicit payroll code --> PayrollTool")
            return ("PayrollTool", 1.0, "explicit_format")
        
        # Exact BOR phrase
        if 'board of regents' in q_norm:
            logger.info("[Layer 0] Explicit BOR phrase --> BorPlannerTool")
            return ("BorPlannerTool", 1.0, "explicit_phrase")
        
        # =================================================================
        # LAYER 1: Semantic Router (Candidate Generator, 50-100ms)
        # =================================================================
        
        if not self.semantic_router or not self.semantic_router.model:
            # Fallback if semantic router unavailable
            logger.warning("[Routing] Semantic router unavailable --> GenericRagTool")
            return ("GenericRagTool", 0.3, "no_semantic_router")
        
        candidates = self.semantic_router.get_top_k_candidates(query, k=2)
        
        if not candidates or candidates[0][1] < 0.50:
            logger.info("[Routing] All scores < 0.50 --> GenericRagTool fallback")
            return ("GenericRagTool", 0.3, "low_confidence_fallback")
        
        tool1, score1 = candidates[0]
        tool2, score2 = candidates[1] if len(candidates) > 1 else (None, 0.0)
        margin = score1 - score2
        
        logger.info(
            f"[Routing] Candidates: {tool1}={score1:.3f}, "
            f"{tool2}={score2:.3f}, margin={margin:.3f}"
        )
        
        # =================================================================
        # LAYER 2: Conservative Decision with Parallel Protection
        # =================================================================
        
        # High confidence + clear gap --> Safe to route
        if score1 >= 0.85 and margin >= 0.20:
            logger.info(
                f"[Routing] High confidence + clear gap --> {tool1}"
            )
            return (tool1, score1, "direct_high_confidence")
        
        # Close contest OR uncertain --> PARALLEL (bias protection)
        if tool2 and margin < 0.20:
            logger.info(
                f"[Routing] Close contest (margin={margin:.3f}) --> "
                f"PARALLEL execution to avoid bias"
            )
            
            try:
                selected_tool, result = self._execute_parallel_with_judge(
                    query, 
                    candidates[:2]
                )
                self._cached_parallel_result = result
                return (selected_tool, 0.85, "parallel_judge")
                
            except Exception as e:
                logger.error(f"[Routing] Parallel execution failed: {e}")
                return (tool1, score1, "parallel_failed_fallback")
        
        # Medium confidence with reasonable gap
        logger.info(f"[Routing] Medium confidence --> {tool1}")
        return (tool1, max(0.70, score1), "direct_medium_confidence")
    
    def _execute_parallel_with_judge(
        self, 
        query: str, 
        candidate_tools: List[Tuple[str, float]]
    ) -> Tuple[str, ToolResult]:
        """
        Execute top 2 candidate tools in parallel with LLM judge.
        ONLY called when ambiguous (margin < 0.20).
        """
        logger.info(f"[Parallel] Executing {len(candidate_tools)} tools")
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}
            
            for tool_name, semantic_conf in candidate_tools:
                if tool_name in TOOL_MAP:
                    future = executor.submit(TOOL_MAP[tool_name], query, {})
                    futures[future] = (tool_name, semantic_conf)
            
            # Collect results with timeout
            for future in as_completed(futures, timeout=30):
                tool_name, semantic_conf = futures[future]
                try:
                    result = future.result()
                    results[tool_name] = {
                        "result": result,
                        "semantic_confidence": semantic_conf
                    }
                    logger.debug(f"[Parallel] {tool_name} completed")
                except Exception as e:
                    logger.error(f"[Parallel] {tool_name} failed: {e}")
        
        if not results:
            # Both failed
            return ("GenericRagTool", ToolResult(
                data={},
                explanation="Tools failed to execute.",
                confidence=0.0,
                format_hint="text",
                citations=[]
            ))
        
        if len(results) == 1:
            # Only one succeeded
            tool_name = list(results.keys())[0]
            return (tool_name, results[tool_name]["result"])
        
        # LLM Judge - pick best answer
        tool_names = list(results.keys())
        judge_prompt = f"""You are judging which answer is better for this query.

Query: "{query}"

Answer A ({tool_names[0]}):
{results[tool_names[0]]["result"].explanation[:300]}

Answer B ({tool_names[1]}):
{results[tool_names[1]]["result"].explanation[:300]}

Which answer is more relevant and accurate? Respond with ONLY: A or B
"""
        
        try:
            decision = self.rag_pipeline._generate_answer("", judge_prompt).strip().upper()
            
            if decision == "A":
                selected = tool_names[0]
            elif decision == "B":
                selected = tool_names[1]
            else:
                # Fallback to highest confidence
                selected = max(
                    results.items(), 
                    key=lambda x: x[1]["result"].confidence
                )[0]
            
            logger.info(f"[Judge] Selected {selected}")
            return (selected, results[selected]["result"])
            
        except Exception as e:
            logger.error(f"[Judge] Failed: {e}")
            # Fallback to highest confidence
            selected = max(
                results.items(), 
                key=lambda x: x[1]["result"].confidence
            )[0]
            return (selected, results[selected]["result"])
    
    def get_routing_metrics(self) -> Dict[str, Any]:
        """Get routing performance metrics."""
        return self.feedback.get_metrics()
