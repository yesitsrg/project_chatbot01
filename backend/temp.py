#!/usr/bin/env python3
"""Test the enhanced 3-tier transcript tool"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from core import init_logger
init_logger()

from services.orchestrator import Orchestrator, OrchestratorRequest

# Create orchestrator
orchestrator = Orchestrator()

# Test queries
test_cases = [
    # Tier 1 tests
    ("Give me the GPA details of Trista Barrett", "Tier 1 - Single student"),
    ("What is the GPA of Leslie Nichole Bright?", "Tier 1 - Single student with middle name"),
    ("How many students are in the transcript data?", "Tier 1 - Count"),
    ("Show me the top 5 students by GPA", "Tier 1 - Top N"),
    
    # Tier 2 tests
    ("How many students have A grade in Fall 2024?", "Tier 2 - Grade distribution"),
    
    # Routing tests (enrollment ambiguity)
    ("What courses is Leslie enrolled in?", "Should route to TranscriptTool (entity detected)"),
    ("What is the enrollment policy?", "Should route to GenericRagTool (policy intent)"),
]

print("="*80)
print("TESTING ENHANCED 3-TIER TRANSCRIPT TOOL")
print("="*80)

for query, description in test_cases:
    print(f"\n{'='*80}")
    print(f"TEST: {description}")
    print(f"QUERY: {query}")
    print(f"{'='*80}")
    
    request = OrchestratorRequest(query=query)
    response = orchestrator.handle_query(request)
    
    print(f"\n✅ TOOLS USED: {response.tools_used}")
    print(f"✅ CONFIDENCE: {response.confidence:.2f}")
    print(f"\n📄 ANSWER:\n{response.answer[:500]}")
    print(f"\n{'='*80}")

print("\n✅ ALL TESTS COMPLETED")
