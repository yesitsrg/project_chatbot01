#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core import init_logger
init_logger()

from services.tools.transcript_tool import answer

test_queries = [
    # "What is the GPA of Trista Barrett?",
    # "What is the GPA of Leslie Nichole Bright?",
    # "How many students are in this data?",
    # "Show me the top 5 students by GPA",
    "What courses is Leslie enrolled in?",
]

print("="*80)
print("TESTING CSV AGENT APPROACH")
print("="*80)

for query in test_queries:
    print(f"\n{'='*80}")
    print(f"QUERY: {query}")
    print(f"{'='*80}")
    
    result = answer(query)
    
    print(f"✅ CONFIDENCE: {result.confidence:.2f}")
    print(f"📄 ANSWER:\n{result.explanation}\n")
