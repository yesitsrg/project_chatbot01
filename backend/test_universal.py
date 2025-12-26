#!/usr/bin/env python3
"""
UNIVERSAL TESTING SCRIPT - Multi-Domain Question Answering
Tests questions across ALL domains: TRANSCRIPT, RAG, GENERAL, etc.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core import init_logger
init_logger()

import pandas as pd
import time
from datetime import datetime
from services.orchestrator import Orchestrator, OrchestratorRequest
from services.rag_pipeline import RAGPipeline


# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_CSV = r"D:\jericho\backend\Test-Question.csv"
OUTPUT_CSV = r"D:\jericho\backend\Test-Results-{timestamp}.csv"
SLEEP_BETWEEN_CALLS = 3  # seconds to wait between API calls


# ============================================================================
# BATCH TEST FUNCTION
# ============================================================================

def run_batch_test(input_path: str, output_path: str, sleep_seconds: int = 3):
    """
    Run batch testing on questions from CSV file across ALL domains.
    
    Args:
        input_path: Path to input CSV with 'domain' and 'question' columns
        output_path: Path to output CSV for results
        sleep_seconds: Seconds to sleep between API calls
    """
    
    print("="*80)
    print("UNIVERSAL MULTI-DOMAIN TESTING")
    print("="*80)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Rate limit: {sleep_seconds}s between calls")
    print("="*80)
    print()
    
    # Initialize orchestrator with RAG pipeline (enables all 4 layers)
    print("Initializing orchestrator...")
    rag_pipeline = RAGPipeline()
    orchestrator = Orchestrator(rag_pipeline=rag_pipeline)
    print("Orchestrator ready!\n")
    
    # Read input CSV
    try:
        df_input = pd.read_csv(input_path)
        print(f"Loaded {len(df_input)} questions from input CSV")
    except Exception as e:
        print(f"ERROR: Failed to read input CSV: {e}")
        return
    
    # Validate required columns
    if 'question' not in df_input.columns:
        print("ERROR: Input CSV must have 'question' column")
        return
    
    # Add domain column if missing
    if 'domain' not in df_input.columns:
        df_input['domain'] = 'UNKNOWN'
        print("Warning: No 'domain' column found, defaulting to UNKNOWN")
    
    # Show domain distribution
    print("\nDomain distribution:")
    domain_counts = df_input['domain'].value_counts()
    for domain, count in domain_counts.items():
        print(f"  - {domain}: {count} questions")
    print()
    
    # Initialize results storage
    results = []
    
    # Process each question
    for idx, row in df_input.iterrows():
        question = row['question']
        domain = row.get('domain', 'UNKNOWN')
        
        print(f"\n{'='*80}")
        print(f"[{idx+1}/{len(df_input)}] Testing Question")
        print(f"{'='*80}")
        print(f"Domain:   {domain}")
        print(f"Question: {question}")
        print("-"*80)
        
        try:
            # Create orchestrator request
            request = OrchestratorRequest(
                query=question,
                conversation_history=[]
            )
            
            # Call orchestrator
            response = orchestrator.handle_query(request)
            
            # Extract results
            generated_answer = response.answer
            confidence = response.confidence
            tools_used = ", ".join(response.tools_used) if response.tools_used else "None"
            
            # Extract sources
            sources = []
            if hasattr(response, 'sources') and response.sources:
                for src in response.sources:
                    if isinstance(src, dict):
                        sources.append(src.get('file', src.get('source', 'Unknown')))
                    else:
                        sources.append(str(src))
            sources_str = ", ".join(sources) if sources else "N/A"
            
            status = "SUCCESS"
            error_msg = ""
            
            print(f"Status:     SUCCESS")
            print(f"Tools Used: {tools_used}")
            print(f"Confidence: {confidence:.2f}")
            print(f"Sources:    {sources_str}")
            print(f"Answer:     {generated_answer[:200]}...")
            
        except Exception as e:
            # Handle errors gracefully
            generated_answer = ""
            confidence = 0.0
            tools_used = "ERROR"
            sources_str = ""
            status = "ERROR"
            error_msg = str(e)
            
            print(f"Status: ERROR")
            print(f"Error:  {error_msg}")
        
        # Store result
        results.append({
            'domain': domain,
            'question': question,
            'Generated Answer': generated_answer,
            'Sources': sources_str,
            'Confidence': confidence,
            'Tools Used': tools_used,
            'Status': status,
            'Error': error_msg,
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Sleep to respect rate limits (except for last iteration)
        if idx < len(df_input) - 1:
            print(f"\nSleeping {sleep_seconds}s before next call...")
            time.sleep(sleep_seconds)
    
    # Create results DataFrame
    df_results = pd.DataFrame(results)
    
    # Write to output CSV
    try:
        df_results.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print("\n" + "="*80)
        print("BATCH TEST COMPLETE")
        print("="*80)
        
        # Overall statistics
        success_df = df_results[df_results['Status'] == 'SUCCESS']
        print(f"\nOVERALL STATISTICS:")
        print(f"  Total questions:    {len(df_results)}")
        print(f"  Successful:         {len(success_df)}")
        print(f"  Errors:             {len(df_results[df_results['Status'] == 'ERROR'])}")
        if len(success_df) > 0:
            print(f"  Average confidence: {success_df['Confidence'].mean():.2f}")
        
        # Statistics by domain
        print(f"\nBY DOMAIN:")
        for domain in df_results['domain'].unique():
            domain_data = df_results[df_results['domain'] == domain]
            success_count = len(domain_data[domain_data['Status'] == 'SUCCESS'])
            domain_success = domain_data[domain_data['Status'] == 'SUCCESS']
            avg_conf = domain_success['Confidence'].mean() if len(domain_success) > 0 else 0
            print(f"  {domain}:")
            print(f"    - Success: {success_count}/{len(domain_data)}")
            if avg_conf > 0:
                print(f"    - Avg Confidence: {avg_conf:.2f}")
        
        # Tool usage
        print(f"\nTOOL USAGE:")
        tool_counts = success_df['Tools Used'].value_counts()
        for tool, count in tool_counts.items():
            print(f"  - {tool}: {count}")
        
        print(f"\nResults saved to: {output_path}")
        print("="*80)
        
    except Exception as e:
        print(f"\nERROR: Failed to write output CSV: {e}")
        print("Results DataFrame:")
        print(df_results)


# ============================================================================
# INTERACTIVE TEST (for quick testing across domains)
# ============================================================================

def run_interactive_test():
    """Run quick interactive test with predefined queries across domains"""
    
    test_queries = [
        # TRANSCRIPT domain
        ("TRANSCRIPT", "What is the GPA of Trista Barrett?"),
        ("TRANSCRIPT", "How many students are in this data?"),
        ("TRANSCRIPT", "What courses is Leslie enrolled in?"),
        
        # PAYROLL domain
        ("PAYROLL", "What is the check date for pay period 5?"),
        
        # BOR domain
        ("BOR", "When is the next Board of Regents meeting?"),
        
        # RAG/GENERAL domain
        ("RAG", "What is the enrollment policy?"),
    ]
    
    print("="*80)
    print("INTERACTIVE MULTI-DOMAIN TEST MODE")
    print("="*80)
    
    # Initialize orchestrator
    print("\nInitializing orchestrator...")
    rag_pipeline = RAGPipeline()
    orchestrator = Orchestrator(rag_pipeline=rag_pipeline)
    print("Orchestrator ready!\n")
    
    for idx, (domain, query) in enumerate(test_queries, 1):
        print(f"\n[{idx}/{len(test_queries)}] {'='*70}")
        print(f"DOMAIN: {domain}")
        print(f"QUERY:  {query}")
        print('='*80)
        
        try:
            request = OrchestratorRequest(
                query=query,
                conversation_history=[]
            )
            
            response = orchestrator.handle_query(request)
            
            print(f"TOOLS USED: {', '.join(response.tools_used)}")
            print(f"CONFIDENCE: {response.confidence:.2f}")
            print(f"\nANSWER:\n{response.answer}\n")
            
        except Exception as e:
            print(f"ERROR: {e}\n")
        
        # Sleep between calls
        if idx < len(test_queries):
            time.sleep(2)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Universal multi-domain question answering test')
    parser.add_argument('--mode', choices=['batch', 'interactive'], default='batch',
                        help='Test mode: batch (from CSV) or interactive (predefined queries)')
    parser.add_argument('--input', type=str, default=INPUT_CSV,
                        help='Input CSV file path (batch mode only)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file path (batch mode only)')
    parser.add_argument('--sleep', type=int, default=SLEEP_BETWEEN_CALLS,
                        help='Seconds to sleep between API calls')
    
    args = parser.parse_args()
    
    if args.mode == 'batch':
        # Generate output filename with timestamp
        if args.output is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = OUTPUT_CSV.format(timestamp=timestamp)
        else:
            output_path = args.output
        
        # Check if input file exists
        if not Path(args.input).exists():
            print(f"ERROR: Input file not found: {args.input}")
            sys.exit(1)
        
        # Run batch test
        run_batch_test(args.input, output_path, args.sleep)
        
    else:
        # Run interactive test
        run_interactive_test()
