"""Enterprise RAG Pipeline v2.0 - Multi-Modal Generic."""

from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
import os
import logging
import re
import pandas as pd
from io import StringIO

from core import get_logger
from db import ChromaDBManager
from services import DocumentParser, TextProcessor, AutoTagger
from config import get_settings

logger = get_logger(__name__)

class QueryRouter:
    """Generic query classification for all document types."""
    
    def __init__(self):
        self.rules = {
            'TABLE_QUERY': [
                r'\b(?:how many|count|total|number of|list all|period \d+|row \d+|check date)\b',
                r'\b(?:csv|table|spreadsheet|data)\b.*\b(?:period|row|date|count)\b'
            ],
            'STRUCTURED_QUERY': [
                r'\b(?:period \d+|pay period|check date|employee id|policy \w+|section \w+)\b',
                r'(?:benefits?|salary|payroll|retirement).*?(?:plan|details?|policy)'
            ],
            'TEXT_QUERY': [
                r'\b(?:what is|explain|describe|policy|procedure|benefits?|rules?|requirements?)\b'
            ]
        }
    
    def classify(self, question: str) -> str:
        """Classify: 'TABLE', 'STRUCTURED', 'TEXT'."""
        question_lower = question.lower()
        scores = {}
        
        for qtype, patterns in self.rules.items():
            score = sum(1 for pattern in patterns if re.search(pattern, question_lower))
            scores[qtype] = score
        
        return max(scores, key=scores.get)

class TableAgent:
    """Generic table/CSV/JSON agent - ALL formats."""
    
    def __init__(self, db: ChromaDBManager):
        self.db = db
    
    def extract_tables(self, results: List[Dict]) -> List[pd.DataFrame]:
        """Generic table extraction from markdown + JSON (robust, enterprise-grade)."""
        tables = []

        for result in results:
            content = result.get('content', '')

            # ─────────────────────────────────────────
            # 1) ROBUST MARKDOWN TABLE PARSING
            # ─────────────────────────────────────────
            if '|' in content and content.count('|') >= 10:
                lines = [line.strip() for line in content.split('\n') if '|' in line][:25]
                if len(lines) >= 3:
                    try:
                        clean_lines = []
                        for line in lines:
                            # Remove outer pipes, split on inner pipes with flexible spacing
                            parts = re.split(r'\s*\|\s*', line.strip('|'))
                            parts = [p.strip() for p in parts if p.strip()]
                            if len(parts) >= 2:
                                clean_lines.append('|'.join(parts))

                        if len(clean_lines) >= 3:
                            df = pd.read_csv(StringIO('\n'.join(clean_lines)), sep='|')
                            if len(df.columns) >= 2 and len(df) >= 1:
                                df.columns = df.columns.str.strip()
                                tables.append(df)
                                logger.info(f"Parsed markdown table: {len(df)}x{len(df.columns)}")
                    except Exception as e:
                        logger.debug(f"Markdown table parse failed: {e}")

            # ─────────────────────────────────────────
            # 2) JSON → TABLE (KEEP THIS LOGIC)
            # ─────────────────────────────────────────
            text = content.strip()
            if text.startswith('[') or text.startswith('{'):
                try:
                    import json
                    data = json.loads(text)
                    if isinstance(data, list) and len(data) > 0:
                        df = pd.json_normalize(data)
                        if len(df) > 0:
                            tables.append(df)
                            logger.info(f"Parsed JSON table: {len(df)}x{len(df.columns)}")
                except Exception as e:
                    logger.debug(f"JSON table parse failed: {e}")

        return tables


class RAGPipeline:
    """Enterprise RAG v2.0 - Generic Multi-Modal."""
    
    def __init__(self):
        self.db = ChromaDBManager()
        self.parser = DocumentParser()
        self.processor = TextProcessor()
        self.tagger = AutoTagger()
        self.router = QueryRouter()
        self.table_agent = TableAgent(self.db)
        self.settings = get_settings()
        logger.info("Enterprise RAG v2.0 ready - Multi-Modal")

    def _call_groq(self, prompt: str) -> str:
        """Groq API client."""
        groq_key = self.settings.groq_api_key or os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("No Groq API key")
            
        url = "https://api.groq.com/openai/v1/chat/completions"
        payload = {
            "model": self.settings.groq_model,
            "messages": [
                {"role": "system", "content": "Answer ONLY using provided context. Be precise with dates/numbers. If insufficient info, say so."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": self.settings.llm_max_tokens,
            "temperature": 0.1  # Lower for factual accuracy
        }
        headers = {"Authorization": f"Bearer {groq_key}"}
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    # Keep your existing _call_ollama unchanged
    def _call_ollama(self, prompt: str) -> str:
        base_url = self.settings.ollama_base_url.rstrip("/")
        model = self.settings.ollama_model
        
        for endpoint, payload in [
            (f"{base_url}/api/generate", {"model": model, "prompt": prompt, "stream": False}),
            (f"{base_url}/api/chat", {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            })
        ]:
            try:
                resp = requests.post(endpoint, json=payload, timeout=self.settings.ollama_timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("response") or data.get("message", {}).get("content", "")
            except:
                continue
        raise Exception("Both Ollama endpoints failed")

    def _generate_answer(self, context: str, question: str) -> str:
        """Enhanced prompting for accuracy."""
        prompt = f"""You are an assistant answering questions strictly based on the provided context.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    INSTRUCTIONS:
    - Answer using ONLY the context above.
    - For dates, numbers, and tables: quote EXACT values from the context.
    - If the context truly does not contain enough information to answer, respond with:
    "The documents do not contain information about this question."
    - Do NOT invent facts or guess beyond the context.
    """

        # Multi-provider with temp=0.1 for facts
        providers = []
        if self.settings.llm_provider == "groq" and self.settings.groq_api_key:
            providers.append(self._call_groq)
        elif self.settings.llm_provider == "ollama":
            providers.append(self._call_ollama)

        if os.getenv("GROQ_API_KEY"):
            providers.append(self._call_groq)
        providers.append(self._call_ollama)

        for provider in providers:
            try:
                return provider(prompt).strip()
            except Exception as e:
                logger.warning(f"{provider.__name__} failed: {e}")
        return "LLM unavailable"

    def ingest_documents(self, doc_paths: List[str]) -> Dict[str, int]:
        """Enhanced ingest with full metadata."""
        stats = {"processed": 0, "failed": 0}
        for path in doc_paths:
            try:
                file_path = Path(path)
                parsed = self.parser.parse_file(file_path)
                if not parsed or not parsed.content:
                    stats["failed"] += 1
                    continue
                
                chunks = self.processor.process_document(parsed)
                if not chunks:
                    stats["failed"] += 1
                    continue
                
                # CRITICAL: Full metadata preservation
                for chunk in chunks:
                    chunk.metadata.update({
                        'filename': parsed.filename,
                        'document_type': parsed.document_type.value,
                        'extraction_method': parsed.extraction_method,
                        'confidence': parsed.extraction_confidence,
                        'file_size_mb': parsed.metadata.get('file_size_mb', 0)
                    })
                
                result = self.db.add_chunks(chunks)
                logger.info(f"Ingested {path}: {result}")
                stats["processed"] += 1
            except Exception as e:
                logger.error(f"Failed {path}: {e}")
                stats["failed"] += 1
        return stats

    def query(self, question: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generic Multi-Modal RAG."""
        
        # 1. Smart routing
        query_type = self.router.classify(question)
        logger.info(f"Query type: {query_type}")
        
        # 2. Retrieve MORE context (top_k=10)
        results = self.db.query(question, top_k=max(top_k * 2, 10), filters=filters)
        
        # 3. Fix metadata for ALL sources
        for r in results:
            if 'filename' not in r:
                r['filename'] = r.get('metadata', {}).get('filename', 'Unknown document')
        
        # 4. Route to specialist agents
        # ENTERPRISE TABLE AGENT - Generic Multi-Format
        table_detected = (
            query_type == 'TABLE_QUERY' or
            any(kw in question.lower() for kw in [
                'period', 'check date', 'how many', 'count', 'total', 'number of', 
                'payroll_no', 'row', 'table', 'csv', 'spreadsheet'
            ]) or
            any(kw in str(results[0].get('content', '')).lower() for kw in [
                'payroll_no', '| payroll_no |', 'check_date', '| start_date |'
            ])
        )

        if table_detected:
            logger.info("Table Agent ACTIVATED")
            
            # Extract ALL tables from results
            tables = self.table_agent.extract_tables(results)
            
            if tables:
                logger.info(f"Table Agent: Found {len(tables)} tables")
                
                # Build rich table context
                table_context = []
                for i, df in enumerate(tables[:5]):  # Top 5 tables
                    table_info = f"📊 TABLE {i+1} ({len(df)} rows x {len(df.columns)} cols):"
                    table_info += f"\nColumns: {', '.join(df.columns.tolist())}"
                    table_info += f"\n{df.head(10).to_string(index=False)}"  # First 10 rows
                    table_context.append(table_info)
                
                table_context = "\n\n".join(table_context)
                answer = self._generate_answer(table_context, question)
                
                return {
                    "question": question,
                    "answer": answer,
                    "sources": results[:5],  # More sources for tables
                    "agent": "table_agent",
                    "tables_found": len(tables),
                    "table_preview": tables[0].head(3).to_dict() if tables else None
                }
            else:
                logger.info("Table Agent: No structured tables, falling back to text RAG")

        # 5. Generic text RAG (enhanced)
        context = "\n\n---\n\n".join([f"Source: {r['filename']}\n{r['content']}" for r in results[:5]])
        answer = self._generate_answer(context, question)
        
        return {
            "question": question,
            "answer": answer,
            "sources": results[:3],
            "agent": "text_rag",
            "query_type": query_type
        }
