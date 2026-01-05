# -*- coding: utf-8 -*-
"""
Script to extract success and failure cases from BIRD and Spider training datasets
for the RAG memory module.

Success cases: Correct SQL queries from training data
Failure cases: Synthetically generated common SQL errors with corrections

This approach avoids data leakage by only using training data.
"""

import os
import sys
import json
import random
import sqlite3
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SQLErrorGenerator:
    """Generate common SQL errors for failure cases"""
    
    # Common error patterns that can be applied to correct SQL
    ERROR_PATTERNS = [
        {
            "name": "missing_group_by",
            "description": "Missing GROUP BY clause when using aggregate functions",
            "apply": lambda self, sql, schema: self._remove_group_by(sql),
            "error_msg": "column must appear in the GROUP BY clause or be used in an aggregate function",
        },
        {
            "name": "wrong_table_name",
            "description": "Typo in table name",
            "apply": lambda self, sql, schema: self._typo_table_name(sql, schema),
            "error_msg": "no such table",
        },
        {
            "name": "wrong_column_name",
            "description": "Typo in column name",
            "apply": lambda self, sql, schema: self._typo_column_name(sql, schema),
            "error_msg": "no such column",
        },
        {
            "name": "missing_join_condition",
            "description": "Missing ON clause in JOIN",
            "apply": lambda self, sql, schema: self._remove_join_condition(sql),
            "error_msg": "ambiguous column name",
        },
        {
            "name": "wrong_string_quotes",
            "description": "Using double quotes instead of single quotes for strings",
            "apply": lambda self, sql, schema: self._wrong_quotes(sql),
            "error_msg": "no such column",
        },
        {
            "name": "missing_where_clause",
            "description": "Forgot WHERE keyword",
            "apply": lambda self, sql, schema: self._remove_where_keyword(sql),
            "error_msg": "syntax error",
        },
        {
            "name": "wrong_aggregate",
            "description": "Using wrong aggregate function",
            "apply": lambda self, sql, schema: self._wrong_aggregate(sql),
            "error_msg": "incorrect result due to wrong aggregate function",
        },
        {
            "name": "missing_alias",
            "description": "Missing table alias in multi-table query",
            "apply": lambda self, sql, schema: self._remove_alias(sql),
            "error_msg": "ambiguous column name",
        },
    ]
    
    def __init__(self):
        pass
    
    def _remove_group_by(self, sql: str) -> Optional[str]:
        """Remove GROUP BY clause"""
        if 'GROUP BY' not in sql.upper():
            return None
        # Remove GROUP BY and everything until ORDER BY, HAVING, LIMIT or end
        pattern = r'\s+GROUP\s+BY\s+[^;]*(ORDER|HAVING|LIMIT|;|$)'
        match = re.search(pattern, sql, re.IGNORECASE)
        if match:
            end_keyword = match.group(1) if match.group(1) else ''
            new_sql = re.sub(r'\s+GROUP\s+BY\s+[^;]*?(ORDER|HAVING|LIMIT|$)', f' {end_keyword}', sql, flags=re.IGNORECASE)
            return new_sql.strip()
        return None
    
    def _typo_table_name(self, sql: str, schema: dict) -> Optional[str]:
        """Introduce typo in table name"""
        tables = schema.get('table_names_original', [])
        if not tables:
            return None
        
        for table in tables:
            if table.lower() in sql.lower() and len(table) > 3:
                # Add typo - remove last character
                typo_table = table[:-1]
                new_sql = re.sub(rf'\b{table}\b', typo_table, sql, flags=re.IGNORECASE)
                if new_sql != sql:
                    return new_sql
        return None
    
    def _typo_column_name(self, sql: str, schema: dict) -> Optional[str]:
        """Introduce typo in column name"""
        columns = [col[1] for col in schema.get('column_names_original', []) if col[0] >= 0]
        if not columns:
            return None
        
        for col in columns:
            if col.lower() in sql.lower() and len(col) > 3:
                # Add typo - swap two characters
                typo_col = col[:-2] + col[-1] + col[-2] if len(col) > 2 else col + 's'
                new_sql = re.sub(rf'\b{col}\b', typo_col, sql, flags=re.IGNORECASE)
                if new_sql != sql:
                    return new_sql
        return None
    
    def _remove_join_condition(self, sql: str) -> Optional[str]:
        """Remove ON condition from JOIN"""
        if ' ON ' not in sql.upper():
            return None
        # Remove ON clause
        new_sql = re.sub(r'\s+ON\s+\S+\s*=\s*\S+', '', sql, count=1, flags=re.IGNORECASE)
        if new_sql != sql:
            return new_sql
        return None
    
    def _wrong_quotes(self, sql: str) -> Optional[str]:
        """Replace single quotes with double quotes for strings"""
        if "'" not in sql:
            return None
        # Replace single quotes with double quotes
        new_sql = sql.replace("'", '"')
        if new_sql != sql:
            return new_sql
        return None
    
    def _remove_where_keyword(self, sql: str) -> Optional[str]:
        """Remove WHERE keyword"""
        if ' WHERE ' not in sql.upper():
            return None
        new_sql = re.sub(r'\s+WHERE\s+', ' ', sql, count=1, flags=re.IGNORECASE)
        if new_sql != sql:
            return new_sql
        return None
    
    def _wrong_aggregate(self, sql: str) -> Optional[str]:
        """Replace aggregate function with wrong one"""
        replacements = [
            (r'\bSUM\s*\(', 'COUNT('),
            (r'\bCOUNT\s*\(', 'SUM('),
            (r'\bAVG\s*\(', 'MAX('),
            (r'\bMAX\s*\(', 'MIN('),
            (r'\bMIN\s*\(', 'MAX('),
        ]
        for pattern, replacement in replacements:
            if re.search(pattern, sql, re.IGNORECASE):
                new_sql = re.sub(pattern, replacement, sql, count=1, flags=re.IGNORECASE)
                if new_sql != sql:
                    return new_sql
        return None
    
    def _remove_alias(self, sql: str) -> Optional[str]:
        """Remove table alias"""
        # Check if there are aliases like T1, T2 or AS alias
        if not re.search(r'\s+AS\s+T\d+', sql, re.IGNORECASE):
            return None
        # Remove AS T1, AS T2 etc
        new_sql = re.sub(r'\s+AS\s+T\d+', '', sql, flags=re.IGNORECASE)
        # Also remove T1., T2. prefixes
        new_sql = re.sub(r'\bT\d+\.', '', new_sql)
        if new_sql != sql:
            return new_sql
        return None
    
    def generate_error(self, sql: str, schema: dict) -> Optional[Dict[str, str]]:
        """
        Generate an error version of the SQL with error info
        Returns None if no error could be generated
        """
        # Shuffle patterns to get variety
        patterns = self.ERROR_PATTERNS.copy()
        random.shuffle(patterns)
        
        for pattern in patterns:
            try:
                error_sql = pattern["apply"](self, sql, schema)
                if error_sql and error_sql != sql:
                    return {
                        "error_sql": error_sql,
                        "error_type": pattern["name"],
                        "error_msg": pattern["error_msg"],
                        "description": pattern["description"]
                    }
            except Exception:
                continue
        
        return None


class CaseExtractor:
    """Extract success and failure cases from datasets"""
    
    def __init__(self, 
                 bird_train_path: str = None,
                 bird_tables_path: str = None,
                 spider_train_path: str = None,
                 spider_tables_path: str = None):
        
        self.bird_train_path = bird_train_path or "data/bird/bird_train/train.json"
        self.bird_tables_path = bird_tables_path or "data/bird/bird_train/train_tables.json"
        self.spider_train_path = spider_train_path or "data/spider/train_spider.json"
        self.spider_tables_path = spider_tables_path or "data/spider/tables.json"
        
        self.error_generator = SQLErrorGenerator()
        
        # Load schema information
        self.bird_schemas = self._load_schemas(self.bird_tables_path)
        self.spider_schemas = self._load_schemas(self.spider_tables_path)
    
    def _load_schemas(self, path: str) -> Dict[str, dict]:
        """Load database schemas from tables.json"""
        schemas = {}
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for item in data:
                    schemas[item['db_id']] = item
            except Exception as e:
                print(f"Warning: Failed to load schemas from {path}: {e}")
        return schemas
    
    def _build_schema_description(self, schema: dict) -> str:
        """Build a simple schema description string"""
        if not schema:
            return ""
        
        table_names = schema.get('table_names_original', [])
        column_names = schema.get('column_names_original', [])
        
        desc_parts = []
        for idx, table_name in enumerate(table_names):
            cols = [col[1] for col in column_names if col[0] == idx]
            if cols:
                desc_parts.append(f"# Table: {table_name}\n[{', '.join(cols)}]")
        
        return '\n'.join(desc_parts)
    
    def _extract_reasoning_steps(self, sql: str, question: str) -> List[str]:
        """Generate simple reasoning steps based on SQL structure"""
        steps = []
        sql_upper = sql.upper()
        
        # Analyze SQL structure
        if 'JOIN' in sql_upper:
            steps.append("Step: Identify tables that need to be joined based on the question")
        
        if 'WHERE' in sql_upper:
            steps.append("Step: Add filtering conditions from the question")
        
        if 'GROUP BY' in sql_upper:
            steps.append("Step: Group results by appropriate columns")
        
        if 'ORDER BY' in sql_upper:
            steps.append("Step: Sort results as required")
        
        if 'LIMIT' in sql_upper:
            steps.append("Step: Limit the number of results")
        
        if any(agg in sql_upper for agg in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']):
            steps.append("Step: Apply aggregate functions for calculations")
        
        return steps if steps else ["Step: Construct the SQL query based on the question"]
    
    def extract_bird_cases(self, 
                           max_success: int = 500, 
                           max_failure: int = 200,
                           seed: int = 42) -> Tuple[List[dict], List[dict]]:
        """Extract cases from BIRD training dataset"""
        success_cases = []
        failure_cases = []
        
        if not os.path.exists(self.bird_train_path):
            print(f"Warning: BIRD training data not found at {self.bird_train_path}")
            return [], []
        
        random.seed(seed)
        
        print("Loading BIRD training data...")
        with open(self.bird_train_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        
        # Shuffle for random selection
        random.shuffle(train_data)
        
        print(f"Processing BIRD cases (max success: {max_success}, max failure: {max_failure})...")
        for item in tqdm(train_data):
            if len(success_cases) >= max_success and len(failure_cases) >= max_failure:
                break
            
            db_id = item.get('db_id', '')
            question = item.get('question', '')
            evidence = item.get('evidence', '')
            sql = item.get('SQL', '')
            
            if not sql or not question:
                continue
            
            schema = self.bird_schemas.get(db_id, {})
            schema_desc = self._build_schema_description(schema)
            
            # Add success case
            if len(success_cases) < max_success:
                success_cases.append({
                    "case_id": f"bird_success_{len(success_cases)}",
                    "case_type": "success",
                    "dataset": "bird",
                    "db_id": db_id,
                    "query": question,
                    "evidence": evidence,
                    "db_schema": schema_desc,
                    "sql": sql,
                    "reasoning_steps": self._extract_reasoning_steps(sql, question),
                    "created_at": datetime.now().isoformat()
                })
            
            # Generate failure case
            if len(failure_cases) < max_failure:
                error_info = self.error_generator.generate_error(sql, schema)
                if error_info:
                    failure_cases.append({
                        "case_id": f"bird_failure_{len(failure_cases)}",
                        "case_type": "failure",
                        "dataset": "bird",
                        "db_id": db_id,
                        "query": question,
                        "evidence": evidence,
                        "db_schema": schema_desc,
                        "sql": error_info["error_sql"],
                        "error_info": f"{error_info['error_msg']} ({error_info['description']})",
                        "correction": sql,
                        "correction_explanation": f"Fixed {error_info['error_type']}: {error_info['description']}",
                        "created_at": datetime.now().isoformat()
                    })
        
        print(f"Extracted {len(success_cases)} success cases and {len(failure_cases)} failure cases from BIRD")
        return success_cases, failure_cases
    
    def extract_spider_cases(self,
                             max_success: int = 500,
                             max_failure: int = 200,
                             seed: int = 42) -> Tuple[List[dict], List[dict]]:
        """Extract cases from Spider training dataset"""
        success_cases = []
        failure_cases = []
        
        if not os.path.exists(self.spider_train_path):
            print(f"Warning: Spider training data not found at {self.spider_train_path}")
            return [], []
        
        random.seed(seed)
        
        print("Loading Spider training data...")
        with open(self.spider_train_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        
        # Shuffle for random selection
        random.shuffle(train_data)
        
        print(f"Processing Spider cases (max success: {max_success}, max failure: {max_failure})...")
        for item in tqdm(train_data):
            if len(success_cases) >= max_success and len(failure_cases) >= max_failure:
                break
            
            db_id = item.get('db_id', '')
            question = item.get('question', '')
            sql = item.get('query', '')
            
            if not sql or not question:
                continue
            
            schema = self.spider_schemas.get(db_id, {})
            schema_desc = self._build_schema_description(schema)
            
            # Add success case
            if len(success_cases) < max_success:
                success_cases.append({
                    "case_id": f"spider_success_{len(success_cases)}",
                    "case_type": "success",
                    "dataset": "spider",
                    "db_id": db_id,
                    "query": question,
                    "evidence": "",  # Spider doesn't have evidence field
                    "db_schema": schema_desc,
                    "sql": sql,
                    "reasoning_steps": self._extract_reasoning_steps(sql, question),
                    "created_at": datetime.now().isoformat()
                })
            
            # Generate failure case
            if len(failure_cases) < max_failure:
                error_info = self.error_generator.generate_error(sql, schema)
                if error_info:
                    failure_cases.append({
                        "case_id": f"spider_failure_{len(failure_cases)}",
                        "case_type": "failure",
                        "dataset": "spider",
                        "db_id": db_id,
                        "query": question,
                        "evidence": "",
                        "db_schema": schema_desc,
                        "sql": error_info["error_sql"],
                        "error_info": f"{error_info['error_msg']} ({error_info['description']})",
                        "correction": sql,
                        "correction_explanation": f"Fixed {error_info['error_type']}: {error_info['description']}",
                        "created_at": datetime.now().isoformat()
                    })
        
        print(f"Extracted {len(success_cases)} success cases and {len(failure_cases)} failure cases from Spider")
        return success_cases, failure_cases


def convert_to_memory_format(cases: List[dict]) -> List[dict]:
    """Convert cases to the format expected by MemoryStore"""
    memory_cases = []
    for case in cases:
        memory_case = {
            "case_id": case["case_id"],
            "case_type": case["case_type"],
            "query": case["query"],
            "evidence": case.get("evidence", ""),
            "db_schema": case["db_schema"],
            "sql": case["sql"],
            "reasoning_steps": case.get("reasoning_steps", []),
            "error_info": case.get("error_info"),
            "correction": case.get("correction"),
            "correction_explanation": case.get("correction_explanation"),
            "metadata": {
                "dataset": case.get("dataset", ""),
                "db_id": case.get("db_id", "")
            },
            "created_at": case["created_at"]
        }
        memory_cases.append(memory_case)
    return memory_cases


def main():
    """Main function to extract and save cases"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract cases from training datasets for RAG memory")
    parser.add_argument("--bird-train", default="data/bird/bird_train/train.json", help="Path to BIRD training JSON")
    parser.add_argument("--bird-tables", default="data/bird/bird_train/train_tables.json", help="Path to BIRD tables JSON")
    parser.add_argument("--spider-train", default="data/spider/train_spider.json", help="Path to Spider training JSON")
    parser.add_argument("--spider-tables", default="data/spider/tables.json", help="Path to Spider tables JSON")
    parser.add_argument("--output-dir", default="memory_store", help="Output directory for memory cases")
    parser.add_argument("--max-success", type=int, default=500, help="Maximum success cases per dataset")
    parser.add_argument("--max-failure", type=int, default=200, help="Maximum failure cases per dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--dataset", choices=["bird", "spider", "both"], default="both", help="Which dataset to process")
    parser.add_argument("--separate", action="store_true", default=True, help="Store cases separately by dataset (default: True)")
    parser.add_argument("--combined", action="store_true", help="Also generate combined files for all datasets")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    extractor = CaseExtractor(
        bird_train_path=args.bird_train,
        bird_tables_path=args.bird_tables,
        spider_train_path=args.spider_train,
        spider_tables_path=args.spider_tables
    )
    
    all_success_cases = []
    all_failure_cases = []
    
    # Extract from BIRD
    if args.dataset in ["bird", "both"]:
        bird_success, bird_failure = extractor.extract_bird_cases(
            max_success=args.max_success,
            max_failure=args.max_failure,
            seed=args.seed
        )
        
        # Convert to memory format
        bird_success_memory = convert_to_memory_format(bird_success)
        bird_failure_memory = convert_to_memory_format(bird_failure)
        
        # Save BIRD cases separately
        if args.separate:
            bird_success_path = os.path.join(args.output_dir, "bird_success_cases.json")
            bird_failure_path = os.path.join(args.output_dir, "bird_failure_cases.json")
            
            with open(bird_success_path, 'w', encoding='utf-8') as f:
                json.dump(bird_success_memory, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(bird_success_memory)} BIRD success cases to {bird_success_path}")
            
            with open(bird_failure_path, 'w', encoding='utf-8') as f:
                json.dump(bird_failure_memory, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(bird_failure_memory)} BIRD failure cases to {bird_failure_path}")
        
        all_success_cases.extend(bird_success)
        all_failure_cases.extend(bird_failure)
    
    # Extract from Spider
    if args.dataset in ["spider", "both"]:
        spider_success, spider_failure = extractor.extract_spider_cases(
            max_success=args.max_success,
            max_failure=args.max_failure,
            seed=args.seed
        )
        
        # Convert to memory format
        spider_success_memory = convert_to_memory_format(spider_success)
        spider_failure_memory = convert_to_memory_format(spider_failure)
        
        # Save Spider cases separately
        if args.separate:
            spider_success_path = os.path.join(args.output_dir, "spider_success_cases.json")
            spider_failure_path = os.path.join(args.output_dir, "spider_failure_cases.json")
            
            with open(spider_success_path, 'w', encoding='utf-8') as f:
                json.dump(spider_success_memory, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(spider_success_memory)} Spider success cases to {spider_success_path}")
            
            with open(spider_failure_path, 'w', encoding='utf-8') as f:
                json.dump(spider_failure_memory, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(spider_failure_memory)} Spider failure cases to {spider_failure_path}")
        
        all_success_cases.extend(spider_success)
        all_failure_cases.extend(spider_failure)
    
    # Save combined files if requested
    if args.combined:
        success_memory = convert_to_memory_format(all_success_cases)
        failure_memory = convert_to_memory_format(all_failure_cases)
        
        success_path = os.path.join(args.output_dir, "success_cases.json")
        failure_path = os.path.join(args.output_dir, "failure_cases.json")
        
        with open(success_path, 'w', encoding='utf-8') as f:
            json.dump(success_memory, f, ensure_ascii=False, indent=2)
        print(f"\nSaved {len(success_memory)} combined success cases to {success_path}")
        
        with open(failure_path, 'w', encoding='utf-8') as f:
            json.dump(failure_memory, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(failure_memory)} combined failure cases to {failure_path}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("Extraction Summary")
    print("=" * 50)
    
    if args.dataset in ["bird", "both"]:
        bird_s = len([c for c in all_success_cases if c.get('dataset') == 'bird'])
        bird_f = len([c for c in all_failure_cases if c.get('dataset') == 'bird'])
        print(f"BIRD:   {bird_s} success cases, {bird_f} failure cases")
    
    if args.dataset in ["spider", "both"]:
        spider_s = len([c for c in all_success_cases if c.get('dataset') == 'spider'])
        spider_f = len([c for c in all_failure_cases if c.get('dataset') == 'spider'])
        print(f"Spider: {spider_s} success cases, {spider_f} failure cases")
    
    print(f"\nTotal:  {len(all_success_cases)} success cases, {len(all_failure_cases)} failure cases")
    print(f"Output directory: {args.output_dir}")
    
    print("\nGenerated files:")
    if args.separate:
        if args.dataset in ["bird", "both"]:
            print(f"  - bird_success_cases.json")
            print(f"  - bird_failure_cases.json")
        if args.dataset in ["spider", "both"]:
            print(f"  - spider_success_cases.json")
            print(f"  - spider_failure_cases.json")
    if args.combined:
        print(f"  - success_cases.json (combined)")
        print(f"  - failure_cases.json (combined)")


if __name__ == "__main__":
    main()
