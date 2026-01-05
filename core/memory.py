# -*- coding: utf-8 -*-
"""
Memory Module for RAG (Retrieval-Augmented Generation)

This module provides a flexible memory system for storing and retrieving:
- Success cases: Used by Decomposer for step-by-step SQL reasoning
- Failure cases: Used by Refiner for SQL correction and refinement

The module supports multiple retrieval strategies:
- Embedding-based similarity search
- Keyword matching
- Hybrid approach (combination of above)
"""

import os
import json
import hashlib
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import sqlite3
import re

# Try to import embedding libraries
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


@dataclass
class MemoryCase:
    """Represents a single memory case (success or failure)"""
    case_id: str
    case_type: str  # 'success' or 'failure'
    query: str  # Natural language question
    evidence: str  # Additional context/evidence
    db_schema: str  # Database schema description
    sql: str  # Generated SQL
    reasoning_steps: List[str] = field(default_factory=list)  # Sub-questions and steps
    error_info: Optional[str] = None  # Error message for failure cases
    correction: Optional[str] = None  # Corrected SQL for failure cases
    correction_explanation: Optional[str] = None  # Explanation of the correction
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryCase':
        return cls(**data)
    
    def get_case_hash(self) -> str:
        """Generate a unique hash for the case"""
        content = f"{self.query}|{self.db_schema}|{self.sql}"
        return hashlib.md5(content.encode()).hexdigest()


class RetrievalStrategy(ABC):
    """Abstract base class for retrieval strategies"""
    
    @abstractmethod
    def retrieve(self, query: str, cases: List[MemoryCase], top_k: int = 3) -> List[Tuple[MemoryCase, float]]:
        """
        Retrieve relevant cases based on the query
        
        Args:
            query: The query string to match against
            cases: List of memory cases to search through
            top_k: Number of top results to return
            
        Returns:
            List of tuples (case, score) sorted by relevance
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return the name of the strategy"""
        pass


class KeywordRetrievalStrategy(RetrievalStrategy):
    """
    Simple keyword-based retrieval strategy
    Uses TF-IDF-like scoring based on keyword overlap
    """
    
    def __init__(self):
        self.stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
            'and', 'or', 'but', 'if', 'then', 'else', 'when', 'where', 'what',
            'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'of', 'in',
            'to', 'for', 'with', 'on', 'at', 'by', 'from', 'as', 'into', 'through',
            'select', 'from', 'where', 'join', 'inner', 'left', 'right', 'outer',
            'group', 'order', 'by', 'having', 'limit', 'offset', 'union', 'all'
        }
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize and clean text"""
        # Convert to lowercase and extract words
        words = re.findall(r'\b\w+\b', text.lower())
        # Remove stop words and short words
        return [w for w in words if w not in self.stop_words and len(w) > 2]
    
    def _compute_similarity(self, query_tokens: List[str], case_tokens: List[str]) -> float:
        """Compute Jaccard similarity between token sets"""
        if not query_tokens or not case_tokens:
            return 0.0
        query_set = set(query_tokens)
        case_set = set(case_tokens)
        intersection = query_set & case_set
        union = query_set | case_set
        return len(intersection) / len(union) if union else 0.0
    
    def retrieve(self, query: str, cases: List[MemoryCase], top_k: int = 3) -> List[Tuple[MemoryCase, float]]:
        if not cases:
            return []
        
        query_tokens = self._tokenize(query)
        scored_cases = []
        
        for case in cases:
            # Combine query, evidence, and schema for matching
            case_text = f"{case.query} {case.evidence} {case.db_schema}"
            case_tokens = self._tokenize(case_text)
            score = self._compute_similarity(query_tokens, case_tokens)
            scored_cases.append((case, score))
        
        # Sort by score descending and return top_k
        scored_cases.sort(key=lambda x: x[1], reverse=True)
        return scored_cases[:top_k]
    
    def get_strategy_name(self) -> str:
        return "keyword"


class EmbeddingRetrievalStrategy(RetrievalStrategy):
    """
    Embedding-based retrieval strategy using sentence transformers
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        if not HAS_SENTENCE_TRANSFORMERS or not HAS_NUMPY:
            raise ImportError(
                "EmbeddingRetrievalStrategy requires 'sentence-transformers' and 'numpy'. "
                "Install them with: pip install sentence-transformers numpy"
            )
        self.model = SentenceTransformer(model_name)
        self._embedding_cache: Dict[str, Any] = {}
    
    def _get_embedding(self, text: str) -> Any:
        """Get embedding for text, with caching"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash not in self._embedding_cache:
            self._embedding_cache[text_hash] = self.model.encode(text)
        return self._embedding_cache[text_hash]
    
    def _cosine_similarity(self, vec1: Any, vec2: Any) -> float:
        """Compute cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return float(dot_product / (norm1 * norm2)) if norm1 * norm2 > 0 else 0.0
    
    def retrieve(self, query: str, cases: List[MemoryCase], top_k: int = 3) -> List[Tuple[MemoryCase, float]]:
        if not cases:
            return []
        
        query_embedding = self._get_embedding(query)
        scored_cases = []
        
        for case in cases:
            case_text = f"{case.query} {case.evidence}"
            case_embedding = self._get_embedding(case_text)
            score = self._cosine_similarity(query_embedding, case_embedding)
            scored_cases.append((case, score))
        
        scored_cases.sort(key=lambda x: x[1], reverse=True)
        return scored_cases[:top_k]
    
    def get_strategy_name(self) -> str:
        return "embedding"


class HybridRetrievalStrategy(RetrievalStrategy):
    """
    Hybrid retrieval strategy combining keyword and embedding approaches
    """
    
    def __init__(self, 
                 keyword_weight: float = 0.3,
                 embedding_weight: float = 0.7,
                 embedding_model: str = 'all-MiniLM-L6-v2'):
        self.keyword_strategy = KeywordRetrievalStrategy()
        self.keyword_weight = keyword_weight
        self.embedding_weight = embedding_weight
        
        # Try to initialize embedding strategy, fall back to keyword-only if unavailable
        try:
            self.embedding_strategy = EmbeddingRetrievalStrategy(embedding_model)
            self.has_embedding = True
        except ImportError:
            print("Warning: Embedding model not available, using keyword-only retrieval")
            self.embedding_strategy = None
            self.has_embedding = False
            self.keyword_weight = 1.0
    
    def retrieve(self, query: str, cases: List[MemoryCase], top_k: int = 3) -> List[Tuple[MemoryCase, float]]:
        if not cases:
            return []
        
        # Get keyword scores
        keyword_results = self.keyword_strategy.retrieve(query, cases, len(cases))
        keyword_scores = {case.case_id: score for case, score in keyword_results}
        
        # Get embedding scores if available
        if self.has_embedding:
            embedding_results = self.embedding_strategy.retrieve(query, cases, len(cases))
            embedding_scores = {case.case_id: score for case, score in embedding_results}
        else:
            embedding_scores = {}
        
        # Combine scores
        scored_cases = []
        for case in cases:
            kw_score = keyword_scores.get(case.case_id, 0.0)
            emb_score = embedding_scores.get(case.case_id, 0.0) if self.has_embedding else 0.0
            combined_score = self.keyword_weight * kw_score + self.embedding_weight * emb_score
            scored_cases.append((case, combined_score))
        
        scored_cases.sort(key=lambda x: x[1], reverse=True)
        return scored_cases[:top_k]
    
    def get_strategy_name(self) -> str:
        return "hybrid"


class MemoryStore:
    """
    Main memory storage class that manages success and failure cases
    """
    
    def __init__(self, 
                 storage_path: str = "./memory_store",
                 retrieval_strategy: str = "keyword",
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 max_cases: int = 1000,
                 dataset_name: str = None,
                 success_cases_file: str = None,
                 failure_cases_file: str = None):
        """
        Initialize the memory store
        
        Args:
            storage_path: Directory path to store memory cases
            retrieval_strategy: Strategy for retrieval ('keyword', 'embedding', 'hybrid')
            embedding_model: Model name for embedding strategy
            max_cases: Maximum number of cases to keep per type
            dataset_name: Dataset name ('bird', 'spider') to load dataset-specific files
            success_cases_file: Custom filename for success cases (overrides dataset_name)
            failure_cases_file: Custom filename for failure cases (overrides dataset_name)
        """
        self.storage_path = storage_path
        self.max_cases = max_cases
        self.dataset_name = dataset_name
        
        # Initialize storage directories
        os.makedirs(storage_path, exist_ok=True)
        
        # Determine file names based on dataset or custom names
        if success_cases_file:
            success_filename = success_cases_file
        elif dataset_name:
            success_filename = f"{dataset_name}_success_cases.json"
        else:
            success_filename = "success_cases.json"
        
        if failure_cases_file:
            failure_filename = failure_cases_file
        elif dataset_name:
            failure_filename = f"{dataset_name}_failure_cases.json"
        else:
            failure_filename = "failure_cases.json"
        
        self.success_cases_path = os.path.join(storage_path, success_filename)
        self.failure_cases_path = os.path.join(storage_path, failure_filename)
        
        # Load existing cases
        self.success_cases: List[MemoryCase] = self._load_cases(self.success_cases_path)
        self.failure_cases: List[MemoryCase] = self._load_cases(self.failure_cases_path)
        
        print(f"MemoryStore initialized: {len(self.success_cases)} success cases, {len(self.failure_cases)} failure cases")
        if dataset_name:
            print(f"  Dataset: {dataset_name}")
        print(f"  Success file: {self.success_cases_path}")
        print(f"  Failure file: {self.failure_cases_path}")
        
        # Initialize retrieval strategy
        self.strategy = self._create_strategy(retrieval_strategy, embedding_model)
        
        # Case ID tracking
        self._existing_hashes: set = set()
        self._update_hash_set()
    
    def switch_dataset(self, dataset_name: str):
        """
        Switch to a different dataset's memory files
        
        Args:
            dataset_name: Dataset name ('bird', 'spider')
        """
        self.dataset_name = dataset_name
        self.success_cases_path = os.path.join(self.storage_path, f"{dataset_name}_success_cases.json")
        self.failure_cases_path = os.path.join(self.storage_path, f"{dataset_name}_failure_cases.json")
        
        # Reload cases
        self.success_cases = self._load_cases(self.success_cases_path)
        self.failure_cases = self._load_cases(self.failure_cases_path)
        self._update_hash_set()
        
        print(f"Switched to dataset '{dataset_name}': {len(self.success_cases)} success, {len(self.failure_cases)} failure cases")
    
    def _create_strategy(self, strategy_name: str, embedding_model: str) -> RetrievalStrategy:
        """Create a retrieval strategy instance"""
        if strategy_name == "keyword":
            return KeywordRetrievalStrategy()
        elif strategy_name == "embedding":
            try:
                return EmbeddingRetrievalStrategy(embedding_model)
            except ImportError:
                print("Warning: Embedding strategy not available, falling back to keyword")
                return KeywordRetrievalStrategy()
        elif strategy_name == "hybrid":
            return HybridRetrievalStrategy(embedding_model=embedding_model)
        else:
            print(f"Warning: Unknown strategy '{strategy_name}', using keyword")
            return KeywordRetrievalStrategy()
    
    def set_retrieval_strategy(self, strategy_name: str, embedding_model: str = 'all-MiniLM-L6-v2'):
        """Change the retrieval strategy at runtime"""
        self.strategy = self._create_strategy(strategy_name, embedding_model)
    
    def _load_cases(self, path: str) -> List[MemoryCase]:
        """Load cases from a JSON file"""
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return [MemoryCase.from_dict(d) for d in data]
            except Exception as e:
                print(f"Warning: Failed to load cases from {path}: {e}")
                return []
        return []
    
    def _save_cases(self, cases: List[MemoryCase], path: str):
        """Save cases to a JSON file"""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump([c.to_dict() for c in cases], f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save cases to {path}: {e}")
    
    def _update_hash_set(self):
        """Update the set of existing case hashes"""
        self._existing_hashes = set()
        for case in self.success_cases + self.failure_cases:
            self._existing_hashes.add(case.get_case_hash())
    
    def _trim_cases(self, cases: List[MemoryCase]) -> List[MemoryCase]:
        """Trim cases list to max_cases, keeping most recent"""
        if len(cases) > self.max_cases:
            # Sort by creation time and keep most recent
            cases.sort(key=lambda x: x.created_at, reverse=True)
            return cases[:self.max_cases]
        return cases
    
    def add_success_case(self,
                         query: str,
                         evidence: str,
                         db_schema: str,
                         sql: str,
                         reasoning_steps: List[str] = None,
                         metadata: Dict[str, Any] = None) -> Optional[str]:
        """
        Add a success case to the memory store
        
        Args:
            query: Natural language question
            evidence: Additional context/evidence
            db_schema: Database schema description
            sql: Successfully generated SQL
            reasoning_steps: List of reasoning steps/sub-questions
            metadata: Additional metadata
            
        Returns:
            Case ID if added successfully, None if duplicate
        """
        case = MemoryCase(
            case_id=f"success_{len(self.success_cases)}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            case_type='success',
            query=query,
            evidence=evidence or '',
            db_schema=db_schema,
            sql=sql,
            reasoning_steps=reasoning_steps or [],
            metadata=metadata or {}
        )
        
        # Check for duplicates
        case_hash = case.get_case_hash()
        if case_hash in self._existing_hashes:
            return None
        
        self.success_cases.append(case)
        self._existing_hashes.add(case_hash)
        self.success_cases = self._trim_cases(self.success_cases)
        self._save_cases(self.success_cases, self.success_cases_path)
        
        return case.case_id
    
    def add_failure_case(self,
                         query: str,
                         evidence: str,
                         db_schema: str,
                         sql: str,
                         error_info: str,
                         correction: str = None,
                         correction_explanation: str = None,
                         metadata: Dict[str, Any] = None) -> Optional[str]:
        """
        Add a failure case to the memory store
        
        Args:
            query: Natural language question
            evidence: Additional context/evidence
            db_schema: Database schema description
            sql: Failed SQL query
            error_info: Error message or description
            correction: Corrected SQL (if available)
            correction_explanation: Explanation of the correction
            metadata: Additional metadata
            
        Returns:
            Case ID if added successfully, None if duplicate
        """
        case = MemoryCase(
            case_id=f"failure_{len(self.failure_cases)}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            case_type='failure',
            query=query,
            evidence=evidence or '',
            db_schema=db_schema,
            sql=sql,
            error_info=error_info,
            correction=correction,
            correction_explanation=correction_explanation,
            metadata=metadata or {}
        )
        
        # Check for duplicates
        case_hash = case.get_case_hash()
        if case_hash in self._existing_hashes:
            return None
        
        self.failure_cases.append(case)
        self._existing_hashes.add(case_hash)
        self.failure_cases = self._trim_cases(self.failure_cases)
        self._save_cases(self.failure_cases, self.failure_cases_path)
        
        return case.case_id
    
    def retrieve_success_cases(self, 
                               query: str, 
                               evidence: str = '',
                               top_k: int = 3) -> List[Tuple[MemoryCase, float]]:
        """
        Retrieve relevant success cases for the given query
        
        Args:
            query: Natural language question
            evidence: Additional context/evidence
            top_k: Number of cases to retrieve
            
        Returns:
            List of (case, score) tuples
        """
        search_text = f"{query} {evidence}" if evidence else query
        return self.strategy.retrieve(search_text, self.success_cases, top_k)
    
    def retrieve_failure_cases(self,
                               query: str,
                               evidence: str = '',
                               error_info: str = '',
                               top_k: int = 3) -> List[Tuple[MemoryCase, float]]:
        """
        Retrieve relevant failure cases for the given query
        
        Args:
            query: Natural language question
            evidence: Additional context/evidence
            error_info: Current error information
            top_k: Number of cases to retrieve
            
        Returns:
            List of (case, score) tuples
        """
        search_text = f"{query} {evidence} {error_info}" if error_info else f"{query} {evidence}"
        return self.strategy.retrieve(search_text, self.failure_cases, top_k)
    
    def format_success_cases_for_prompt(self, 
                                        cases: List[Tuple[MemoryCase, float]],
                                        max_cases: int = 2) -> str:
        """
        Format success cases for inclusion in a prompt
        
        Args:
            cases: List of (case, score) tuples
            max_cases: Maximum number of cases to include
            
        Returns:
            Formatted string for prompt
        """
        if not cases:
            return ""
        
        formatted_parts = []
        for i, (case, score) in enumerate(cases[:max_cases]):
            part = f"【Reference Example {i + 1}】\n"
            part += f"Question: {case.query}\n"
            if case.evidence:
                part += f"Evidence: {case.evidence}\n"
            
            if case.reasoning_steps:
                part += "Reasoning Steps:\n"
                for step in case.reasoning_steps:
                    part += f"  {step}\n"
            
            part += f"SQL Solution:\n```sql\n{case.sql}\n```\n"
            formatted_parts.append(part)
        
        return "\n".join(formatted_parts)
    
    def format_failure_cases_for_prompt(self,
                                        cases: List[Tuple[MemoryCase, float]],
                                        max_cases: int = 2) -> str:
        """
        Format failure cases for inclusion in a prompt
        
        Args:
            cases: List of (case, score) tuples
            max_cases: Maximum number of cases to include
            
        Returns:
            Formatted string for prompt
        """
        if not cases:
            return ""
        
        formatted_parts = []
        for i, (case, score) in enumerate(cases[:max_cases]):
            part = f"【Error Case {i + 1}】\n"
            part += f"Question: {case.query}\n"
            part += f"Incorrect SQL:\n```sql\n{case.sql}\n```\n"
            part += f"Error: {case.error_info}\n"
            
            if case.correction:
                part += f"Corrected SQL:\n```sql\n{case.correction}\n```\n"
            if case.correction_explanation:
                part += f"Explanation: {case.correction_explanation}\n"
            
            formatted_parts.append(part)
        
        return "\n".join(formatted_parts)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory store statistics"""
        return {
            "success_cases_count": len(self.success_cases),
            "failure_cases_count": len(self.failure_cases),
            "total_cases": len(self.success_cases) + len(self.failure_cases),
            "retrieval_strategy": self.strategy.get_strategy_name(),
            "max_cases": self.max_cases,
            "storage_path": self.storage_path
        }
    
    def clear_all(self):
        """Clear all cases from memory"""
        self.success_cases = []
        self.failure_cases = []
        self._existing_hashes = set()
        self._save_cases([], self.success_cases_path)
        self._save_cases([], self.failure_cases_path)
    
    def clear_success_cases(self):
        """Clear only success cases"""
        for case in self.success_cases:
            self._existing_hashes.discard(case.get_case_hash())
        self.success_cases = []
        self._save_cases([], self.success_cases_path)
    
    def clear_failure_cases(self):
        """Clear only failure cases"""
        for case in self.failure_cases:
            self._existing_hashes.discard(case.get_case_hash())
        self.failure_cases = []
        self._save_cases([], self.failure_cases_path)


class MemoryConfig:
    """Configuration class for memory module"""
    
    def __init__(self,
                 enabled: bool = True,
                 storage_path: str = "./memory_store",
                 retrieval_strategy: str = "keyword",
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 max_cases: int = 1000,
                 decomposer_top_k: int = 2,
                 refiner_top_k: int = 2,
                 min_similarity_threshold: float = 0.1,
                 auto_save_success: bool = True,
                 auto_save_failure: bool = True,
                 dataset_name: str = None,
                 success_cases_file: str = None,
                 failure_cases_file: str = None):
        """
        Initialize memory configuration
        
        Args:
            enabled: Whether memory is enabled
            storage_path: Directory path to store memory cases
            retrieval_strategy: Strategy for retrieval ('keyword', 'embedding', 'hybrid')
            embedding_model: Model name for embedding strategy
            max_cases: Maximum number of cases to keep per type
            decomposer_top_k: Number of success cases for Decomposer
            refiner_top_k: Number of failure cases for Refiner
            min_similarity_threshold: Minimum similarity score to include a case
            auto_save_success: Automatically save successful cases
            auto_save_failure: Automatically save failure cases
            dataset_name: Dataset name ('bird', 'spider') to load dataset-specific files
            success_cases_file: Custom filename for success cases (overrides dataset_name)
            failure_cases_file: Custom filename for failure cases (overrides dataset_name)
        """
        self.enabled = enabled
        self.storage_path = storage_path
        self.retrieval_strategy = retrieval_strategy
        self.embedding_model = embedding_model
        self.max_cases = max_cases
        self.decomposer_top_k = decomposer_top_k
        self.refiner_top_k = refiner_top_k
        self.min_similarity_threshold = min_similarity_threshold
        self.auto_save_success = auto_save_success
        self.auto_save_failure = auto_save_failure
        self.dataset_name = dataset_name
        self.success_cases_file = success_cases_file
        self.failure_cases_file = failure_cases_file
    
    def to_dict(self) -> Dict:
        return {
            'enabled': self.enabled,
            'storage_path': self.storage_path,
            'retrieval_strategy': self.retrieval_strategy,
            'embedding_model': self.embedding_model,
            'max_cases': self.max_cases,
            'decomposer_top_k': self.decomposer_top_k,
            'refiner_top_k': self.refiner_top_k,
            'min_similarity_threshold': self.min_similarity_threshold,
            'auto_save_success': self.auto_save_success,
            'auto_save_failure': self.auto_save_failure,
            'dataset_name': self.dataset_name,
            'success_cases_file': self.success_cases_file,
            'failure_cases_file': self.failure_cases_file
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryConfig':
        return cls(**data)
    
    @classmethod
    def from_json_file(cls, path: str) -> 'MemoryConfig':
        """Load configuration from a JSON file"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def save_to_json(self, path: str):
        """Save configuration to a JSON file"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)


# Global memory store instance (lazily initialized)
_global_memory_store: Optional[MemoryStore] = None
_global_memory_config: Optional[MemoryConfig] = None


def get_memory_store(config: MemoryConfig = None) -> MemoryStore:
    """
    Get or create the global memory store instance
    
    Args:
        config: Optional configuration (used for initialization)
        
    Returns:
        MemoryStore instance
    """
    global _global_memory_store, _global_memory_config
    
    if config is not None:
        _global_memory_config = config
    
    if _global_memory_store is None:
        if _global_memory_config is None:
            _global_memory_config = MemoryConfig()
        
        _global_memory_store = MemoryStore(
            storage_path=_global_memory_config.storage_path,
            retrieval_strategy=_global_memory_config.retrieval_strategy,
            embedding_model=_global_memory_config.embedding_model,
            max_cases=_global_memory_config.max_cases,
            dataset_name=_global_memory_config.dataset_name,
            success_cases_file=_global_memory_config.success_cases_file,
            failure_cases_file=_global_memory_config.failure_cases_file
        )
    
    return _global_memory_store


def get_memory_config() -> MemoryConfig:
    """Get the global memory configuration"""
    global _global_memory_config
    if _global_memory_config is None:
        _global_memory_config = MemoryConfig()
    return _global_memory_config


def set_memory_config(config: MemoryConfig):
    """Set the global memory configuration (resets memory store)"""
    global _global_memory_store, _global_memory_config
    _global_memory_config = config
    _global_memory_store = None  # Reset to force re-initialization


def reset_memory_store():
    """Reset the global memory store"""
    global _global_memory_store
    _global_memory_store = None
