# -*- coding: utf-8 -*-
"""
Test script for the Memory Module

This script demonstrates and tests the functionality of the memory module.
"""

import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory import (
    MemoryStore, MemoryConfig, MemoryCase,
    KeywordRetrievalStrategy, HybridRetrievalStrategy,
    get_memory_store, get_memory_config, set_memory_config, reset_memory_store
)


def test_memory_config():
    """Test MemoryConfig creation and serialization"""
    print("=" * 50)
    print("Testing MemoryConfig...")
    
    # Create config with default values
    config = MemoryConfig()
    assert config.enabled == True
    assert config.retrieval_strategy == "keyword"
    print("  ✓ Default config created")
    
    # Create config with custom values
    config = MemoryConfig(
        enabled=True,
        storage_path="./test_memory_store",
        retrieval_strategy="hybrid",
        decomposer_top_k=3,
        refiner_top_k=3
    )
    assert config.retrieval_strategy == "hybrid"
    print("  ✓ Custom config created")
    
    # Test serialization
    config_dict = config.to_dict()
    config2 = MemoryConfig.from_dict(config_dict)
    assert config2.retrieval_strategy == "hybrid"
    print("  ✓ Config serialization works")
    
    print("MemoryConfig tests passed!\n")


def test_keyword_retrieval():
    """Test keyword-based retrieval strategy"""
    print("=" * 50)
    print("Testing KeywordRetrievalStrategy...")
    
    strategy = KeywordRetrievalStrategy()
    
    # Create test cases
    cases = [
        MemoryCase(
            case_id="test_1",
            case_type="success",
            query="What is the total sales amount for each customer?",
            evidence="",
            db_schema="# Table: customers, orders",
            sql="SELECT customer_id, SUM(amount) FROM orders GROUP BY customer_id"
        ),
        MemoryCase(
            case_id="test_2",
            case_type="success",
            query="Find all products with price greater than 100",
            evidence="",
            db_schema="# Table: products",
            sql="SELECT * FROM products WHERE price > 100"
        ),
        MemoryCase(
            case_id="test_3",
            case_type="success",
            query="Get the average order amount per month",
            evidence="",
            db_schema="# Table: orders",
            sql="SELECT MONTH(order_date), AVG(amount) FROM orders GROUP BY MONTH(order_date)"
        ),
    ]
    
    # Test retrieval
    results = strategy.retrieve("total sales for customers", cases, top_k=2)
    assert len(results) <= 2
    assert results[0][0].case_id == "test_1"  # Should match first case best
    print(f"  ✓ Retrieved {len(results)} cases")
    print(f"    Best match: {results[0][0].case_id} (score: {results[0][1]:.3f})")
    
    print("KeywordRetrievalStrategy tests passed!\n")


def test_memory_store():
    """Test MemoryStore basic operations"""
    print("=" * 50)
    print("Testing MemoryStore...")
    
    # Reset and create a new memory store for testing
    reset_memory_store()
    config = MemoryConfig(
        storage_path="./test_memory_store",
        retrieval_strategy="keyword"
    )
    set_memory_config(config)
    
    store = get_memory_store()
    
    # Clear any existing data
    store.clear_all()
    
    # Add success cases
    case_id_1 = store.add_success_case(
        query="What is the name of the customer with the highest order count?",
        evidence="Order count = COUNT(order_id)",
        db_schema="# Table: customers\n# Table: orders",
        sql="SELECT c.name FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id ORDER BY COUNT(o.id) DESC LIMIT 1",
        reasoning_steps=[
            "Step 1: Join customers and orders tables",
            "Step 2: Group by customer and count orders",
            "Step 3: Order by count descending and limit 1"
        ]
    )
    print(f"  ✓ Added success case: {case_id_1}")
    
    case_id_2 = store.add_success_case(
        query="List all products with inventory below 10",
        evidence="",
        db_schema="# Table: products",
        sql="SELECT * FROM products WHERE inventory < 10"
    )
    print(f"  ✓ Added success case: {case_id_2}")
    
    # Add failure cases
    case_id_3 = store.add_failure_case(
        query="Get the total revenue per category",
        evidence="",
        db_schema="# Table: products, categories, orders",
        sql="SELECT category, SUM(price) FROM products",  # Missing GROUP BY
        error_info="column 'category' must appear in GROUP BY clause",
        correction="SELECT category, SUM(price) FROM products GROUP BY category",
        correction_explanation="Added GROUP BY clause for non-aggregated column"
    )
    print(f"  ✓ Added failure case: {case_id_3}")
    
    # Test retrieval
    success_results = store.retrieve_success_cases("customer with most orders", top_k=1)
    assert len(success_results) == 1
    print(f"  ✓ Retrieved success case with score: {success_results[0][1]:.3f}")
    
    failure_results = store.retrieve_failure_cases("revenue by category", top_k=1)
    assert len(failure_results) == 1
    print(f"  ✓ Retrieved failure case with score: {failure_results[0][1]:.3f}")
    
    # Test formatting
    formatted_success = store.format_success_cases_for_prompt(success_results, max_cases=1)
    assert "Reference Example" in formatted_success
    print("  ✓ Success cases formatted for prompt")
    
    formatted_failure = store.format_failure_cases_for_prompt(failure_results, max_cases=1)
    assert "Error Case" in formatted_failure
    print("  ✓ Failure cases formatted for prompt")
    
    # Test statistics
    stats = store.get_statistics()
    assert stats["success_cases_count"] == 2
    assert stats["failure_cases_count"] == 1
    print(f"  ✓ Statistics: {stats['success_cases_count']} success, {stats['failure_cases_count']} failure")
    
    # Test duplicate prevention
    duplicate_id = store.add_success_case(
        query="What is the name of the customer with the highest order count?",
        evidence="Order count = COUNT(order_id)",
        db_schema="# Table: customers\n# Table: orders",
        sql="SELECT c.name FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id ORDER BY COUNT(o.id) DESC LIMIT 1"
    )
    assert duplicate_id is None
    print("  ✓ Duplicate prevention works")
    
    # Clean up
    store.clear_all()
    print("  ✓ Cleared all cases")
    
    print("MemoryStore tests passed!\n")


def test_strategy_switching():
    """Test runtime strategy switching"""
    print("=" * 50)
    print("Testing Strategy Switching...")
    
    reset_memory_store()
    config = MemoryConfig(
        storage_path="./test_memory_store",
        retrieval_strategy="keyword"
    )
    set_memory_config(config)
    
    store = get_memory_store()
    assert store.strategy.get_strategy_name() == "keyword"
    print("  ✓ Initial strategy: keyword")
    
    # Switch to hybrid (will fall back to keyword if embedding not available)
    store.set_retrieval_strategy("hybrid")
    strategy_name = store.strategy.get_strategy_name()
    print(f"  ✓ Switched to: {strategy_name}")
    
    # Switch back to keyword
    store.set_retrieval_strategy("keyword")
    assert store.strategy.get_strategy_name() == "keyword"
    print("  ✓ Switched back to: keyword")
    
    print("Strategy Switching tests passed!\n")


def main():
    """Run all tests"""
    print("\n" + "=" * 50)
    print("Memory Module Test Suite")
    print("=" * 50 + "\n")
    
    try:
        test_memory_config()
        test_keyword_retrieval()
        test_memory_store()
        test_strategy_switching()
        
        print("=" * 50)
        print("All tests passed! ✓")
        print("=" * 50)
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Cleanup test directory
    import shutil
    if os.path.exists("./test_memory_store"):
        shutil.rmtree("./test_memory_store")
        print("\nTest directory cleaned up.")


if __name__ == "__main__":
    main()
