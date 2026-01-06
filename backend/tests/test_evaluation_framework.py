"""
Tests for Evaluation Framework

This module tests the comprehensive evaluation framework including:
- Faithfulness evaluation with claim-level analysis
- Relevance assessment with semantic similarity
- Retrieval metrics calculation (precision, recall, F1, MRR, NDCG)
- Comparative evaluation with statistical testing
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime
from typing import List, Dict

from hypothesis import given, strategies as st, assume
from hypothesis import settings

from app.services.evaluation_framework import (
    FaithfulnessEvaluator,
    RelevanceAssessor,
    RetrievalMetricsCalculator,
    ComparativeEvaluator,
    EvaluationFramework,
    FaithfulnessScore,
    RelevanceScore,
    RetrievalMetrics,
    ComparativeResult,
    EvaluationReport
)
from app.services.rag_pipeline import RankedChunk

class TestFaithfulnessEvaluator:
    """Test suite for FaithfulnessEvaluator"""
    
    @pytest.fixture
    def evaluator(self):
        """Create evaluator instance for testing"""
        return FaithfulnessEvaluator()
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample ranked chunks for testing"""
        return [
            RankedChunk(
                chunk_id="chunk1",
                document_id="doc1",
                content="The capital of France is Paris. It is located in northern France.",
                page_number=1,
                vector_score=0.9,
                bm25_score=0.8,
                combined_score=0.85,
                relevance_score=0.9,
                final_score=0.9,
                metadata={"filename": "geography.pdf"}
            ),
            RankedChunk(
                chunk_id="chunk2",
                document_id="doc1",
                content="Paris has a population of approximately 2.2 million people.",
                page_number=2,
                vector_score=0.8,
                bm25_score=0.7,
                combined_score=0.75,
                relevance_score=0.8,
                final_score=0.8,
                metadata={"filename": "geography.pdf"}
            )
        ]
    
    def test_initialization(self, evaluator):
        """Test evaluator initialization"""
        assert evaluator.similarity_threshold == 0.7
        assert evaluator.claim_extraction_method == "sentence_split"
    
    def test_evaluate_faithfulness_empty_response(self, evaluator, sample_chunks):
        """Test faithfulness evaluation with empty response"""
        result = evaluator.evaluate_faithfulness("", sample_chunks)
        
        assert isinstance(result, FaithfulnessScore)
        assert result.overall_score == 0.0
        assert result.total_claims == 0
        assert result.supported_claims == 0
        assert "Empty response" in result.hallucination_indicators
    
    def test_evaluate_faithfulness_no_sources(self, evaluator):
        """Test faithfulness evaluation with no source chunks"""
        response = "Paris is the capital of France."
        result = evaluator.evaluate_faithfulness(response, [])
        
        assert result.overall_score == 0.0
        assert result.total_claims == 0
        assert result.unsupported_claims == [response]
        assert "No source documents provided" in result.hallucination_indicators
    
    def test_evaluate_faithfulness_supported_claims(self, evaluator, sample_chunks):
        """Test faithfulness evaluation with supported claims"""
        response = "Paris is the capital of France. It is located in northern France."
        result = evaluator.evaluate_faithfulness(response, sample_chunks)
        
        assert isinstance(result, FaithfulnessScore)
        assert result.overall_score > 0.0
        assert result.total_claims > 0
        assert result.confidence > 0.0
    
    def test_claim_extraction_by_sentences(self, evaluator):
        """Test claim extraction by sentence splitting"""
        response = "Paris is the capital. It has many museums. The Eiffel Tower is famous."
        claims = evaluator._extract_claims_by_sentences(response)
        
        assert len(claims) == 3
        assert "Paris is the capital" in claims
        assert "It has many museums" in claims
        assert "The Eiffel Tower is famous" in claims
    
    def test_lexical_claim_support(self, evaluator):
        """Test lexical claim support evaluation"""
        claim = "Paris is the capital of France"
        source_texts = ["The capital of France is Paris", "London is in England"]
        
        score = evaluator._evaluate_claim_support_lexical(claim, source_texts)
        assert score > 0.0
    
    def test_hallucination_detection(self, evaluator, sample_chunks):
        """Test hallucination indicator detection"""
        response = "Paris has 5.7 million people. It was definitely founded in 1234."
        result = evaluator.evaluate_faithfulness(response, sample_chunks)
        
        # Should detect unsupported numbers and definitive statements
        assert len(result.hallucination_indicators) > 0
    
    def test_batch_evaluation(self, evaluator, sample_chunks):
        """Test batch faithfulness evaluation"""
        responses_and_sources = [
            ("Paris is the capital of France.", sample_chunks),
            ("London is the capital of England.", sample_chunks)
        ]
        
        results = evaluator.batch_evaluate_faithfulness(responses_and_sources)
        
        assert len(results) == 2
        assert all(isinstance(result, FaithfulnessScore) for result in results)
    
    def test_faithfulness_summary(self, evaluator):
        """Test faithfulness summary generation"""
        scores = [
            FaithfulnessScore(0.8, [0.8, 0.9], 2, 2, [], 0.85, [], 0.9),
            FaithfulnessScore(0.6, [0.5, 0.7], 1, 2, ["unsupported claim"], 0.65, ["indicator"], 0.7)
        ]
        
        summary = evaluator.get_faithfulness_summary(scores)
        
        assert summary["total_evaluations"] == 2
        assert "avg_faithfulness_score" in summary
        assert "total_claims" in summary
        assert "hallucination_rate" in summary

class TestRelevanceAssessor:
    """Test suite for RelevanceAssessor"""
    
    @pytest.fixture
    def assessor(self):
        """Create assessor instance for testing"""
        return RelevanceAssessor()
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing"""
        return [
            RankedChunk(
                chunk_id="chunk1",
                document_id="doc1",
                content="Information about Paris and French geography",
                page_number=1,
                vector_score=0.9,
                bm25_score=0.8,
                combined_score=0.85,
                relevance_score=0.9,
                final_score=0.9,
                metadata={}
            )
        ]
    
    def test_initialization(self, assessor):
        """Test assessor initialization"""
        assert assessor.relevance_threshold == 0.6
    
    def test_assess_relevance_empty_inputs(self, assessor):
        """Test relevance assessment with empty inputs"""
        result = assessor.assess_relevance("", "")
        
        assert isinstance(result, RelevanceScore)
        assert result.overall_score == 0.0
        assert result.semantic_similarity == 0.0
    
    def test_assess_relevance_matching_query_response(self, assessor, sample_chunks):
        """Test relevance assessment with matching query and response"""
        query = "What is the capital of France?"
        response = "The capital of France is Paris."
        
        result = assessor.assess_relevance(query, response, sample_chunks)
        
        assert isinstance(result, RelevanceScore)
        assert result.overall_score > 0.0
        assert result.semantic_similarity >= 0.0
        assert result.query_coverage >= 0.0
        assert result.confidence > 0.0
    
    def test_query_coverage_calculation(self, assessor):
        """Test query coverage calculation"""
        query = "capital France Paris"
        response = "Paris is the capital of France and a beautiful city"
        
        coverage = assessor._calculate_query_coverage(query, response)
        assert coverage > 0.5  # Should cover most query terms
    
    def test_answer_focus_calculation(self, assessor):
        """Test answer focus calculation"""
        query = "Paris capital"
        response = "Paris is the capital"  # Focused response
        
        focus = assessor._calculate_answer_focus(query, response)
        assert focus > 0.0
    
    def test_topic_extraction(self, assessor):
        """Test topic extraction from text"""
        text = "Paris France capital European city"
        topics = assessor._extract_topics(text)
        
        assert len(topics) > 0
        assert any(topic in ["paris", "france", "capital", "european"] for topic in topics)
    
    def test_query_complexity_analysis(self, assessor):
        """Test query complexity analysis"""
        simple_query = "Paris capital"
        complex_query = "What is the capital of France and why was it chosen as the capital?"
        
        simple_complexity = assessor._analyze_query_complexity(simple_query)
        complex_complexity = assessor._analyze_query_complexity(complex_query)
        
        assert complex_complexity > simple_complexity
    
    def test_batch_assessment(self, assessor, sample_chunks):
        """Test batch relevance assessment"""
        queries_and_responses = [
            ("What is the capital of France?", "Paris is the capital of France."),
            ("Tell me about London.", "London is the capital of England.")
        ]
        
        results = assessor.batch_assess_relevance(queries_and_responses)
        
        assert len(results) == 2
        assert all(isinstance(result, RelevanceScore) for result in results)
    
    def test_relevance_summary(self, assessor):
        """Test relevance summary generation"""
        scores = [
            RelevanceScore(0.8, 0.7, 0.9, 0.8, 0.75, 0.85, 0.9),
            RelevanceScore(0.6, 0.5, 0.7, 0.6, 0.65, 0.7, 0.8)
        ]
        
        summary = assessor.get_relevance_summary(scores)
        
        assert summary["total_evaluations"] == 2
        assert "avg_relevance_score" in summary
        assert "avg_semantic_similarity" in summary
        assert "high_relevance_rate" in summary

class TestRetrievalMetricsCalculator:
    """Test suite for RetrievalMetricsCalculator"""
    
    @pytest.fixture
    def calculator(self):
        """Create calculator instance for testing"""
        return RetrievalMetricsCalculator()
    
    @pytest.fixture
    def sample_retrieved_chunks(self):
        """Create sample retrieved chunks"""
        return [
            RankedChunk("chunk1", "doc1", "content1", 1, 0.9, 0.8, 0.85, 0.9, 0.9, {}),
            RankedChunk("chunk2", "doc1", "content2", 2, 0.8, 0.7, 0.75, 0.8, 0.8, {}),
            RankedChunk("chunk3", "doc2", "content3", 1, 0.7, 0.6, 0.65, 0.7, 0.7, {}),
        ]
    
    def test_initialization(self, calculator):
        """Test calculator initialization"""
        assert calculator is not None
    
    def test_calculate_metrics_empty_retrieval(self, calculator):
        """Test metrics calculation with empty retrieval"""
        result = calculator.calculate_retrieval_metrics([], ["chunk1", "chunk2"])
        
        assert isinstance(result, RetrievalMetrics)
        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.f1_score == 0.0
        assert result.total_retrieved == 0
    
    def test_calculate_precision(self, calculator):
        """Test precision calculation"""
        retrieved_ids = ["chunk1", "chunk2", "chunk3"]
        relevant_ids = ["chunk1", "chunk3"]
        
        precision = calculator._calculate_precision(retrieved_ids, relevant_ids)
        assert precision == 2/3  # 2 relevant out of 3 retrieved
    
    def test_calculate_recall(self, calculator):
        """Test recall calculation"""
        retrieved_ids = ["chunk1", "chunk2"]
        relevant_ids = ["chunk1", "chunk3", "chunk4"]
        
        recall = calculator._calculate_recall(retrieved_ids, relevant_ids)
        assert recall == 1/3  # 1 relevant retrieved out of 3 total relevant
    
    def test_calculate_f1_score(self, calculator):
        """Test F1 score calculation"""
        precision = 0.6
        recall = 0.4
        
        f1 = calculator._calculate_f1_score(precision, recall)
        expected_f1 = 2 * (0.6 * 0.4) / (0.6 + 0.4)
        assert abs(f1 - expected_f1) < 1e-6
    
    def test_calculate_mrr(self, calculator):
        """Test Mean Reciprocal Rank calculation"""
        retrieved_ids = ["chunk1", "chunk2", "chunk3"]
        relevant_ids = ["chunk2"]
        
        mrr = calculator._calculate_mrr(retrieved_ids, relevant_ids)
        assert mrr == 1/2  # First relevant at position 2
    
    def test_calculate_mrr_no_relevant(self, calculator):
        """Test MRR when no relevant documents found"""
        retrieved_ids = ["chunk1", "chunk2"]
        relevant_ids = ["chunk3", "chunk4"]
        
        mrr = calculator._calculate_mrr(retrieved_ids, relevant_ids)
        assert mrr == 0.0
    
    def test_diversity_score_calculation(self, calculator, sample_retrieved_chunks):
        """Test diversity score calculation"""
        diversity = calculator._calculate_diversity_score(sample_retrieved_chunks)
        
        assert 0.0 <= diversity <= 1.0
        # Should have some diversity due to different documents
        assert diversity > 0.0
    
    def test_context_quality_calculation(self, calculator, sample_retrieved_chunks):
        """Test context quality calculation"""
        quality = calculator._calculate_context_quality(sample_retrieved_chunks)
        
        assert 0.0 <= quality <= 1.0
        assert quality > 0.0  # Should have some quality
    
    def test_full_metrics_calculation(self, calculator, sample_retrieved_chunks):
        """Test full metrics calculation"""
        relevant_ids = ["chunk1", "chunk2"]
        
        metrics = calculator.calculate_retrieval_metrics(
            sample_retrieved_chunks, relevant_ids
        )
        
        assert isinstance(metrics, RetrievalMetrics)
        assert 0.0 <= metrics.precision <= 1.0
        assert 0.0 <= metrics.recall <= 1.0
        assert 0.0 <= metrics.f1_score <= 1.0
        assert metrics.total_retrieved == 3
        assert metrics.relevant_retrieved == 2
    
    def test_batch_calculation(self, calculator, sample_retrieved_chunks):
        """Test batch metrics calculation"""
        retrieved_chunks_list = [sample_retrieved_chunks, sample_retrieved_chunks[:2]]
        relevant_ids_list = [["chunk1", "chunk2"], ["chunk1"]]
        
        results = calculator.batch_calculate_retrieval_metrics(
            retrieved_chunks_list, relevant_ids_list
        )
        
        assert len(results) == 2
        assert all(isinstance(result, RetrievalMetrics) for result in results)
    
    def test_retrieval_summary(self, calculator):
        """Test retrieval summary generation"""
        metrics_list = [
            RetrievalMetrics(0.8, 0.6, 0.7, 0.5, 0.6, 0.7, 0.8, 0.75, 5, 4),
            RetrievalMetrics(0.6, 0.8, 0.7, 0.4, 0.5, 0.6, 0.7, 0.65, 3, 2)
        ]
        
        summary = calculator.get_retrieval_summary(metrics_list)
        
        assert summary["total_queries"] == 2
        assert "avg_precision" in summary
        assert "avg_recall" in summary
        assert "avg_f1_score" in summary

class TestComparativeEvaluator:
    """Test suite for ComparativeEvaluator"""
    
    @pytest.fixture
    def evaluator(self):
        """Create evaluator instance for testing"""
        return ComparativeEvaluator()
    
    def test_initialization(self, evaluator):
        """Test evaluator initialization"""
        assert evaluator.significance_level == 0.05
    
    def test_compare_strategies_empty(self, evaluator):
        """Test strategy comparison with empty scores"""
        result = evaluator.compare_strategies([], [])
        
        assert isinstance(result, ComparativeResult)
        assert result.strategy_a_score == 0.0
        assert result.strategy_b_score == 0.0
        assert result.winner == "tie"
    
    def test_compare_strategies_clear_winner(self, evaluator):
        """Test strategy comparison with clear winner"""
        strategy_a_scores = [0.5, 0.6, 0.55, 0.58, 0.52]
        strategy_b_scores = [0.8, 0.85, 0.82, 0.88, 0.83]
        
        result = evaluator.compare_strategies(strategy_a_scores, strategy_b_scores)
        
        assert isinstance(result, ComparativeResult)
        assert result.strategy_b_score > result.strategy_a_score
        assert result.improvement > 0
        assert result.winner in ["B", "tie"]  # B should win or tie due to statistical test
    
    def test_compare_strategies_similar_performance(self, evaluator):
        """Test strategy comparison with similar performance"""
        strategy_a_scores = [0.7, 0.72, 0.68, 0.71, 0.69]
        strategy_b_scores = [0.71, 0.73, 0.69, 0.72, 0.70]
        
        result = evaluator.compare_strategies(strategy_a_scores, strategy_b_scores)
        
        assert isinstance(result, ComparativeResult)
        # Should likely be a tie due to similar performance
        assert result.statistical_significance > 0.01  # Not very significant
    
    def test_effect_size_calculation(self, evaluator):
        """Test effect size calculation"""
        scores_a = [0.5, 0.6, 0.55]
        scores_b = [0.8, 0.9, 0.85]
        
        effect_size = evaluator._calculate_effect_size(scores_a, scores_b)
        
        assert effect_size > 0.0  # Should have large effect size
    
    def test_confidence_interval_calculation(self, evaluator):
        """Test confidence interval calculation"""
        scores_a = [0.5, 0.6, 0.55, 0.58, 0.52]
        scores_b = [0.8, 0.85, 0.82, 0.88, 0.83]
        
        ci = evaluator._calculate_confidence_interval(scores_a, scores_b)
        
        assert isinstance(ci, tuple)
        assert len(ci) == 2
        assert ci[0] < ci[1]  # Lower bound < upper bound
    
    def test_multiple_strategy_comparison(self, evaluator):
        """Test multiple strategy comparison"""
        strategy_scores = {
            "Strategy A": [0.5, 0.6, 0.55],
            "Strategy B": [0.8, 0.85, 0.82],
            "Strategy C": [0.7, 0.75, 0.72]
        }
        
        results = evaluator.compare_multiple_strategies(strategy_scores)
        
        assert len(results) == 3
        assert "Strategy A" in results
        assert "Strategy B" in results["Strategy A"]
        assert isinstance(results["Strategy A"]["Strategy B"], ComparativeResult)
    
    def test_strategy_ranking(self, evaluator):
        """Test strategy ranking"""
        strategy_scores = {
            "Strategy A": [0.5, 0.6, 0.55],
            "Strategy B": [0.8, 0.85, 0.82],
            "Strategy C": [0.7, 0.75, 0.72]
        }
        
        rankings = evaluator.rank_strategies(strategy_scores)
        
        assert len(rankings) == 3
        assert rankings[0][0] == "Strategy B"  # Best strategy
        assert rankings[0][1] > rankings[1][1]  # Scores in descending order
    
    def test_comparison_report_generation(self, evaluator):
        """Test comparison report generation"""
        strategy_scores = {
            "Strategy A": [0.5, 0.6, 0.55],
            "Strategy B": [0.8, 0.85, 0.82]
        }
        
        comparison_results = evaluator.compare_multiple_strategies(strategy_scores)
        report = evaluator.generate_comparison_report(comparison_results, strategy_scores)
        
        assert "total_strategies" in report
        assert "rankings" in report
        assert "best_strategy" in report
        assert "strategy_statistics" in report
        assert report["total_strategies"] == 2

class TestEvaluationFramework:
    """Test suite for comprehensive EvaluationFramework"""
    
    @pytest.fixture
    def framework(self):
        """Create framework instance for testing"""
        return EvaluationFramework()
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing"""
        return [
            RankedChunk(
                chunk_id="chunk1",
                document_id="doc1",
                content="Paris is the capital of France.",
                page_number=1,
                vector_score=0.9,
                bm25_score=0.8,
                combined_score=0.85,
                relevance_score=0.9,
                final_score=0.9,
                metadata={"filename": "geography.pdf"}
            )
        ]
    
    def test_initialization(self, framework):
        """Test framework initialization"""
        assert framework.faithfulness_evaluator is not None
        assert framework.relevance_assessor is not None
        assert framework.retrieval_calculator is not None
        assert framework.comparative_evaluator is not None
    
    def test_evaluate_response(self, framework, sample_chunks):
        """Test comprehensive response evaluation"""
        query = "What is the capital of France?"
        response = "Paris is the capital of France."
        
        report = framework.evaluate_response(query, response, sample_chunks)
        
        assert isinstance(report, EvaluationReport)
        assert report.query == query
        assert report.response == response
        assert isinstance(report.faithfulness, FaithfulnessScore)
        assert isinstance(report.relevance, RelevanceScore)
        assert 0.0 <= report.overall_quality <= 1.0
        assert len(report.recommendations) > 0
    
    def test_evaluate_response_with_retrieval_metrics(self, framework, sample_chunks):
        """Test evaluation with retrieval metrics"""
        query = "What is the capital of France?"
        response = "Paris is the capital of France."
        relevant_chunk_ids = ["chunk1"]
        
        report = framework.evaluate_response(
            query, response, sample_chunks, relevant_chunk_ids
        )
        
        assert report.retrieval_metrics is not None
        assert isinstance(report.retrieval_metrics, RetrievalMetrics)
    
    def test_batch_evaluation(self, framework, sample_chunks):
        """Test batch evaluation"""
        evaluation_data = [
            ("What is the capital of France?", "Paris is the capital of France.", sample_chunks),
            ("Tell me about London.", "London is the capital of England.", sample_chunks)
        ]
        
        reports = framework.batch_evaluate(evaluation_data)
        
        assert len(reports) == 2
        assert all(isinstance(report, EvaluationReport) for report in reports)
    
    def test_evaluation_summary(self, framework):
        """Test evaluation summary generation"""
        # Create mock reports
        reports = [
            EvaluationReport(
                timestamp=datetime.now(),
                query="test query",
                response="test response",
                source_chunks=[],
                faithfulness=FaithfulnessScore(0.8, [0.8], 1, 1, [], 0.8, [], 0.9),
                relevance=RelevanceScore(0.7, 0.7, 0.8, 0.6, 0.7, 0.8, 0.9),
                retrieval_metrics=None,
                overall_quality=0.75,
                recommendations=["Improve faithfulness"]
            )
        ]
        
        summary = framework.generate_evaluation_summary(reports)
        
        assert summary["total_evaluations"] == 1
        assert "overall_quality" in summary
        assert "faithfulness_summary" in summary
        assert "relevance_summary" in summary

class TestEvaluationProperties:
    """Property-based tests for evaluation framework"""
    
    @given(
        response_length=st.integers(min_value=10, max_value=500),
        num_chunks=st.integers(min_value=1, max_value=10),
        relevance_scores=st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=10)
    )
    @settings(max_examples=50)
    def test_property_18_faithfulness_measurement(self, response_length, num_chunks, relevance_scores):
        """
        **Feature: enterprise-genai-platform, Property 18: Faithfulness Measurement**
        
        For any generated response, the evaluation framework should produce quantitative 
        faithfulness scores measuring alignment with source documents.
        
        **Validates: Requirements 6.1**
        """
        # Create evaluator
        evaluator = FaithfulnessEvaluator()
        
        # Generate synthetic response and chunks
        response = " ".join([f"Statement {i} about the topic." for i in range(response_length // 20)])
        
        chunks = []
        for i in range(num_chunks):
            score = relevance_scores[i % len(relevance_scores)]
            chunk = RankedChunk(
                chunk_id=f"chunk{i}",
                document_id=f"doc{i}",
                content=f"Source content {i} with relevant information.",
                page_number=1,
                vector_score=score,
                bm25_score=score * 0.9,
                combined_score=score * 0.95,
                relevance_score=score,
                final_score=score,
                metadata={}
            )
            chunks.append(chunk)
        
        # Evaluate faithfulness
        result = evaluator.evaluate_faithfulness(response, chunks)
        
        # Property: Must produce quantitative faithfulness scores
        assert isinstance(result, FaithfulnessScore)
        assert 0.0 <= result.overall_score <= 1.0
        assert isinstance(result.claim_scores, list)
        assert all(0.0 <= score <= 1.0 for score in result.claim_scores)
        assert result.supported_claims >= 0
        assert result.total_claims >= 0
        assert result.supported_claims <= result.total_claims
        assert 0.0 <= result.source_alignment <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        
        # Property: Faithfulness should correlate with source quality
        if chunks:
            avg_chunk_score = np.mean([chunk.final_score for chunk in chunks])
            # Higher quality sources should generally lead to higher faithfulness confidence
            if avg_chunk_score > 0.8:
                assert result.confidence > 0.3  # Should have reasonable confidence with good sources
    
    @given(
        query_length=st.integers(min_value=3, max_value=20),
        response_length=st.integers(min_value=5, max_value=50),
        semantic_overlap=st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(max_examples=50)
    def test_property_19_relevance_assessment(self, query_length, response_length, semantic_overlap):
        """
        **Feature: enterprise-genai-platform, Property 19: Relevance Assessment**
        
        For any query-answer pair, the system should calculate quantitative relevance 
        scores measuring how well the answer addresses the query.
        
        **Validates: Requirements 6.2**
        """
        # Create assessor
        assessor = RelevanceAssessor()
        
        # Generate synthetic query and response with controlled overlap
        base_words = ["capital", "city", "country", "population", "location", "history", "culture"]
        
        # Create query
        query_words = np.random.choice(base_words, size=min(query_length, len(base_words)), replace=False)
        query = " ".join(query_words)
        
        # Create response with controlled semantic overlap
        overlap_count = int(len(query_words) * semantic_overlap)
        response_overlap_words = query_words[:overlap_count]
        response_other_words = np.random.choice(
            [w for w in base_words if w not in response_overlap_words], 
            size=min(response_length - overlap_count, len(base_words) - overlap_count),
            replace=False
        )
        response_words = list(response_overlap_words) + list(response_other_words)
        response = " ".join(response_words)
        
        # Assess relevance
        result = assessor.assess_relevance(query, response)
        
        # Property: Must produce quantitative relevance scores
        assert isinstance(result, RelevanceScore)
        assert 0.0 <= result.overall_score <= 1.0
        assert 0.0 <= result.semantic_similarity <= 1.0
        assert 0.0 <= result.query_coverage <= 1.0
        assert 0.0 <= result.answer_focus <= 1.0
        assert 0.0 <= result.topic_alignment <= 1.0
        assert 0.0 <= result.completeness <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        
        # Property: Higher semantic overlap should lead to higher relevance
        if semantic_overlap > 0.7:
            assert result.query_coverage > 0.5  # Should have good coverage with high overlap
        
        # Property: Query coverage should reflect actual word overlap
        if semantic_overlap > 0.0:
            assert result.query_coverage > 0.0  # Should detect some coverage
    
    @given(
        num_retrieved=st.integers(min_value=1, max_value=20),
        num_relevant=st.integers(min_value=1, max_value=10),
        relevance_overlap=st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(max_examples=50)
    def test_property_20_retrieval_metrics_calculation(self, num_retrieved, num_relevant, relevance_overlap):
        """
        **Feature: enterprise-genai-platform, Property 20: Retrieval Metrics Calculation**
        
        For any retrieval operation, the system should calculate and report precision 
        and recall metrics for the retrieved context.
        
        **Validates: Requirements 6.3**
        """
        assume(num_relevant <= num_retrieved * 2)  # Reasonable assumption
        
        # Create calculator
        calculator = RetrievalMetricsCalculator()
        
        # Generate retrieved chunks
        retrieved_chunks = []
        for i in range(num_retrieved):
            chunk = RankedChunk(
                chunk_id=f"chunk{i}",
                document_id=f"doc{i % 3}",  # Some document diversity
                content=f"Content for chunk {i}",
                page_number=1,
                vector_score=np.random.uniform(0.5, 1.0),
                bm25_score=np.random.uniform(0.5, 1.0),
                combined_score=np.random.uniform(0.5, 1.0),
                relevance_score=np.random.uniform(0.5, 1.0),
                final_score=np.random.uniform(0.5, 1.0),
                metadata={}
            )
            retrieved_chunks.append(chunk)
        
        # Generate relevant chunk IDs with controlled overlap
        all_chunk_ids = [f"chunk{i}" for i in range(num_retrieved)]
        overlap_count = int(num_retrieved * relevance_overlap)
        relevant_from_retrieved = all_chunk_ids[:overlap_count]
        relevant_not_retrieved = [f"missing_chunk{i}" for i in range(num_relevant - overlap_count)]
        relevant_chunk_ids = relevant_from_retrieved + relevant_not_retrieved
        
        # Calculate metrics
        metrics = calculator.calculate_retrieval_metrics(retrieved_chunks, relevant_chunk_ids)
        
        # Property: Must calculate valid precision and recall metrics
        assert isinstance(metrics, RetrievalMetrics)
        assert 0.0 <= metrics.precision <= 1.0
        assert 0.0 <= metrics.recall <= 1.0
        assert 0.0 <= metrics.f1_score <= 1.0
        assert 0.0 <= metrics.mrr <= 1.0
        assert 0.0 <= metrics.ndcg <= 1.0
        assert metrics.total_retrieved == num_retrieved
        assert metrics.relevant_retrieved >= 0
        assert metrics.relevant_retrieved <= min(num_retrieved, num_relevant)
        
        # Property: Precision calculation should be mathematically correct
        expected_precision = metrics.relevant_retrieved / num_retrieved
        assert abs(metrics.precision - expected_precision) < 1e-10
        
        # Property: Recall calculation should be mathematically correct
        expected_recall = metrics.relevant_retrieved / num_relevant
        assert abs(metrics.recall - expected_recall) < 1e-10
        
        # Property: F1 score should be harmonic mean of precision and recall
        if metrics.precision + metrics.recall > 0:
            expected_f1 = 2 * (metrics.precision * metrics.recall) / (metrics.precision + metrics.recall)
            assert abs(metrics.f1_score - expected_f1) < 1e-10
        else:
            assert metrics.f1_score == 0.0
    
    @given(
        strategy_a_scores=st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=3, max_size=20),
        strategy_b_scores=st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=3, max_size=20)
    )
    @settings(max_examples=50)
    def test_property_21_comparative_evaluation(self, strategy_a_scores, strategy_b_scores):
        """
        **Feature: enterprise-genai-platform, Property 21: Comparative Evaluation**
        
        For any set of different prompt strategies tested on the same queries, 
        the system should provide quantitative performance comparisons.
        
        **Validates: Requirements 6.4, 6.5**
        """
        # Create evaluator
        evaluator = ComparativeEvaluator()
        
        # Compare strategies
        result = evaluator.compare_strategies(strategy_a_scores, strategy_b_scores)
        
        # Property: Must provide quantitative comparison results
        assert isinstance(result, ComparativeResult)
        assert isinstance(result.strategy_a_score, float)
        assert isinstance(result.strategy_b_score, float)
        assert isinstance(result.improvement, float)
        assert 0.0 <= result.statistical_significance <= 1.0
        assert isinstance(result.confidence_interval, tuple)
        assert len(result.confidence_interval) == 2
        assert result.winner in ["A", "B", "tie"]
        assert result.effect_size >= 0.0
        assert result.sample_size == len(strategy_a_scores) + len(strategy_b_scores)
        
        # Property: Strategy scores should match input means
        expected_a_score = np.mean(strategy_a_scores)
        expected_b_score = np.mean(strategy_b_scores)
        assert abs(result.strategy_a_score - expected_a_score) < 1e-10
        assert abs(result.strategy_b_score - expected_b_score) < 1e-10
        
        # Property: Improvement calculation should be mathematically correct
        if expected_a_score > 0:
            expected_improvement = ((expected_b_score - expected_a_score) / expected_a_score) * 100
            assert abs(result.improvement - expected_improvement) < 1e-8
        
        # Property: Confidence interval should be ordered correctly
        assert result.confidence_interval[0] <= result.confidence_interval[1]
        
        # Property: Winner determination should be consistent with scores and significance
        if result.statistical_significance <= 0.05:  # Significant difference
            if result.strategy_b_score > result.strategy_a_score:
                assert result.winner == "B"
            elif result.strategy_a_score > result.strategy_b_score:
                assert result.winner == "A"
        # If not significant, winner could be "tie" regardless of score difference

if __name__ == "__main__":
    print("Running evaluation framework tests...")
    pytest.main([__file__, "-v"])