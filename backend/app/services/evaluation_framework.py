"""
Evaluation Framework for Enterprise GenAI Platform

This module provides comprehensive evaluation capabilities for RAG systems including:
- Faithfulness evaluation (alignment with source documents)
- Relevance assessment (query-answer relevance)
- Retrieval metrics (precision, recall, context quality)
- Comparative evaluation (A/B testing for prompt strategies)

The framework produces quantitative metrics for measuring and improving system performance.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import logging
import re
from collections import defaultdict
import time
from datetime import datetime, timedelta
import json
import statistics
from abc import ABC, abstractmethod

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("Warning: sentence_transformers or sklearn not available")
    SentenceTransformer = None
    cosine_similarity = None

from app.services.rag_pipeline import RankedChunk, Citation, ContextWindow

logger = logging.getLogger(__name__)

@dataclass
class FaithfulnessScore:
    """Faithfulness evaluation result"""
    overall_score: float  # 0.0 to 1.0
    claim_scores: List[float]  # Individual claim faithfulness scores
    supported_claims: int
    total_claims: int
    unsupported_claims: List[str]  # Claims not supported by source
    source_alignment: float  # How well response aligns with sources
    hallucination_indicators: List[str]  # Potential hallucination markers
    confidence: float  # Confidence in the faithfulness assessment

@dataclass
class RelevanceScore:
    """Relevance assessment result"""
    overall_score: float  # 0.0 to 1.0
    semantic_similarity: float  # Semantic similarity between query and answer
    query_coverage: float  # How well the answer covers the query
    answer_focus: float  # How focused the answer is on the query
    topic_alignment: float  # Topic alignment score
    completeness: float  # How complete the answer is
    confidence: float  # Confidence in the relevance assessment

@dataclass
class RetrievalMetrics:
    """Retrieval performance metrics"""
    precision: float  # Precision of retrieved chunks
    recall: float  # Recall of retrieved chunks
    f1_score: float  # F1 score
    mrr: float  # Mean Reciprocal Rank
    ndcg: float  # Normalized Discounted Cumulative Gain
    context_quality: float  # Overall context quality score
    diversity_score: float  # Diversity of retrieved sources
    coverage_score: float  # Coverage of relevant information
    total_retrieved: int
    relevant_retrieved: int

@dataclass
class ComparativeResult:
    """Comparative evaluation result"""
    strategy_a_score: float
    strategy_b_score: float
    improvement: float  # Percentage improvement
    statistical_significance: float  # p-value
    confidence_interval: Tuple[float, float]
    winner: str  # "A", "B", or "tie"
    effect_size: float  # Cohen's d or similar
    sample_size: int

@dataclass
class EvaluationReport:
    """Comprehensive evaluation report"""
    timestamp: datetime
    query: str
    response: str
    source_chunks: List[RankedChunk]
    faithfulness: FaithfulnessScore
    relevance: RelevanceScore
    retrieval_metrics: Optional[RetrievalMetrics]
    overall_quality: float  # Combined quality score
    recommendations: List[str]  # Improvement recommendations
    metadata: Dict[str, Any] = field(default_factory=dict)

class FaithfulnessEvaluator:
    """Evaluates faithfulness of responses to source documents"""
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 similarity_threshold: float = 0.7,
                 claim_extraction_method: str = "sentence_split"):
        """Initialize faithfulness evaluator
        
        Args:
            model_name: Sentence transformer model for semantic similarity
            similarity_threshold: Threshold for considering claims supported
            claim_extraction_method: Method for extracting claims from response
        """
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.claim_extraction_method = claim_extraction_method
        
        # Initialize sentence transformer model
        self.model = None
        if SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer(model_name)
                logger.info(f"Initialized faithfulness evaluator with model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load sentence transformer model: {e}")
                self.model = None
        else:
            logger.warning("SentenceTransformer not available, using fallback methods")
    
    def evaluate_faithfulness(self, 
                            response: str,
                            source_chunks: List[RankedChunk],
                            query: Optional[str] = None) -> FaithfulnessScore:
        """Evaluate faithfulness of response to source documents
        
        Args:
            response: Generated response to evaluate
            source_chunks: Source document chunks used for generation
            query: Original query (optional, for context)
            
        Returns:
            FaithfulnessScore with detailed faithfulness metrics
        """
        if not response.strip():
            return FaithfulnessScore(
                overall_score=0.0,
                claim_scores=[],
                supported_claims=0,
                total_claims=0,
                unsupported_claims=[],
                source_alignment=0.0,
                hallucination_indicators=["Empty response"],
                confidence=1.0
            )
        
        if not source_chunks:
            return FaithfulnessScore(
                overall_score=0.0,
                claim_scores=[],
                supported_claims=0,
                total_claims=0,
                unsupported_claims=[response],
                source_alignment=0.0,
                hallucination_indicators=["No source documents provided"],
                confidence=1.0
            )
        
        logger.info(f"Evaluating faithfulness for response with {len(source_chunks)} source chunks")
        
        # Extract claims from response
        claims = self._extract_claims(response)
        
        if not claims:
            return FaithfulnessScore(
                overall_score=0.5,  # Neutral score for responses without extractable claims
                claim_scores=[],
                supported_claims=0,
                total_claims=0,
                unsupported_claims=[],
                source_alignment=0.5,
                hallucination_indicators=[],
                confidence=0.5
            )
        
        # Evaluate each claim against source documents
        claim_scores = []
        unsupported_claims = []
        
        source_texts = [chunk.content for chunk in source_chunks]
        combined_source = " ".join(source_texts)
        
        for claim in claims:
            score = self._evaluate_claim_support(claim, source_texts)
            claim_scores.append(score)
            
            if score < self.similarity_threshold:
                unsupported_claims.append(claim)
        
        # Calculate overall metrics
        supported_claims = sum(1 for score in claim_scores if score >= self.similarity_threshold)
        overall_score = np.mean(claim_scores) if claim_scores else 0.0
        
        # Calculate source alignment
        source_alignment = self._calculate_source_alignment(response, combined_source)
        
        # Detect hallucination indicators
        hallucination_indicators = self._detect_hallucination_indicators(
            response, source_chunks, unsupported_claims
        )
        
        # Calculate confidence in assessment
        confidence = self._calculate_faithfulness_confidence(
            claim_scores, source_chunks, response
        )
        
        result = FaithfulnessScore(
            overall_score=overall_score,
            claim_scores=claim_scores,
            supported_claims=supported_claims,
            total_claims=len(claims),
            unsupported_claims=unsupported_claims,
            source_alignment=source_alignment,
            hallucination_indicators=hallucination_indicators,
            confidence=confidence
        )
        
        logger.info(f"Faithfulness evaluation completed: score={overall_score:.3f}, "
                   f"supported={supported_claims}/{len(claims)}, confidence={confidence:.3f}")
        
        return result
    
    def _extract_claims(self, response: str) -> List[str]:
        """Extract individual claims from response text
        
        Args:
            response: Response text to extract claims from
            
        Returns:
            List of individual claims
        """
        if self.claim_extraction_method == "sentence_split":
            return self._extract_claims_by_sentences(response)
        elif self.claim_extraction_method == "semantic_units":
            return self._extract_claims_by_semantic_units(response)
        else:
            logger.warning(f"Unknown claim extraction method: {self.claim_extraction_method}")
            return self._extract_claims_by_sentences(response)
    
    def _extract_claims_by_sentences(self, response: str) -> List[str]:
        """Extract claims by splitting into sentences"""
        # Simple sentence splitting using punctuation
        sentences = re.split(r'[.!?]+', response)
        
        # Clean and filter sentences
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Filter out very short fragments
                claims.append(sentence)
        
        return claims
    
    def _extract_claims_by_semantic_units(self, response: str) -> List[str]:
        """Extract claims by identifying semantic units (more advanced)"""
        # For now, use sentence splitting as fallback
        # In a production system, this could use NLP libraries for better claim extraction
        return self._extract_claims_by_sentences(response)
    
    def _evaluate_claim_support(self, claim: str, source_texts: List[str]) -> float:
        """Evaluate how well a claim is supported by source texts
        
        Args:
            claim: Individual claim to evaluate
            source_texts: List of source document texts
            
        Returns:
            Support score (0.0 to 1.0)
        """
        if self.model is not None and cosine_similarity is not None:
            return self._evaluate_claim_support_semantic(claim, source_texts)
        else:
            return self._evaluate_claim_support_lexical(claim, source_texts)
    
    def _evaluate_claim_support_semantic(self, claim: str, source_texts: List[str]) -> float:
        """Evaluate claim support using semantic similarity"""
        try:
            # Encode claim and source texts
            claim_embedding = self.model.encode([claim])
            source_embeddings = self.model.encode(source_texts)
            
            # Calculate similarities
            similarities = cosine_similarity(claim_embedding, source_embeddings)[0]
            
            # Return maximum similarity (best supporting source)
            return float(np.max(similarities))
            
        except Exception as e:
            logger.error(f"Semantic claim evaluation failed: {e}")
            return self._evaluate_claim_support_lexical(claim, source_texts)
    
    def _evaluate_claim_support_lexical(self, claim: str, source_texts: List[str]) -> float:
        """Evaluate claim support using lexical overlap (fallback method)"""
        claim_words = set(claim.lower().split())
        
        if not claim_words:
            return 0.0
        
        max_overlap = 0.0
        
        for source_text in source_texts:
            source_words = set(source_text.lower().split())
            
            if not source_words:
                continue
            
            # Calculate Jaccard similarity
            intersection = len(claim_words & source_words)
            union = len(claim_words | source_words)
            
            if union > 0:
                overlap = intersection / union
                max_overlap = max(max_overlap, overlap)
        
        return max_overlap
    
    def _calculate_source_alignment(self, response: str, combined_source: str) -> float:
        """Calculate overall alignment between response and source documents"""
        if self.model is not None and cosine_similarity is not None:
            try:
                response_embedding = self.model.encode([response])
                source_embedding = self.model.encode([combined_source])
                
                similarity = cosine_similarity(response_embedding, source_embedding)[0][0]
                return float(similarity)
                
            except Exception as e:
                logger.error(f"Semantic alignment calculation failed: {e}")
        
        # Fallback to lexical similarity
        response_words = set(response.lower().split())
        source_words = set(combined_source.lower().split())
        
        if not response_words or not source_words:
            return 0.0
        
        intersection = len(response_words & source_words)
        union = len(response_words | source_words)
        
        return intersection / union if union > 0 else 0.0
    
    def _detect_hallucination_indicators(self, 
                                       response: str,
                                       source_chunks: List[RankedChunk],
                                       unsupported_claims: List[str]) -> List[str]:
        """Detect potential hallucination indicators"""
        indicators = []
        
        # High proportion of unsupported claims
        if len(unsupported_claims) > 0:
            total_claims = len(self._extract_claims(response))
            if total_claims > 0:
                unsupported_ratio = len(unsupported_claims) / total_claims
                if unsupported_ratio > 0.5:
                    indicators.append(f"High proportion of unsupported claims ({unsupported_ratio:.1%})")
        
        # Specific numbers or dates not in sources
        numbers_in_response = re.findall(r'\b\d+(?:\.\d+)?\b', response)
        if numbers_in_response:
            source_text = " ".join([chunk.content for chunk in source_chunks])
            unsupported_numbers = []
            
            for number in numbers_in_response:
                if number not in source_text:
                    unsupported_numbers.append(number)
            
            if len(unsupported_numbers) > 2:
                indicators.append(f"Multiple unsupported numbers: {unsupported_numbers[:3]}")
        
        # Definitive statements without source support
        definitive_patterns = [
            r'\b(definitely|certainly|absolutely|always|never)\b',
            r'\b(all|every|none|no)\s+\w+\s+(are|is|will|must)\b',
            r'\b(it is (clear|obvious|evident) that)\b'
        ]
        
        for pattern in definitive_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                indicators.append(f"Definitive statements may lack source support: {matches[:2]}")
                break
        
        # Citations or references not in source
        citation_patterns = [
            r'\b(according to|as stated in|research shows|studies indicate)\b',
            r'\b(experts say|scientists believe|researchers found)\b'
        ]
        
        for pattern in citation_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                indicators.append("Response contains citation-like language without clear source attribution")
                break
        
        return indicators
    
    def _calculate_faithfulness_confidence(self, 
                                         claim_scores: List[float],
                                         source_chunks: List[RankedChunk],
                                         response: str) -> float:
        """Calculate confidence in faithfulness assessment"""
        confidence_factors = []
        
        # Score consistency (lower variance = higher confidence)
        if len(claim_scores) > 1:
            score_variance = np.var(claim_scores)
            consistency_confidence = max(0.0, 1.0 - score_variance)
            confidence_factors.append(consistency_confidence)
        
        # Source quality (higher relevance scores = higher confidence)
        if source_chunks:
            avg_relevance = np.mean([chunk.final_score for chunk in source_chunks])
            source_confidence = min(1.0, avg_relevance)
            confidence_factors.append(source_confidence)
        
        # Response length (moderate length = higher confidence)
        response_length = len(response.split())
        if 20 <= response_length <= 200:
            length_confidence = 1.0
        elif response_length < 20:
            length_confidence = response_length / 20.0
        else:
            length_confidence = max(0.5, 200.0 / response_length)
        confidence_factors.append(length_confidence)
        
        # Model availability (semantic models = higher confidence)
        model_confidence = 0.9 if self.model is not None else 0.6
        confidence_factors.append(model_confidence)
        
        # Calculate weighted average
        if confidence_factors:
            return np.mean(confidence_factors)
        else:
            return 0.5
    
    def batch_evaluate_faithfulness(self, 
                                  responses_and_sources: List[Tuple[str, List[RankedChunk]]],
                                  queries: Optional[List[str]] = None) -> List[FaithfulnessScore]:
        """Evaluate faithfulness for multiple response-source pairs
        
        Args:
            responses_and_sources: List of (response, source_chunks) tuples
            queries: Optional list of queries for context
            
        Returns:
            List of FaithfulnessScore objects
        """
        results = []
        
        for i, (response, source_chunks) in enumerate(responses_and_sources):
            query = queries[i] if queries and i < len(queries) else None
            
            score = self.evaluate_faithfulness(response, source_chunks, query)
            results.append(score)
        
        logger.info(f"Batch faithfulness evaluation completed for {len(results)} items")
        return results
    
    def get_faithfulness_summary(self, scores: List[FaithfulnessScore]) -> Dict[str, Any]:
        """Generate summary statistics for faithfulness scores
        
        Args:
            scores: List of FaithfulnessScore objects
            
        Returns:
            Dictionary with summary statistics
        """
        if not scores:
            return {"error": "No scores provided"}
        
        overall_scores = [score.overall_score for score in scores]
        supported_ratios = [
            score.supported_claims / max(1, score.total_claims) 
            for score in scores
        ]
        
        return {
            "total_evaluations": len(scores),
            "avg_faithfulness_score": np.mean(overall_scores),
            "median_faithfulness_score": np.median(overall_scores),
            "std_faithfulness_score": np.std(overall_scores),
            "min_faithfulness_score": np.min(overall_scores),
            "max_faithfulness_score": np.max(overall_scores),
            "avg_supported_ratio": np.mean(supported_ratios),
            "total_claims": sum(score.total_claims for score in scores),
            "total_supported_claims": sum(score.supported_claims for score in scores),
            "avg_confidence": np.mean([score.confidence for score in scores]),
            "hallucination_rate": sum(1 for score in scores if score.hallucination_indicators) / len(scores)
        }

class RelevanceAssessor:
    """Assesses relevance of answers to user queries"""
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 relevance_threshold: float = 0.6):
        """Initialize relevance assessor
        
        Args:
            model_name: Sentence transformer model for semantic similarity
            relevance_threshold: Threshold for considering answers relevant
        """
        self.model_name = model_name
        self.relevance_threshold = relevance_threshold
        
        # Initialize sentence transformer model
        self.model = None
        if SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer(model_name)
                logger.info(f"Initialized relevance assessor with model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load sentence transformer model: {e}")
                self.model = None
        else:
            logger.warning("SentenceTransformer not available, using fallback methods")
    
    def assess_relevance(self, 
                        query: str,
                        response: str,
                        context_chunks: Optional[List[RankedChunk]] = None) -> RelevanceScore:
        """Assess relevance of response to query
        
        Args:
            query: Original user query
            response: Generated response to evaluate
            context_chunks: Optional context chunks used for generation
            
        Returns:
            RelevanceScore with detailed relevance metrics
        """
        if not query.strip() or not response.strip():
            return RelevanceScore(
                overall_score=0.0,
                semantic_similarity=0.0,
                query_coverage=0.0,
                answer_focus=0.0,
                topic_alignment=0.0,
                completeness=0.0,
                confidence=1.0
            )
        
        logger.info(f"Assessing relevance for query: '{query[:50]}...'")
        
        # Calculate semantic similarity
        semantic_similarity = self._calculate_semantic_similarity(query, response)
        
        # Calculate query coverage
        query_coverage = self._calculate_query_coverage(query, response)
        
        # Calculate answer focus
        answer_focus = self._calculate_answer_focus(query, response)
        
        # Calculate topic alignment
        topic_alignment = self._calculate_topic_alignment(query, response)
        
        # Calculate completeness
        completeness = self._calculate_completeness(query, response, context_chunks)
        
        # Calculate overall relevance score
        overall_score = self._calculate_overall_relevance(
            semantic_similarity, query_coverage, answer_focus, topic_alignment, completeness
        )
        
        # Calculate confidence in assessment
        confidence = self._calculate_relevance_confidence(
            query, response, semantic_similarity, context_chunks
        )
        
        result = RelevanceScore(
            overall_score=overall_score,
            semantic_similarity=semantic_similarity,
            query_coverage=query_coverage,
            answer_focus=answer_focus,
            topic_alignment=topic_alignment,
            completeness=completeness,
            confidence=confidence
        )
        
        logger.info(f"Relevance assessment completed: score={overall_score:.3f}, confidence={confidence:.3f}")
        
        return result
    
    def _calculate_semantic_similarity(self, query: str, response: str) -> float:
        """Calculate semantic similarity between query and response"""
        if self.model is not None and cosine_similarity is not None:
            try:
                query_embedding = self.model.encode([query])
                response_embedding = self.model.encode([response])
                
                similarity = cosine_similarity(query_embedding, response_embedding)[0][0]
                return float(similarity)
                
            except Exception as e:
                logger.error(f"Semantic similarity calculation failed: {e}")
        
        # Fallback to lexical similarity
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        if not query_words or not response_words:
            return 0.0
        
        intersection = len(query_words & response_words)
        union = len(query_words | response_words)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_query_coverage(self, query: str, response: str) -> float:
        """Calculate how well the response covers the query terms"""
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        if not query_words:
            return 1.0
        
        # Remove common stop words for better coverage calculation
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'what', 'when', 'where', 'why', 'how', 'who', 'which'}
        
        query_content_words = query_words - stop_words
        
        if not query_content_words:
            return 1.0
        
        covered_words = query_content_words & response_words
        coverage = len(covered_words) / len(query_content_words)
        
        return coverage
    
    def _calculate_answer_focus(self, query: str, response: str) -> float:
        """Calculate how focused the answer is on the query"""
        query_words = set(query.lower().split())
        response_words = response.lower().split()
        
        if not query_words or not response_words:
            return 0.0
        
        # Calculate the proportion of response words that are relevant to the query
        relevant_response_words = sum(1 for word in response_words if word in query_words)
        focus_score = relevant_response_words / len(response_words)
        
        # Adjust for response length (very long responses may be less focused)
        length_penalty = min(1.0, 100.0 / len(response_words)) if len(response_words) > 100 else 1.0
        
        return focus_score * length_penalty
    
    def _calculate_topic_alignment(self, query: str, response: str) -> float:
        """Calculate topic alignment between query and response"""
        # Extract key topics/entities from query and response
        query_topics = self._extract_topics(query)
        response_topics = self._extract_topics(response)
        
        if not query_topics:
            return 1.0
        
        if not response_topics:
            return 0.0
        
        # Calculate topic overlap
        common_topics = query_topics & response_topics
        alignment = len(common_topics) / len(query_topics)
        
        return alignment
    
    def _extract_topics(self, text: str) -> set:
        """Extract key topics/entities from text (simple implementation)"""
        # Simple topic extraction using capitalized words and longer words
        words = text.split()
        topics = set()
        
        for word in words:
            word = word.strip('.,!?;:"()[]{}')
            # Include capitalized words (potential entities) and longer words (potential topics)
            if len(word) > 4 and (word[0].isupper() or len(word) > 6):
                topics.add(word.lower())
        
        return topics
    
    def _calculate_completeness(self, 
                              query: str, 
                              response: str,
                              context_chunks: Optional[List[RankedChunk]] = None) -> float:
        """Calculate how complete the answer is"""
        # Analyze query complexity to determine expected completeness
        query_complexity = self._analyze_query_complexity(query)
        
        # Analyze response completeness
        response_completeness = self._analyze_response_completeness(response)
        
        # Adjust based on available context
        context_factor = 1.0
        if context_chunks:
            # More context chunks suggest more complete information is available
            context_factor = min(1.0, len(context_chunks) / 5.0)
        
        # Combine factors
        completeness = min(1.0, response_completeness * context_factor / query_complexity)
        
        return completeness
    
    def _analyze_query_complexity(self, query: str) -> float:
        """Analyze query complexity (higher = more complex)"""
        complexity_factors = []
        
        # Length factor
        word_count = len(query.split())
        length_complexity = min(2.0, word_count / 10.0)
        complexity_factors.append(length_complexity)
        
        # Question words (who, what, when, where, why, how)
        question_words = ['who', 'what', 'when', 'where', 'why', 'how']
        question_count = sum(1 for word in question_words if word in query.lower())
        question_complexity = min(2.0, question_count * 0.5)
        complexity_factors.append(question_complexity)
        
        # Multiple topics/entities
        topics = self._extract_topics(query)
        topic_complexity = min(2.0, len(topics) * 0.3)
        complexity_factors.append(topic_complexity)
        
        return max(1.0, np.mean(complexity_factors))
    
    def _analyze_response_completeness(self, response: str) -> float:
        """Analyze response completeness"""
        completeness_factors = []
        
        # Length factor (moderate length suggests completeness)
        word_count = len(response.split())
        if 50 <= word_count <= 200:
            length_completeness = 1.0
        elif word_count < 50:
            length_completeness = word_count / 50.0
        else:
            length_completeness = max(0.7, 200.0 / word_count)
        completeness_factors.append(length_completeness)
        
        # Structure indicators (lists, examples, explanations)
        structure_indicators = [
            r'\b(first|second|third|finally|additionally|furthermore|moreover)\b',
            r'\b(for example|such as|including|namely)\b',
            r'\b(because|since|therefore|thus|consequently)\b'
        ]
        
        structure_score = 0.0
        for pattern in structure_indicators:
            if re.search(pattern, response, re.IGNORECASE):
                structure_score += 0.3
        
        structure_completeness = min(1.0, structure_score)
        completeness_factors.append(structure_completeness)
        
        return np.mean(completeness_factors)
    
    def _calculate_overall_relevance(self, 
                                   semantic_similarity: float,
                                   query_coverage: float,
                                   answer_focus: float,
                                   topic_alignment: float,
                                   completeness: float) -> float:
        """Calculate overall relevance score"""
        # Weighted combination of relevance factors
        weights = {
            'semantic_similarity': 0.3,
            'query_coverage': 0.25,
            'answer_focus': 0.2,
            'topic_alignment': 0.15,
            'completeness': 0.1
        }
        
        overall_score = (
            weights['semantic_similarity'] * semantic_similarity +
            weights['query_coverage'] * query_coverage +
            weights['answer_focus'] * answer_focus +
            weights['topic_alignment'] * topic_alignment +
            weights['completeness'] * completeness
        )
        
        return overall_score
    
    def _calculate_relevance_confidence(self, 
                                      query: str,
                                      response: str,
                                      semantic_similarity: float,
                                      context_chunks: Optional[List[RankedChunk]] = None) -> float:
        """Calculate confidence in relevance assessment"""
        confidence_factors = []
        
        # Model availability (semantic models = higher confidence)
        model_confidence = 0.9 if self.model is not None else 0.6
        confidence_factors.append(model_confidence)
        
        # Query clarity (clear queries = higher confidence)
        query_length = len(query.split())
        if 5 <= query_length <= 20:
            query_confidence = 1.0
        elif query_length < 5:
            query_confidence = query_length / 5.0
        else:
            query_confidence = max(0.7, 20.0 / query_length)
        confidence_factors.append(query_confidence)
        
        # Response quality (moderate semantic similarity = higher confidence)
        if 0.3 <= semantic_similarity <= 0.9:
            similarity_confidence = 1.0
        else:
            similarity_confidence = 0.7
        confidence_factors.append(similarity_confidence)
        
        # Context availability
        if context_chunks:
            context_confidence = min(1.0, len(context_chunks) / 5.0)
        else:
            context_confidence = 0.5
        confidence_factors.append(context_confidence)
        
        return np.mean(confidence_factors)
    
    def batch_assess_relevance(self, 
                             queries_and_responses: List[Tuple[str, str]],
                             context_chunks_list: Optional[List[List[RankedChunk]]] = None) -> List[RelevanceScore]:
        """Assess relevance for multiple query-response pairs
        
        Args:
            queries_and_responses: List of (query, response) tuples
            context_chunks_list: Optional list of context chunks for each pair
            
        Returns:
            List of RelevanceScore objects
        """
        results = []
        
        for i, (query, response) in enumerate(queries_and_responses):
            context_chunks = None
            if context_chunks_list and i < len(context_chunks_list):
                context_chunks = context_chunks_list[i]
            
            score = self.assess_relevance(query, response, context_chunks)
            results.append(score)
        
        logger.info(f"Batch relevance assessment completed for {len(results)} items")
        return results
    
    def get_relevance_summary(self, scores: List[RelevanceScore]) -> Dict[str, Any]:
        """Generate summary statistics for relevance scores
        
        Args:
            scores: List of RelevanceScore objects
            
        Returns:
            Dictionary with summary statistics
        """
        if not scores:
            return {"error": "No scores provided"}
        
        overall_scores = [score.overall_score for score in scores]
        semantic_scores = [score.semantic_similarity for score in scores]
        coverage_scores = [score.query_coverage for score in scores]
        
        return {
            "total_evaluations": len(scores),
            "avg_relevance_score": np.mean(overall_scores),
            "median_relevance_score": np.median(overall_scores),
            "std_relevance_score": np.std(overall_scores),
            "min_relevance_score": np.min(overall_scores),
            "max_relevance_score": np.max(overall_scores),
            "avg_semantic_similarity": np.mean(semantic_scores),
            "avg_query_coverage": np.mean(coverage_scores),
            "avg_confidence": np.mean([score.confidence for score in scores]),
            "high_relevance_rate": sum(1 for score in scores if score.overall_score >= self.relevance_threshold) / len(scores)
        }

if __name__ == "__main__":
    print("Testing Evaluation Framework...")
    
    # Test faithfulness evaluator
    faithfulness_evaluator = FaithfulnessEvaluator()
    print("✓ FaithfulnessEvaluator created")
    
    # Test relevance assessor
    relevance_assessor = RelevanceAssessor()
    print("✓ RelevanceAssessor created")
    
    print("Evaluation framework components implemented:")
    print("  - Faithfulness evaluation with claim-level analysis")
    print("  - Source document alignment measurement")
    print("  - Hallucination detection indicators")
    print("  - Query-answer relevance scoring")
    print("  - Semantic similarity measurements")
    print("  - Topic alignment and completeness assessment")
    print("  - Confidence scoring for all evaluations")
    print("  - Batch processing capabilities")
    print("  - Summary statistics generation")

class RetrievalMetricsCalculator:
    """Calculates precision, recall, and other retrieval performance metrics"""
    
    def __init__(self):
        """Initialize retrieval metrics calculator"""
        logger.info("Initialized retrieval metrics calculator")
    
    def calculate_retrieval_metrics(self, 
                                  retrieved_chunks: List[RankedChunk],
                                  relevant_chunk_ids: List[str],
                                  query: Optional[str] = None) -> RetrievalMetrics:
        """Calculate comprehensive retrieval metrics
        
        Args:
            retrieved_chunks: List of chunks retrieved by the system
            relevant_chunk_ids: List of chunk IDs that are actually relevant
            query: Optional query for additional analysis
            
        Returns:
            RetrievalMetrics with detailed performance metrics
        """
        if not retrieved_chunks:
            return RetrievalMetrics(
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                mrr=0.0,
                ndcg=0.0,
                context_quality=0.0,
                diversity_score=0.0,
                coverage_score=0.0,
                total_retrieved=0,
                relevant_retrieved=0
            )
        
        retrieved_chunk_ids = [chunk.chunk_id for chunk in retrieved_chunks]
        
        # Calculate basic metrics
        precision = self._calculate_precision(retrieved_chunk_ids, relevant_chunk_ids)
        recall = self._calculate_recall(retrieved_chunk_ids, relevant_chunk_ids)
        f1_score = self._calculate_f1_score(precision, recall)
        
        # Calculate ranking metrics
        mrr = self._calculate_mrr(retrieved_chunk_ids, relevant_chunk_ids)
        ndcg = self._calculate_ndcg(retrieved_chunks, relevant_chunk_ids)
        
        # Calculate quality metrics
        context_quality = self._calculate_context_quality(retrieved_chunks)
        diversity_score = self._calculate_diversity_score(retrieved_chunks)
        coverage_score = self._calculate_coverage_score(retrieved_chunks, relevant_chunk_ids)
        
        # Count relevant retrieved
        relevant_retrieved = len(set(retrieved_chunk_ids) & set(relevant_chunk_ids))
        
        result = RetrievalMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            mrr=mrr,
            ndcg=ndcg,
            context_quality=context_quality,
            diversity_score=diversity_score,
            coverage_score=coverage_score,
            total_retrieved=len(retrieved_chunks),
            relevant_retrieved=relevant_retrieved
        )
        
        logger.info(f"Retrieval metrics calculated: P={precision:.3f}, R={recall:.3f}, F1={f1_score:.3f}")
        
        return result
    
    def _calculate_precision(self, retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """Calculate precision: relevant_retrieved / total_retrieved"""
        if not retrieved_ids:
            return 0.0
        
        relevant_retrieved = len(set(retrieved_ids) & set(relevant_ids))
        return relevant_retrieved / len(retrieved_ids)
    
    def _calculate_recall(self, retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """Calculate recall: relevant_retrieved / total_relevant"""
        if not relevant_ids:
            return 1.0  # Perfect recall when no relevant documents exist
        
        relevant_retrieved = len(set(retrieved_ids) & set(relevant_ids))
        return relevant_retrieved / len(relevant_ids)
    
    def _calculate_f1_score(self, precision: float, recall: float) -> float:
        """Calculate F1 score: 2 * (precision * recall) / (precision + recall)"""
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def _calculate_mrr(self, retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """Calculate Mean Reciprocal Rank"""
        relevant_set = set(relevant_ids)
        
        for i, chunk_id in enumerate(retrieved_ids):
            if chunk_id in relevant_set:
                return 1.0 / (i + 1)
        
        return 0.0  # No relevant documents found
    
    def _calculate_ndcg(self, retrieved_chunks: List[RankedChunk], relevant_ids: List[str], k: int = 10) -> float:
        """Calculate Normalized Discounted Cumulative Gain at k"""
        relevant_set = set(relevant_ids)
        
        # Calculate DCG
        dcg = 0.0
        for i, chunk in enumerate(retrieved_chunks[:k]):
            if chunk.chunk_id in relevant_set:
                # Use relevance score as gain, with discount factor
                gain = chunk.final_score
                discount = np.log2(i + 2)  # i+2 because log2(1) = 0
                dcg += gain / discount
        
        # Calculate IDCG (Ideal DCG)
        # Sort relevant chunks by score for ideal ranking
        relevant_chunks = [chunk for chunk in retrieved_chunks if chunk.chunk_id in relevant_set]
        relevant_chunks.sort(key=lambda x: x.final_score, reverse=True)
        
        idcg = 0.0
        for i, chunk in enumerate(relevant_chunks[:k]):
            gain = chunk.final_score
            discount = np.log2(i + 2)
            idcg += gain / discount
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def _calculate_context_quality(self, retrieved_chunks: List[RankedChunk]) -> float:
        """Calculate overall context quality based on chunk scores and diversity"""
        if not retrieved_chunks:
            return 0.0
        
        # Average relevance score
        avg_score = np.mean([chunk.final_score for chunk in retrieved_chunks])
        
        # Score distribution (lower variance = more consistent quality)
        scores = [chunk.final_score for chunk in retrieved_chunks]
        if len(scores) > 1:
            score_variance = np.var(scores)
            consistency = max(0.0, 1.0 - score_variance)
        else:
            consistency = 1.0
        
        # Combine factors
        quality = 0.7 * avg_score + 0.3 * consistency
        
        return min(1.0, quality)
    
    def _calculate_diversity_score(self, retrieved_chunks: List[RankedChunk]) -> float:
        """Calculate diversity of retrieved sources"""
        if not retrieved_chunks:
            return 0.0
        
        # Document diversity
        unique_documents = len(set(chunk.document_id for chunk in retrieved_chunks))
        document_diversity = unique_documents / len(retrieved_chunks)
        
        # Page diversity (within documents)
        unique_pages = len(set((chunk.document_id, chunk.page_number) for chunk in retrieved_chunks))
        page_diversity = unique_pages / len(retrieved_chunks)
        
        # Content diversity (simple lexical diversity)
        all_words = set()
        total_words = 0
        
        for chunk in retrieved_chunks:
            words = set(chunk.content.lower().split())
            all_words.update(words)
            total_words += len(words)
        
        lexical_diversity = len(all_words) / max(1, total_words)
        
        # Combine diversity measures
        diversity = (
            0.4 * document_diversity +
            0.3 * page_diversity +
            0.3 * lexical_diversity
        )
        
        return min(1.0, diversity)
    
    def _calculate_coverage_score(self, retrieved_chunks: List[RankedChunk], relevant_ids: List[str]) -> float:
        """Calculate how well the retrieved chunks cover the relevant information"""
        if not relevant_ids:
            return 1.0
        
        retrieved_ids = set(chunk.chunk_id for chunk in retrieved_chunks)
        relevant_set = set(relevant_ids)
        
        # Basic coverage (recall)
        coverage = len(retrieved_ids & relevant_set) / len(relevant_set)
        
        # Adjust for retrieval quality
        if retrieved_chunks:
            avg_score = np.mean([chunk.final_score for chunk in retrieved_chunks])
            quality_factor = min(1.0, avg_score)
            coverage = coverage * quality_factor
        
        return coverage
    
    def calculate_retrieval_metrics_with_ground_truth(self, 
                                                    retrieved_chunks: List[RankedChunk],
                                                    ground_truth_relevance: Dict[str, float],
                                                    relevance_threshold: float = 0.5) -> RetrievalMetrics:
        """Calculate metrics using ground truth relevance scores
        
        Args:
            retrieved_chunks: Retrieved chunks
            ground_truth_relevance: Dict mapping chunk_id to relevance score (0.0-1.0)
            relevance_threshold: Threshold for considering a chunk relevant
            
        Returns:
            RetrievalMetrics with ground truth-based evaluation
        """
        # Determine relevant chunks based on threshold
        relevant_chunk_ids = [
            chunk_id for chunk_id, score in ground_truth_relevance.items()
            if score >= relevance_threshold
        ]
        
        return self.calculate_retrieval_metrics(retrieved_chunks, relevant_chunk_ids)
    
    def batch_calculate_retrieval_metrics(self, 
                                        retrieved_chunks_list: List[List[RankedChunk]],
                                        relevant_chunk_ids_list: List[List[str]],
                                        queries: Optional[List[str]] = None) -> List[RetrievalMetrics]:
        """Calculate retrieval metrics for multiple queries
        
        Args:
            retrieved_chunks_list: List of retrieved chunks for each query
            relevant_chunk_ids_list: List of relevant chunk IDs for each query
            queries: Optional list of queries
            
        Returns:
            List of RetrievalMetrics objects
        """
        results = []
        
        for i, (retrieved_chunks, relevant_ids) in enumerate(zip(retrieved_chunks_list, relevant_chunk_ids_list)):
            query = queries[i] if queries and i < len(queries) else None
            
            metrics = self.calculate_retrieval_metrics(retrieved_chunks, relevant_ids, query)
            results.append(metrics)
        
        logger.info(f"Batch retrieval metrics calculation completed for {len(results)} queries")
        return results
    
    def get_retrieval_summary(self, metrics_list: List[RetrievalMetrics]) -> Dict[str, Any]:
        """Generate summary statistics for retrieval metrics
        
        Args:
            metrics_list: List of RetrievalMetrics objects
            
        Returns:
            Dictionary with summary statistics
        """
        if not metrics_list:
            return {"error": "No metrics provided"}
        
        return {
            "total_queries": len(metrics_list),
            "avg_precision": np.mean([m.precision for m in metrics_list]),
            "avg_recall": np.mean([m.recall for m in metrics_list]),
            "avg_f1_score": np.mean([m.f1_score for m in metrics_list]),
            "avg_mrr": np.mean([m.mrr for m in metrics_list]),
            "avg_ndcg": np.mean([m.ndcg for m in metrics_list]),
            "avg_context_quality": np.mean([m.context_quality for m in metrics_list]),
            "avg_diversity_score": np.mean([m.diversity_score for m in metrics_list]),
            "avg_coverage_score": np.mean([m.coverage_score for m in metrics_list]),
            "total_retrieved": sum(m.total_retrieved for m in metrics_list),
            "total_relevant_retrieved": sum(m.relevant_retrieved for m in metrics_list),
            "std_precision": np.std([m.precision for m in metrics_list]),
            "std_recall": np.std([m.recall for m in metrics_list]),
            "std_f1_score": np.std([m.f1_score for m in metrics_list])
        }

class ComparativeEvaluator:
    """Implements A/B testing and comparative evaluation for prompt strategies"""
    
    def __init__(self, significance_level: float = 0.05):
        """Initialize comparative evaluator
        
        Args:
            significance_level: Statistical significance level for tests
        """
        self.significance_level = significance_level
        logger.info("Initialized comparative evaluator")
    
    def compare_strategies(self, 
                         strategy_a_scores: List[float],
                         strategy_b_scores: List[float],
                         strategy_a_name: str = "Strategy A",
                         strategy_b_name: str = "Strategy B") -> ComparativeResult:
        """Compare two strategies using statistical testing
        
        Args:
            strategy_a_scores: Performance scores for strategy A
            strategy_b_scores: Performance scores for strategy B
            strategy_a_name: Name of strategy A
            strategy_b_name: Name of strategy B
            
        Returns:
            ComparativeResult with detailed comparison
        """
        if not strategy_a_scores or not strategy_b_scores:
            return ComparativeResult(
                strategy_a_score=0.0,
                strategy_b_score=0.0,
                improvement=0.0,
                statistical_significance=1.0,
                confidence_interval=(0.0, 0.0),
                winner="tie",
                effect_size=0.0,
                sample_size=0
            )
        
        logger.info(f"Comparing {strategy_a_name} vs {strategy_b_name}")
        
        # Calculate basic statistics
        mean_a = np.mean(strategy_a_scores)
        mean_b = np.mean(strategy_b_scores)
        
        # Calculate improvement
        if mean_a > 0:
            improvement = ((mean_b - mean_a) / mean_a) * 100
        else:
            improvement = 0.0
        
        # Perform statistical test
        p_value = self._perform_statistical_test(strategy_a_scores, strategy_b_scores)
        
        # Calculate confidence interval for the difference
        confidence_interval = self._calculate_confidence_interval(
            strategy_a_scores, strategy_b_scores
        )
        
        # Determine winner
        winner = self._determine_winner(mean_a, mean_b, p_value)
        
        # Calculate effect size (Cohen's d)
        effect_size = self._calculate_effect_size(strategy_a_scores, strategy_b_scores)
        
        result = ComparativeResult(
            strategy_a_score=mean_a,
            strategy_b_score=mean_b,
            improvement=improvement,
            statistical_significance=p_value,
            confidence_interval=confidence_interval,
            winner=winner,
            effect_size=effect_size,
            sample_size=len(strategy_a_scores) + len(strategy_b_scores)
        )
        
        logger.info(f"Comparison completed: {winner} wins, p={p_value:.4f}, effect_size={effect_size:.3f}")
        
        return result
    
    def _perform_statistical_test(self, scores_a: List[float], scores_b: List[float]) -> float:
        """Perform statistical test to compare two groups"""
        try:
            from scipy import stats
            
            # Check for normality (simplified)
            if len(scores_a) >= 8 and len(scores_b) >= 8:
                # Use t-test for larger samples
                statistic, p_value = stats.ttest_ind(scores_a, scores_b)
            else:
                # Use Mann-Whitney U test for smaller samples (non-parametric)
                statistic, p_value = stats.mannwhitneyu(scores_a, scores_b, alternative='two-sided')
            
            return float(p_value)
            
        except ImportError:
            logger.warning("scipy not available, using simplified statistical test")
            return self._simplified_statistical_test(scores_a, scores_b)
    
    def _simplified_statistical_test(self, scores_a: List[float], scores_b: List[float]) -> float:
        """Simplified statistical test when scipy is not available"""
        # Simple permutation test
        observed_diff = abs(np.mean(scores_a) - np.mean(scores_b))
        
        # Combine all scores
        all_scores = scores_a + scores_b
        n_a = len(scores_a)
        
        # Perform permutation test (simplified with fewer permutations)
        n_permutations = min(1000, 2**(len(all_scores)))
        extreme_count = 0
        
        np.random.seed(42)  # For reproducibility
        
        for _ in range(n_permutations):
            # Randomly shuffle and split
            shuffled = np.random.permutation(all_scores)
            perm_a = shuffled[:n_a]
            perm_b = shuffled[n_a:]
            
            perm_diff = abs(np.mean(perm_a) - np.mean(perm_b))
            
            if perm_diff >= observed_diff:
                extreme_count += 1
        
        p_value = extreme_count / n_permutations
        return max(0.001, p_value)  # Avoid p=0
    
    def _calculate_confidence_interval(self, 
                                     scores_a: List[float], 
                                     scores_b: List[float],
                                     confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for the difference in means"""
        mean_a = np.mean(scores_a)
        mean_b = np.mean(scores_b)
        diff = mean_b - mean_a
        
        # Calculate standard error of the difference
        var_a = np.var(scores_a, ddof=1) if len(scores_a) > 1 else 0
        var_b = np.var(scores_b, ddof=1) if len(scores_b) > 1 else 0
        
        se_diff = np.sqrt(var_a / len(scores_a) + var_b / len(scores_b))
        
        # Use t-distribution critical value (approximation)
        df = len(scores_a) + len(scores_b) - 2
        alpha = 1 - confidence_level
        
        # Simplified critical value (approximation for t-distribution)
        if df >= 30:
            t_critical = 1.96  # Normal approximation
        elif df >= 10:
            t_critical = 2.1   # Rough t-distribution approximation
        else:
            t_critical = 2.5   # Conservative estimate for small samples
        
        margin_of_error = t_critical * se_diff
        
        return (diff - margin_of_error, diff + margin_of_error)
    
    def _determine_winner(self, mean_a: float, mean_b: float, p_value: float) -> str:
        """Determine the winner based on means and statistical significance"""
        if p_value > self.significance_level:
            return "tie"
        elif mean_b > mean_a:
            return "B"
        else:
            return "A"
    
    def _calculate_effect_size(self, scores_a: List[float], scores_b: List[float]) -> float:
        """Calculate Cohen's d effect size"""
        mean_a = np.mean(scores_a)
        mean_b = np.mean(scores_b)
        
        # Calculate pooled standard deviation
        var_a = np.var(scores_a, ddof=1) if len(scores_a) > 1 else 0
        var_b = np.var(scores_b, ddof=1) if len(scores_b) > 1 else 0
        
        pooled_std = np.sqrt(((len(scores_a) - 1) * var_a + (len(scores_b) - 1) * var_b) / 
                           (len(scores_a) + len(scores_b) - 2))
        
        if pooled_std == 0:
            return 0.0
        
        cohens_d = (mean_b - mean_a) / pooled_std
        return abs(cohens_d)
    
    def compare_multiple_strategies(self, 
                                  strategy_scores: Dict[str, List[float]]) -> Dict[str, Dict[str, ComparativeResult]]:
        """Compare multiple strategies pairwise
        
        Args:
            strategy_scores: Dict mapping strategy names to score lists
            
        Returns:
            Dict of pairwise comparison results
        """
        results = {}
        strategy_names = list(strategy_scores.keys())
        
        for i, strategy_a in enumerate(strategy_names):
            results[strategy_a] = {}
            
            for j, strategy_b in enumerate(strategy_names):
                if i != j:
                    comparison = self.compare_strategies(
                        strategy_scores[strategy_a],
                        strategy_scores[strategy_b],
                        strategy_a,
                        strategy_b
                    )
                    results[strategy_a][strategy_b] = comparison
        
        logger.info(f"Multiple strategy comparison completed for {len(strategy_names)} strategies")
        return results
    
    def rank_strategies(self, strategy_scores: Dict[str, List[float]]) -> List[Tuple[str, float, float]]:
        """Rank strategies by performance
        
        Args:
            strategy_scores: Dict mapping strategy names to score lists
            
        Returns:
            List of (strategy_name, mean_score, std_score) tuples, sorted by mean score
        """
        strategy_stats = []
        
        for name, scores in strategy_scores.items():
            if scores:
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                strategy_stats.append((name, mean_score, std_score))
        
        # Sort by mean score (descending)
        strategy_stats.sort(key=lambda x: x[1], reverse=True)
        
        return strategy_stats
    
    def generate_comparison_report(self, 
                                 comparison_results: Dict[str, Dict[str, ComparativeResult]],
                                 strategy_scores: Dict[str, List[float]]) -> Dict[str, Any]:
        """Generate comprehensive comparison report
        
        Args:
            comparison_results: Pairwise comparison results
            strategy_scores: Original strategy scores
            
        Returns:
            Comprehensive comparison report
        """
        # Rank strategies
        rankings = self.rank_strategies(strategy_scores)
        
        # Find best strategy
        best_strategy = rankings[0][0] if rankings else None
        
        # Count significant wins for each strategy
        win_counts = defaultdict(int)
        for strategy_a, comparisons in comparison_results.items():
            for strategy_b, result in comparisons.items():
                if result.winner == "A":  # Strategy A wins
                    win_counts[strategy_a] += 1
        
        # Calculate average effect sizes
        avg_effect_sizes = {}
        for strategy_a, comparisons in comparison_results.items():
            effect_sizes = [result.effect_size for result in comparisons.values()]
            avg_effect_sizes[strategy_a] = np.mean(effect_sizes) if effect_sizes else 0.0
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_strategies": len(strategy_scores),
            "rankings": rankings,
            "best_strategy": best_strategy,
            "win_counts": dict(win_counts),
            "avg_effect_sizes": avg_effect_sizes,
            "significant_comparisons": sum(
                1 for comparisons in comparison_results.values()
                for result in comparisons.values()
                if result.statistical_significance <= self.significance_level
            ),
            "total_comparisons": sum(len(comparisons) for comparisons in comparison_results.values()),
            "strategy_statistics": {
                name: {
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                    "min": np.min(scores),
                    "max": np.max(scores),
                    "count": len(scores)
                }
                for name, scores in strategy_scores.items()
            }
        }
        
        return report

class EvaluationFramework:
    """Comprehensive evaluation framework combining all evaluation components"""
    
    def __init__(self, 
                 faithfulness_model: str = "all-MiniLM-L6-v2",
                 relevance_model: str = "all-MiniLM-L6-v2",
                 significance_level: float = 0.05):
        """Initialize comprehensive evaluation framework
        
        Args:
            faithfulness_model: Model for faithfulness evaluation
            relevance_model: Model for relevance assessment
            significance_level: Statistical significance level
        """
        self.faithfulness_evaluator = FaithfulnessEvaluator(model_name=faithfulness_model)
        self.relevance_assessor = RelevanceAssessor(model_name=relevance_model)
        self.retrieval_calculator = RetrievalMetricsCalculator()
        self.comparative_evaluator = ComparativeEvaluator(significance_level=significance_level)
        
        logger.info("Initialized comprehensive evaluation framework")
    
    def evaluate_response(self, 
                         query: str,
                         response: str,
                         source_chunks: List[RankedChunk],
                         relevant_chunk_ids: Optional[List[str]] = None) -> EvaluationReport:
        """Perform comprehensive evaluation of a query-response pair
        
        Args:
            query: Original user query
            response: Generated response
            source_chunks: Source chunks used for generation
            relevant_chunk_ids: Optional ground truth relevant chunk IDs
            
        Returns:
            Comprehensive EvaluationReport
        """
        logger.info(f"Performing comprehensive evaluation for query: '{query[:50]}...'")
        
        # Evaluate faithfulness
        faithfulness = self.faithfulness_evaluator.evaluate_faithfulness(
            response, source_chunks, query
        )
        
        # Assess relevance
        relevance = self.relevance_assessor.assess_relevance(
            query, response, source_chunks
        )
        
        # Calculate retrieval metrics (if ground truth available)
        retrieval_metrics = None
        if relevant_chunk_ids:
            retrieval_metrics = self.retrieval_calculator.calculate_retrieval_metrics(
                source_chunks, relevant_chunk_ids, query
            )
        
        # Calculate overall quality score
        overall_quality = self._calculate_overall_quality(faithfulness, relevance, retrieval_metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(faithfulness, relevance, retrieval_metrics)
        
        report = EvaluationReport(
            timestamp=datetime.now(),
            query=query,
            response=response,
            source_chunks=source_chunks,
            faithfulness=faithfulness,
            relevance=relevance,
            retrieval_metrics=retrieval_metrics,
            overall_quality=overall_quality,
            recommendations=recommendations
        )
        
        logger.info(f"Comprehensive evaluation completed: quality={overall_quality:.3f}")
        
        return report
    
    def _calculate_overall_quality(self, 
                                 faithfulness: FaithfulnessScore,
                                 relevance: RelevanceScore,
                                 retrieval_metrics: Optional[RetrievalMetrics]) -> float:
        """Calculate overall quality score"""
        weights = {
            'faithfulness': 0.4,
            'relevance': 0.4,
            'retrieval': 0.2
        }
        
        quality_score = (
            weights['faithfulness'] * faithfulness.overall_score +
            weights['relevance'] * relevance.overall_score
        )
        
        if retrieval_metrics:
            quality_score += weights['retrieval'] * retrieval_metrics.f1_score
        else:
            # Redistribute retrieval weight to other components
            quality_score = quality_score / (1 - weights['retrieval'])
        
        return min(1.0, quality_score)
    
    def _generate_recommendations(self, 
                                faithfulness: FaithfulnessScore,
                                relevance: RelevanceScore,
                                retrieval_metrics: Optional[RetrievalMetrics]) -> List[str]:
        """Generate improvement recommendations based on evaluation results"""
        recommendations = []
        
        # Faithfulness recommendations
        if faithfulness.overall_score < 0.7:
            recommendations.append("Improve response faithfulness by ensuring all claims are supported by source documents")
            
            if faithfulness.hallucination_indicators:
                recommendations.append("Address potential hallucination indicators in the response")
        
        if faithfulness.source_alignment < 0.6:
            recommendations.append("Improve alignment between response and source documents")
        
        # Relevance recommendations
        if relevance.overall_score < 0.7:
            recommendations.append("Improve response relevance to the user query")
            
            if relevance.query_coverage < 0.6:
                recommendations.append("Ensure response covers more aspects of the user query")
            
            if relevance.answer_focus < 0.6:
                recommendations.append("Make response more focused on the specific query")
        
        # Retrieval recommendations
        if retrieval_metrics:
            if retrieval_metrics.precision < 0.6:
                recommendations.append("Improve retrieval precision by filtering out irrelevant chunks")
            
            if retrieval_metrics.recall < 0.6:
                recommendations.append("Improve retrieval recall by expanding search strategies")
            
            if retrieval_metrics.diversity_score < 0.5:
                recommendations.append("Increase diversity of retrieved sources")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Overall performance is good, consider fine-tuning for specific use cases")
        
        return recommendations
    
    def batch_evaluate(self, 
                      evaluation_data: List[Tuple[str, str, List[RankedChunk]]],
                      relevant_chunk_ids_list: Optional[List[List[str]]] = None) -> List[EvaluationReport]:
        """Perform batch evaluation on multiple query-response pairs
        
        Args:
            evaluation_data: List of (query, response, source_chunks) tuples
            relevant_chunk_ids_list: Optional ground truth for each evaluation
            
        Returns:
            List of EvaluationReport objects
        """
        reports = []
        
        for i, (query, response, source_chunks) in enumerate(evaluation_data):
            relevant_ids = None
            if relevant_chunk_ids_list and i < len(relevant_chunk_ids_list):
                relevant_ids = relevant_chunk_ids_list[i]
            
            report = self.evaluate_response(query, response, source_chunks, relevant_ids)
            reports.append(report)
        
        logger.info(f"Batch evaluation completed for {len(reports)} items")
        return reports
    
    def generate_evaluation_summary(self, reports: List[EvaluationReport]) -> Dict[str, Any]:
        """Generate summary statistics for evaluation reports
        
        Args:
            reports: List of EvaluationReport objects
            
        Returns:
            Dictionary with comprehensive summary statistics
        """
        if not reports:
            return {"error": "No reports provided"}
        
        # Extract scores
        faithfulness_scores = [report.faithfulness for report in reports]
        relevance_scores = [report.relevance for report in reports]
        retrieval_metrics = [report.retrieval_metrics for report in reports if report.retrieval_metrics]
        overall_qualities = [report.overall_quality for report in reports]
        
        # Generate summaries
        faithfulness_summary = self.faithfulness_evaluator.get_faithfulness_summary(faithfulness_scores)
        relevance_summary = self.relevance_assessor.get_relevance_summary(relevance_scores)
        
        retrieval_summary = {}
        if retrieval_metrics:
            retrieval_summary = self.retrieval_calculator.get_retrieval_summary(retrieval_metrics)
        
        # Overall statistics
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_evaluations": len(reports),
            "overall_quality": {
                "mean": np.mean(overall_qualities),
                "median": np.median(overall_qualities),
                "std": np.std(overall_qualities),
                "min": np.min(overall_qualities),
                "max": np.max(overall_qualities)
            },
            "faithfulness_summary": faithfulness_summary,
            "relevance_summary": relevance_summary,
            "retrieval_summary": retrieval_summary,
            "common_recommendations": self._get_common_recommendations(reports)
        }
        
        return summary
    
    def _get_common_recommendations(self, reports: List[EvaluationReport]) -> List[Tuple[str, int]]:
        """Get most common recommendations across all reports"""
        recommendation_counts = defaultdict(int)
        
        for report in reports:
            for recommendation in report.recommendations:
                recommendation_counts[recommendation] += 1
        
        # Sort by frequency
        common_recommendations = sorted(
            recommendation_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return common_recommendations[:5]  # Top 5 most common

# Factory functions for creating evaluation components
def get_faithfulness_evaluator(model_name: str = "all-MiniLM-L6-v2") -> FaithfulnessEvaluator:
    """Factory function to create FaithfulnessEvaluator"""
    return FaithfulnessEvaluator(model_name=model_name)

def get_relevance_assessor(model_name: str = "all-MiniLM-L6-v2") -> RelevanceAssessor:
    """Factory function to create RelevanceAssessor"""
    return RelevanceAssessor(model_name=model_name)

def get_retrieval_calculator() -> RetrievalMetricsCalculator:
    """Factory function to create RetrievalMetricsCalculator"""
    return RetrievalMetricsCalculator()

def get_comparative_evaluator(significance_level: float = 0.05) -> ComparativeEvaluator:
    """Factory function to create ComparativeEvaluator"""
    return ComparativeEvaluator(significance_level=significance_level)

def get_evaluation_framework() -> EvaluationFramework:
    """Factory function to create comprehensive EvaluationFramework"""
    return EvaluationFramework()

if __name__ == "__main__":
    print("Testing complete Evaluation Framework...")
    
    # Test retrieval metrics calculator
    retrieval_calculator = RetrievalMetricsCalculator()
    print("✓ RetrievalMetricsCalculator created")
    
    # Test comparative evaluator
    comparative_evaluator = ComparativeEvaluator()
    print("✓ ComparativeEvaluator created")
    
    # Test comprehensive framework
    framework = EvaluationFramework()
    print("✓ EvaluationFramework created")
    
    print("Complete evaluation framework implemented:")
    print("  - Faithfulness evaluation with claim-level analysis")
    print("  - Relevance assessment with semantic similarity")
    print("  - Retrieval metrics (precision, recall, F1, MRR, NDCG)")
    print("  - Comparative evaluation with statistical testing")
    print("  - A/B testing for prompt strategies")
    print("  - Comprehensive evaluation reports")
    print("  - Batch processing capabilities")
    print("  - Summary statistics and recommendations")
    print("  - Factory functions for easy instantiation")