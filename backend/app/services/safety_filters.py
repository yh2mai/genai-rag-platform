"""
Safety Filters Service

This module implements comprehensive safety filters for the Enterprise GenAI platform,
including PII detection, hallucination detection, and content filtering mechanisms.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import spacy
from transformers import pipeline
import numpy as np
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)


class PIIType(Enum):
    """Types of PII that can be detected"""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    PERSON_NAME = "person_name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"
    BANK_ACCOUNT = "bank_account"
    IP_ADDRESS = "ip_address"


class SafetyViolationType(Enum):
    """Types of safety violations"""
    PII_DETECTED = "pii_detected"
    HALLUCINATION_DETECTED = "hallucination_detected"
    INAPPROPRIATE_CONTENT = "inappropriate_content"
    TOXIC_CONTENT = "toxic_content"


@dataclass
class PIIDetection:
    """Represents a detected PII instance"""
    pii_type: PIIType
    text: str
    start_pos: int
    end_pos: int
    confidence: float
    masked_text: str


@dataclass
class SafetyViolation:
    """Represents a safety violation"""
    violation_type: SafetyViolationType
    description: str
    confidence: float
    details: Dict[str, Any]
    suggested_action: str


class PIIDetector:
    """Detects Personally Identifiable Information using regex and NLP models"""
    
    def __init__(self):
        self.patterns = {
            PIIType.EMAIL: re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            PIIType.PHONE: re.compile(r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'),
            PIIType.SSN: re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),
            PIIType.CREDIT_CARD: re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
            PIIType.DATE_OF_BIRTH: re.compile(r'\b(?:0[1-9]|1[0-2])[/-](?:0[1-9]|[12]\d|3[01])[/-](?:19|20)\d{2}\b'),
            PIIType.BANK_ACCOUNT: re.compile(r'\b\d{8,17}\b'),
            PIIType.IP_ADDRESS: re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b')
        }
        
        # Load spaCy model for named entity recognition
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def detect_pii_regex(self, text: str) -> List[PIIDetection]:
        """Detect PII using regex patterns"""
        detections = []
        
        for pii_type, pattern in self.patterns.items():
            matches = pattern.finditer(text)
            for match in matches:
                masked_text = self._mask_text(match.group(), pii_type)
                detection = PIIDetection(
                    pii_type=pii_type,
                    text=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.9,  # High confidence for regex matches
                    masked_text=masked_text
                )
                detections.append(detection)
        
        return detections
    
    def detect_pii_nlp(self, text: str) -> List[PIIDetection]:
        """Detect PII using NLP models (spaCy NER)"""
        detections = []
        
        if not self.nlp:
            return detections
        
        doc = self.nlp(text)
        
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                masked_text = self._mask_text(ent.text, PIIType.PERSON_NAME)
                detection = PIIDetection(
                    pii_type=PIIType.PERSON_NAME,
                    text=ent.text,
                    start_pos=ent.start_char,
                    end_pos=ent.end_char,
                    confidence=0.8,  # Medium confidence for NER
                    masked_text=masked_text
                )
                detections.append(detection)
            elif ent.label_ in ["GPE", "LOC"]:  # Geopolitical entities, locations
                masked_text = self._mask_text(ent.text, PIIType.ADDRESS)
                detection = PIIDetection(
                    pii_type=PIIType.ADDRESS,
                    text=ent.text,
                    start_pos=ent.start_char,
                    end_pos=ent.end_char,
                    confidence=0.7,  # Lower confidence for location detection
                    masked_text=masked_text
                )
                detections.append(detection)
        
        return detections
    
    def _mask_text(self, text: str, pii_type: PIIType) -> str:
        """Mask PII text based on type"""
        if pii_type == PIIType.EMAIL:
            parts = text.split('@')
            if len(parts) == 2:
                return f"{'*' * len(parts[0])}@{parts[1]}"
        elif pii_type == PIIType.PHONE:
            return "***-***-****"
        elif pii_type == PIIType.SSN:
            return "***-**-****"
        elif pii_type == PIIType.CREDIT_CARD:
            return "**** **** **** ****"
        elif pii_type == PIIType.PERSON_NAME:
            return "[PERSON]"
        elif pii_type == PIIType.ADDRESS:
            return "[LOCATION]"
        else:
            return "*" * len(text)
        
        return text
    
    def detect_all_pii(self, text: str) -> List[PIIDetection]:
        """Detect all PII using both regex and NLP approaches"""
        regex_detections = self.detect_pii_regex(text)
        nlp_detections = self.detect_pii_nlp(text)
        
        # Combine and deduplicate detections
        all_detections = regex_detections + nlp_detections
        return self._deduplicate_detections(all_detections)
    
    def _deduplicate_detections(self, detections: List[PIIDetection]) -> List[PIIDetection]:
        """Remove duplicate PII detections based on position overlap"""
        if not detections:
            return detections
        
        # Sort by start position
        detections.sort(key=lambda x: x.start_pos)
        
        deduplicated = []
        for detection in detections:
            # Check if this detection overlaps with any existing detection
            overlaps = False
            for existing in deduplicated:
                if (detection.start_pos < existing.end_pos and 
                    detection.end_pos > existing.start_pos):
                    # Keep the detection with higher confidence
                    if detection.confidence > existing.confidence:
                        deduplicated.remove(existing)
                        deduplicated.append(detection)
                    overlaps = True
                    break
            
            if not overlaps:
                deduplicated.append(detection)
        
        return deduplicated


class HallucinationDetector:
    """Detects potential hallucinations by comparing responses with source context"""
    
    def __init__(self):
        # Load sentence transformer for semantic similarity
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
            self.sentence_model = None
        
        # Load entailment model for factual consistency
        try:
            self.entailment_model = pipeline(
                "text-classification",
                model="microsoft/DialoGPT-medium",
                return_all_scores=True
            )
        except Exception as e:
            logger.warning(f"Failed to load entailment model: {e}")
            self.entailment_model = None
    
    def detect_hallucination(self, response: str, context: str) -> Tuple[bool, float, str]:
        """
        Detect potential hallucination in response given the context
        
        Returns:
            Tuple of (is_hallucination, confidence_score, explanation)
        """
        if not self.sentence_model:
            return False, 0.0, "Hallucination detection unavailable"
        
        # Method 1: Semantic similarity check
        similarity_score = self._calculate_semantic_similarity(response, context)
        
        # Method 2: Factual consistency check
        consistency_score = self._check_factual_consistency(response, context)
        
        # Method 3: Novel information detection
        novelty_score = self._detect_novel_information(response, context)
        
        # Combine scores to determine hallucination likelihood
        hallucination_score = self._combine_hallucination_scores(
            similarity_score, consistency_score, novelty_score
        )
        
        is_hallucination = hallucination_score > 0.7  # Threshold for hallucination
        
        explanation = self._generate_hallucination_explanation(
            similarity_score, consistency_score, novelty_score, hallucination_score
        )
        
        return is_hallucination, hallucination_score, explanation
    
    def _calculate_semantic_similarity(self, response: str, context: str) -> float:
        """Calculate semantic similarity between response and context"""
        if not self.sentence_model:
            return 0.5  # Neutral score if model unavailable
        
        try:
            response_embedding = self.sentence_model.encode([response])
            context_embedding = self.sentence_model.encode([context])
            
            similarity = util.cos_sim(response_embedding, context_embedding)[0][0].item()
            return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.5
    
    def _check_factual_consistency(self, response: str, context: str) -> float:
        """Check factual consistency between response and context"""
        # Simple heuristic: check if key entities in response appear in context
        if not self.sentence_model:
            return 0.5
        
        # Extract key phrases from response (simple approach)
        response_words = set(response.lower().split())
        context_words = set(context.lower().split())
        
        # Calculate overlap ratio
        if not response_words:
            return 1.0  # Empty response is consistent
        
        overlap = len(response_words.intersection(context_words))
        consistency_score = overlap / len(response_words)
        
        return consistency_score
    
    def _detect_novel_information(self, response: str, context: str) -> float:
        """Detect if response contains information not present in context"""
        # Simple approach: check for specific claims or numbers not in context
        response_lower = response.lower()
        context_lower = context.lower()
        
        # Look for specific patterns that might indicate novel information
        novel_patterns = [
            r'\b\d{4}\b',  # Years
            r'\b\d+%\b',   # Percentages
            r'\$\d+',      # Dollar amounts
            r'\b\d+\.\d+\b'  # Decimal numbers
        ]
        
        novel_count = 0
        total_patterns = 0
        
        for pattern in novel_patterns:
            response_matches = re.findall(pattern, response_lower)
            for match in response_matches:
                total_patterns += 1
                if match not in context_lower:
                    novel_count += 1
        
        if total_patterns == 0:
            return 0.0  # No specific claims to verify
        
        return novel_count / total_patterns
    
    def _combine_hallucination_scores(self, similarity: float, consistency: float, novelty: float) -> float:
        """Combine different scores to determine overall hallucination likelihood"""
        # Lower similarity and consistency, higher novelty = higher hallucination risk
        hallucination_score = (
            (1.0 - similarity) * 0.4 +  # 40% weight on semantic similarity
            (1.0 - consistency) * 0.4 +  # 40% weight on factual consistency
            novelty * 0.2  # 20% weight on novel information
        )
        
        return max(0.0, min(1.0, hallucination_score))
    
    def _generate_hallucination_explanation(self, similarity: float, consistency: float, 
                                          novelty: float, overall: float) -> str:
        """Generate human-readable explanation of hallucination detection"""
        explanations = []
        
        if similarity < 0.3:
            explanations.append("Low semantic similarity with source context")
        
        if consistency < 0.3:
            explanations.append("Poor factual consistency with source material")
        
        if novelty > 0.7:
            explanations.append("Contains information not present in source context")
        
        if overall > 0.7:
            explanations.append("High likelihood of hallucination detected")
        elif overall > 0.4:
            explanations.append("Moderate risk of hallucination")
        else:
            explanations.append("Low risk of hallucination")
        
        return "; ".join(explanations) if explanations else "Response appears grounded in context"


class ContentFilter:
    """Filters inappropriate and toxic content"""
    
    def __init__(self):
        # Load toxicity detection model
        try:
            self.toxicity_model = pipeline(
                "text-classification",
                model="unitary/toxic-bert",
                return_all_scores=True
            )
        except Exception as e:
            logger.warning(f"Failed to load toxicity model: {e}")
            self.toxicity_model = None
        
        # Define inappropriate content patterns
        self.inappropriate_patterns = [
            r'\b(?:hate|violence|discrimination)\b',
            r'\b(?:illegal|harmful|dangerous)\b',
            r'\b(?:explicit|inappropriate|offensive)\b'
        ]
    
    def detect_toxic_content(self, text: str) -> Tuple[bool, float, str]:
        """Detect toxic content using ML model"""
        if not self.toxicity_model:
            return False, 0.0, "Toxicity detection unavailable"
        
        try:
            results = self.toxicity_model(text)
            
            # Find toxicity score
            toxicity_score = 0.0
            for result in results[0]:  # First (and only) text result
                if result['label'].lower() in ['toxic', 'toxicity', '1']:
                    toxicity_score = result['score']
                    break
            
            is_toxic = toxicity_score > 0.7  # Threshold for toxicity
            explanation = f"Toxicity score: {toxicity_score:.2f}"
            
            return is_toxic, toxicity_score, explanation
        
        except Exception as e:
            logger.error(f"Error in toxicity detection: {e}")
            return False, 0.0, f"Error in toxicity detection: {str(e)}"
    
    def detect_inappropriate_content(self, text: str) -> Tuple[bool, float, List[str]]:
        """Detect inappropriate content using pattern matching"""
        matches = []
        text_lower = text.lower()
        
        for pattern in self.inappropriate_patterns:
            pattern_matches = re.findall(pattern, text_lower, re.IGNORECASE)
            matches.extend(pattern_matches)
        
        is_inappropriate = len(matches) > 0
        confidence = min(1.0, len(matches) * 0.3)  # Scale confidence based on matches
        
        return is_inappropriate, confidence, matches


class SafetyFilters:
    """Main safety filters service that coordinates all safety checks"""
    
    def __init__(self):
        self.pii_detector = PIIDetector()
        self.hallucination_detector = HallucinationDetector()
        self.content_filter = ContentFilter()
        logger.info("Safety filters initialized")
    
    def check_content_safety(self, text: str, context: Optional[str] = None) -> List[SafetyViolation]:
        """
        Perform comprehensive safety checks on content
        
        Args:
            text: The text to check for safety violations
            context: Optional context for hallucination detection
            
        Returns:
            List of detected safety violations
        """
        violations = []
        
        # 1. PII Detection
        pii_detections = self.pii_detector.detect_all_pii(text)
        if pii_detections:
            violation = SafetyViolation(
                violation_type=SafetyViolationType.PII_DETECTED,
                description=f"Detected {len(pii_detections)} PII instances",
                confidence=max(d.confidence for d in pii_detections),
                details={"pii_detections": [
                    {
                        "type": d.pii_type.value,
                        "text": d.text,
                        "masked": d.masked_text,
                        "position": (d.start_pos, d.end_pos)
                    } for d in pii_detections
                ]},
                suggested_action="Remove or mask PII before processing"
            )
            violations.append(violation)
        
        # 2. Hallucination Detection (if context provided)
        if context:
            is_hallucination, confidence, explanation = self.hallucination_detector.detect_hallucination(text, context)
            if is_hallucination:
                violation = SafetyViolation(
                    violation_type=SafetyViolationType.HALLUCINATION_DETECTED,
                    description="Potential hallucination detected",
                    confidence=confidence,
                    details={"explanation": explanation},
                    suggested_action="Verify response against source documents"
                )
                violations.append(violation)
        
        # 3. Toxic Content Detection
        is_toxic, toxicity_score, toxicity_explanation = self.content_filter.detect_toxic_content(text)
        if is_toxic:
            violation = SafetyViolation(
                violation_type=SafetyViolationType.TOXIC_CONTENT,
                description="Toxic content detected",
                confidence=toxicity_score,
                details={"explanation": toxicity_explanation},
                suggested_action="Review and filter toxic content"
            )
            violations.append(violation)
        
        # 4. Inappropriate Content Detection
        is_inappropriate, inappropriate_confidence, matches = self.content_filter.detect_inappropriate_content(text)
        if is_inappropriate:
            violation = SafetyViolation(
                violation_type=SafetyViolationType.INAPPROPRIATE_CONTENT,
                description="Inappropriate content detected",
                confidence=inappropriate_confidence,
                details={"matches": matches},
                suggested_action="Review content for appropriateness"
            )
            violations.append(violation)
        
        return violations
    
    def mask_pii_in_text(self, text: str) -> Tuple[str, List[PIIDetection]]:
        """
        Mask PII in text and return both masked text and detections
        
        Returns:
            Tuple of (masked_text, pii_detections)
        """
        pii_detections = self.pii_detector.detect_all_pii(text)
        
        if not pii_detections:
            return text, pii_detections
        
        # Sort detections by position (reverse order to maintain positions)
        pii_detections.sort(key=lambda x: x.start_pos, reverse=True)
        
        masked_text = text
        for detection in pii_detections:
            masked_text = (
                masked_text[:detection.start_pos] + 
                detection.masked_text + 
                masked_text[detection.end_pos:]
            )
        
        return masked_text, pii_detections
    
    def is_content_safe(self, text: str, context: Optional[str] = None) -> bool:
        """
        Quick check if content is safe (no violations detected)
        
        Returns:
            True if content is safe, False if violations detected
        """
        violations = self.check_content_safety(text, context)
        return len(violations) == 0
    
    def get_safety_summary(self, text: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive safety summary for content
        
        Returns:
            Dictionary with safety analysis results
        """
        violations = self.check_content_safety(text, context)
        masked_text, pii_detections = self.mask_pii_in_text(text)
        
        return {
            "is_safe": len(violations) == 0,
            "violations": [
                {
                    "type": v.violation_type.value,
                    "description": v.description,
                    "confidence": v.confidence,
                    "details": v.details,
                    "suggested_action": v.suggested_action
                } for v in violations
            ],
            "pii_detected": len(pii_detections) > 0,
            "pii_count": len(pii_detections),
            "masked_text": masked_text,
            "original_length": len(text),
            "masked_length": len(masked_text)
        }


# Singleton instance for global use
safety_filters = SafetyFilters()