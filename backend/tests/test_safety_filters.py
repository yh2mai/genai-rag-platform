"""
Tests for Safety Filters Service

This module contains comprehensive tests for PII detection, hallucination detection,
and content filtering functionality.
"""

import pytest
from unittest.mock import Mock, patch
import sys
import os
from hypothesis import given, strategies as st, settings, HealthCheck

# Add the parent directory to the path to import the app modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.services.safety_filters import (
    SafetyFilters, PIIDetector, HallucinationDetector, ContentFilter,
    PIIType, SafetyViolationType, PIIDetection, SafetyViolation
)


# Test data generators for property-based testing
@st.composite
def text_with_pii(draw):
    """Generate text containing various types of PII"""
    base_text = draw(st.text(min_size=10, max_size=200, alphabet='abcdefghijklmnopqrstuvwxyz '))
    
    # Add PII elements
    email = f"{draw(st.text(min_size=3, max_size=10, alphabet='abcdefghijklmnopqrstuvwxyz'))}@{draw(st.text(min_size=3, max_size=10, alphabet='abcdefghijklmnopqrstuvwxyz'))}.com"
    phone = f"({draw(st.integers(min_value=100, max_value=999))}) {draw(st.integers(min_value=100, max_value=999))}-{draw(st.integers(min_value=1000, max_value=9999))}"
    ssn = f"{draw(st.integers(min_value=100, max_value=999))}-{draw(st.integers(min_value=10, max_value=99))}-{draw(st.integers(min_value=1000, max_value=9999))}"
    
    pii_elements = [email, phone, ssn]
    selected_pii = draw(st.lists(st.sampled_from(pii_elements), min_size=1, max_size=3, unique=True))
    
    # Insert PII into text
    text_with_pii = base_text + " " + " ".join(selected_pii)
    
    return {
        'text': text_with_pii,
        'expected_pii_types': len(selected_pii),
        'contains_email': email in selected_pii,
        'contains_phone': phone in selected_pii,
        'contains_ssn': ssn in selected_pii
    }


@st.composite
def safe_text_data(draw):
    """Generate safe text without PII or inappropriate content"""
    return draw(st.text(
        min_size=10, 
        max_size=200, 
        alphabet='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?'
    ).filter(lambda x: '@' not in x and not any(char.isdigit() for char in x.replace(' ', '')[:10])))


class TestPIIDetector:
    """Test PII detection functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.pii_detector = PIIDetector()
    
    def test_email_detection(self):
        """Test email PII detection"""
        text = "Contact me at john.doe@company.com for more information."
        detections = self.pii_detector.detect_pii_regex(text)
        
        email_detections = [d for d in detections if d.pii_type == PIIType.EMAIL]
        assert len(email_detections) == 1
        assert email_detections[0].text == "john.doe@company.com"
        assert "john.doe@company.com" not in email_detections[0].masked_text
    
    def test_phone_detection(self):
        """Test phone number PII detection"""
        text = "Call me at (555) 123-4567 or 555-987-6543."
        detections = self.pii_detector.detect_pii_regex(text)
        
        phone_detections = [d for d in detections if d.pii_type == PIIType.PHONE]
        assert len(phone_detections) == 2
        assert any("555" in d.text for d in phone_detections)
    
    def test_ssn_detection(self):
        """Test SSN PII detection"""
        text = "My SSN is 123-45-6789 and my friend's is 987654321."
        detections = self.pii_detector.detect_pii_regex(text)
        
        ssn_detections = [d for d in detections if d.pii_type == PIIType.SSN]
        assert len(ssn_detections) == 2
        assert "123-45-6789" in [d.text for d in ssn_detections]
    
    def test_credit_card_detection(self):
        """Test credit card PII detection"""
        text = "My card number is 4532 1234 5678 9012."
        detections = self.pii_detector.detect_pii_regex(text)
        
        cc_detections = [d for d in detections if d.pii_type == PIIType.CREDIT_CARD]
        assert len(cc_detections) == 1
        assert "4532 1234 5678 9012" in cc_detections[0].text
    
    def test_no_pii_detection(self):
        """Test that clean text has no PII detections"""
        text = "This is a normal sentence with no personal information."
        detections = self.pii_detector.detect_all_pii(text)
        assert len(detections) == 0
    
    def test_multiple_pii_types(self):
        """Test detection of multiple PII types in one text"""
        text = "John Smith (john@email.com) can be reached at 555-123-4567. SSN: 123-45-6789"
        detections = self.pii_detector.detect_all_pii(text)
        
        # Should detect email, phone, and SSN
        pii_types = {d.pii_type for d in detections}
        assert PIIType.EMAIL in pii_types
        assert PIIType.PHONE in pii_types
        assert PIIType.SSN in pii_types
    
    def test_pii_masking(self):
        """Test PII masking functionality"""
        text = "Email: test@example.com, Phone: 555-123-4567"
        masked_text, detections = self.pii_detector._mask_text("test@example.com", PIIType.EMAIL), []
        
        assert "@example.com" in masked_text
        assert "test" not in masked_text
    
    def test_deduplication(self):
        """Test that overlapping detections are deduplicated"""
        # Create overlapping detections
        detection1 = PIIDetection(PIIType.EMAIL, "test@example.com", 0, 16, 0.9, "***@example.com")
        detection2 = PIIDetection(PIIType.EMAIL, "test@example.com", 0, 16, 0.8, "***@example.com")
        
        detections = [detection1, detection2]
        deduplicated = self.pii_detector._deduplicate_detections(detections)
        
        assert len(deduplicated) == 1
        assert deduplicated[0].confidence == 0.9  # Higher confidence kept


class TestHallucinationDetector:
    """Test hallucination detection functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.hallucination_detector = HallucinationDetector()
    
    def test_semantic_similarity_calculation(self):
        """Test semantic similarity calculation"""
        response = "The company was founded in 2020."
        context = "The organization was established in 2020."
        
        similarity = self.hallucination_detector._calculate_semantic_similarity(response, context)
        assert 0.0 <= similarity <= 1.0
        assert similarity > 0.5  # Should be reasonably similar
    
    def test_factual_consistency_check(self):
        """Test factual consistency checking"""
        response = "The company has 50 employees."
        context = "The organization employs 50 people."
        
        consistency = self.hallucination_detector._check_factual_consistency(response, context)
        assert 0.0 <= consistency <= 1.0
        assert consistency > 0.3  # Should have some consistency
    
    def test_novel_information_detection(self):
        """Test novel information detection"""
        response = "The company was founded in 2025 and has $100 million revenue."
        context = "The company was founded in 2020."
        
        novelty = self.hallucination_detector._detect_novel_information(response, context)
        assert 0.0 <= novelty <= 1.0
        assert novelty > 0.0  # Should detect novel information (2025, $100 million)
    
    def test_hallucination_detection_grounded(self):
        """Test hallucination detection with grounded response"""
        response = "The company was founded in 2020 and has 50 employees."
        context = "The organization was established in 2020 with a team of 50 people."
        
        is_hallucination, confidence, explanation = self.hallucination_detector.detect_hallucination(response, context)
        
        assert isinstance(is_hallucination, bool)
        assert 0.0 <= confidence <= 1.0
        assert isinstance(explanation, str)
        # Grounded response should have low hallucination risk
        assert confidence < 0.7
    
    def test_hallucination_detection_novel(self):
        """Test hallucination detection with novel information"""
        response = "The company was founded in 1995 and has 500 employees across 10 countries."
        context = "The company was founded in 2020 and has 50 employees."
        
        is_hallucination, confidence, explanation = self.hallucination_detector.detect_hallucination(response, context)
        
        assert isinstance(is_hallucination, bool)
        assert 0.0 <= confidence <= 1.0
        assert isinstance(explanation, str)
        # Novel information should increase hallucination risk
        assert confidence > 0.3
    
    def test_empty_context_handling(self):
        """Test handling of empty context"""
        response = "The company has many employees."
        context = ""
        
        is_hallucination, confidence, explanation = self.hallucination_detector.detect_hallucination(response, context)
        
        assert isinstance(is_hallucination, bool)
        assert 0.0 <= confidence <= 1.0
        assert isinstance(explanation, str)


class TestContentFilter:
    """Test content filtering functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.content_filter = ContentFilter()
    
    def test_inappropriate_content_detection(self):
        """Test inappropriate content detection"""
        inappropriate_text = "This discusses illegal activities and harmful behavior."
        
        is_inappropriate, confidence, matches = self.content_filter.detect_inappropriate_content(inappropriate_text)
        
        assert isinstance(is_inappropriate, bool)
        assert 0.0 <= confidence <= 1.0
        assert isinstance(matches, list)
        
        if is_inappropriate:
            assert len(matches) > 0
    
    def test_clean_content_filtering(self):
        """Test that clean content passes filtering"""
        clean_text = "This is a normal, professional discussion about technology and business."
        
        is_inappropriate, confidence, matches = self.content_filter.detect_inappropriate_content(clean_text)
        
        assert not is_inappropriate
        assert confidence == 0.0
        assert len(matches) == 0
    
    @patch('app.services.safety_filters.pipeline')
    def test_toxic_content_detection_mocked(self, mock_pipeline):
        """Test toxic content detection with mocked model"""
        # Mock the toxicity model
        mock_model = Mock()
        mock_model.return_value = [[{'label': 'toxic', 'score': 0.8}]]
        mock_pipeline.return_value = mock_model
        
        # Create new content filter with mocked model
        content_filter = ContentFilter()
        content_filter.toxicity_model = mock_model
        
        toxic_text = "I hate this stupid system!"
        is_toxic, toxicity_score, explanation = content_filter.detect_toxic_content(toxic_text)
        
        assert is_toxic
        assert toxicity_score == 0.8
        assert "0.8" in explanation
    
    def test_toxic_content_detection_unavailable(self):
        """Test toxic content detection when model is unavailable"""
        content_filter = ContentFilter()
        content_filter.toxicity_model = None
        
        text = "Any text here"
        is_toxic, toxicity_score, explanation = content_filter.detect_toxic_content(text)
        
        assert not is_toxic
        assert toxicity_score == 0.0
        assert "unavailable" in explanation.lower()


class TestSafetyFilters:
    """Test the main SafetyFilters service"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.safety_filters = SafetyFilters()
    
    def test_content_safety_check_clean(self):
        """Test safety check with clean content"""
        clean_text = "This is a normal business discussion about quarterly results."
        
        violations = self.safety_filters.check_content_safety(clean_text)
        assert isinstance(violations, list)
        assert len(violations) == 0
    
    def test_content_safety_check_with_pii(self):
        """Test safety check with PII content"""
        pii_text = "Contact John Smith at john@company.com or call 555-123-4567."
        
        violations = self.safety_filters.check_content_safety(pii_text)
        
        pii_violations = [v for v in violations if v.violation_type == SafetyViolationType.PII_DETECTED]
        assert len(pii_violations) > 0
        
        pii_violation = pii_violations[0]
        assert pii_violation.confidence > 0.0
        assert "PII" in pii_violation.description
        assert "mask" in pii_violation.suggested_action.lower()
    
    def test_content_safety_check_with_context(self):
        """Test safety check with context for hallucination detection"""
        context = "The company was founded in 2020 with 50 employees."
        response = "The company was established in 2020 and has about 50 staff members."
        
        violations = self.safety_filters.check_content_safety(response, context)
        
        # Should not detect hallucination for grounded response
        hallucination_violations = [v for v in violations if v.violation_type == SafetyViolationType.HALLUCINATION_DETECTED]
        assert len(hallucination_violations) == 0
    
    def test_pii_masking(self):
        """Test PII masking functionality"""
        text_with_pii = "Email me at test@example.com or call (555) 123-4567."
        
        masked_text, detections = self.safety_filters.mask_pii_in_text(text_with_pii)
        
        assert len(detections) > 0
        assert "test@example.com" not in masked_text
        assert "(555) 123-4567" not in masked_text
        assert "@example.com" in masked_text  # Domain should remain
    
    def test_is_content_safe(self):
        """Test quick safety check"""
        clean_text = "This is safe content."
        unsafe_text = "Contact me at personal@email.com with your SSN 123-45-6789."
        
        assert self.safety_filters.is_content_safe(clean_text)
        assert not self.safety_filters.is_content_safe(unsafe_text)
    
    def test_safety_summary(self):
        """Test comprehensive safety summary"""
        text_with_issues = "Hi John (john@email.com), your SSN 123-45-6789 is needed."
        
        summary = self.safety_filters.get_safety_summary(text_with_issues)
        
        assert isinstance(summary, dict)
        assert "is_safe" in summary
        assert "violations" in summary
        assert "pii_detected" in summary
        assert "pii_count" in summary
        assert "masked_text" in summary
        
        assert not summary["is_safe"]  # Should detect PII
        assert summary["pii_detected"]
        assert summary["pii_count"] > 0
        assert len(summary["violations"]) > 0
    
    def test_safety_summary_clean_content(self):
        """Test safety summary with clean content"""
        clean_text = "This is a normal business discussion."
        
        summary = self.safety_filters.get_safety_summary(clean_text)
        
        assert summary["is_safe"]
        assert not summary["pii_detected"]
        assert summary["pii_count"] == 0
        assert len(summary["violations"]) == 0
        assert summary["masked_text"] == clean_text  # No masking needed


class TestSafetyViolation:
    """Test SafetyViolation data structure"""
    
    def test_safety_violation_creation(self):
        """Test creating SafetyViolation objects"""
        violation = SafetyViolation(
            violation_type=SafetyViolationType.PII_DETECTED,
            description="Test violation",
            confidence=0.8,
            details={"test": "data"},
            suggested_action="Test action"
        )
        
        assert violation.violation_type == SafetyViolationType.PII_DETECTED
        assert violation.description == "Test violation"
        assert violation.confidence == 0.8
        assert violation.details == {"test": "data"}
        assert violation.suggested_action == "Test action"


class TestPIIDetection:
    """Test PIIDetection data structure"""
    
    def test_pii_detection_creation(self):
        """Test creating PIIDetection objects"""
        detection = PIIDetection(
            pii_type=PIIType.EMAIL,
            text="test@example.com",
            start_pos=0,
            end_pos=16,
            confidence=0.9,
            masked_text="****@example.com"
        )
        
        assert detection.pii_type == PIIType.EMAIL
        assert detection.text == "test@example.com"
        assert detection.start_pos == 0
        assert detection.end_pos == 16
        assert detection.confidence == 0.9
        assert detection.masked_text == "****@example.com"


# Integration tests
class TestSafetyFiltersIntegration:
    """Integration tests for safety filters"""
    
    def test_end_to_end_safety_check(self):
        """Test complete end-to-end safety checking workflow"""
        safety_filters = SafetyFilters()
        
        # Complex text with multiple safety issues
        complex_text = """
        Dear John Smith,
        
        I'm writing to inform you about your account. Your email john.smith@company.com 
        and phone number (555) 123-4567 are on file. Your SSN 123-45-6789 shows some 
        concerning activity.
        
        This is a serious illegal matter that requires immediate attention.
        I hate to be the bearer of bad news, but this situation is quite harmful.
        """
        
        context = "Customer service communication regarding account security."
        
        # Get comprehensive analysis
        summary = safety_filters.get_safety_summary(complex_text, context)
        
        # Verify comprehensive analysis
        assert not summary["is_safe"]  # Should detect multiple issues
        assert summary["pii_detected"]  # Should detect PII
        assert summary["pii_count"] >= 3  # Email, phone, SSN at minimum
        assert len(summary["violations"]) > 0  # Should have violations
        
        # Check that masked text is different from original
        assert summary["masked_text"] != complex_text
        assert "john.smith@company.com" not in summary["masked_text"]
        assert "(555) 123-4567" not in summary["masked_text"]
        assert "123-45-6789" not in summary["masked_text"]
        
        # Verify violation types
        violation_types = {v["type"] for v in summary["violations"]}
        assert "pii_detected" in violation_types
    
    def test_performance_with_large_text(self):
        """Test performance with larger text inputs"""
        safety_filters = SafetyFilters()
        
        # Create large text (simulate document processing)
        large_text = "This is a normal business document. " * 1000
        large_text += "Contact info: john@company.com and phone 555-123-4567."
        
        # Should handle large text without errors
        violations = safety_filters.check_content_safety(large_text)
        
        # Should still detect PII in large text
        pii_violations = [v for v in violations if v.violation_type == SafetyViolationType.PII_DETECTED]
        assert len(pii_violations) > 0


class TestSafetyFiltersProperties:
    """Property-based tests for safety filters system"""
    
    def setup_method(self):
        """Set up test environment"""
        self.safety_filters = SafetyFilters()
    
    @given(pii_data=text_with_pii())
    @settings(max_examples=15, suppress_health_check=[HealthCheck.too_slow])
    def test_property_17_safety_filter_coverage(self, pii_data):
        """
        Property 17: Safety Filter Coverage
        For any content processed by the system, safety filters should detect 
        and flag PII exposure and potential hallucinations
        
        Feature: enterprise-genai-platform, Property 17: Safety Filter Coverage
        Validates: Requirements 5.5
        """
        text = pii_data['text']
        expected_pii_count = pii_data['expected_pii_types']
        
        # Run comprehensive safety check
        violations = self.safety_filters.check_content_safety(text)
        
        # Should detect PII violations when PII is present
        pii_violations = [v for v in violations if v.violation_type == SafetyViolationType.PII_DETECTED]
        assert len(pii_violations) > 0, f"Failed to detect PII in text: {text[:100]}..."
        
        # Verify PII detection details
        pii_violation = pii_violations[0]
        assert pii_violation.confidence > 0.0
        assert "PII" in pii_violation.description
        assert len(pii_violation.details.get('pii_detections', [])) > 0
        
        # Test PII masking functionality
        masked_text, detections = self.safety_filters.mask_pii_in_text(text)
        assert len(detections) > 0
        assert masked_text != text  # Text should be modified
        
        # Verify specific PII types are detected correctly
        detected_types = {d.pii_type for d in detections}
        if pii_data['contains_email']:
            assert PIIType.EMAIL in detected_types
        if pii_data['contains_phone']:
            assert PIIType.PHONE in detected_types
        if pii_data['contains_ssn']:
            assert PIIType.SSN in detected_types
        
        # Verify safety summary provides comprehensive information
        summary = self.safety_filters.get_safety_summary(text)
        assert not summary['is_safe']  # Should be flagged as unsafe
        assert summary['pii_detected']
        assert summary['pii_count'] > 0
        assert len(summary['violations']) > 0
    
    @given(safe_text=safe_text_data())
    @settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow])
    def test_safe_content_passes_filters(self, safe_text):
        """
        Test that genuinely safe content passes through filters without violations
        """
        # Skip very short or problematic texts
        if len(safe_text.strip()) < 5:
            return
        
        violations = self.safety_filters.check_content_safety(safe_text)
        
        # Safe text should have no violations
        assert isinstance(violations, list)
        
        # Check safety summary
        summary = self.safety_filters.get_safety_summary(safe_text)
        assert isinstance(summary, dict)
        assert 'is_safe' in summary
        assert 'pii_detected' in summary
        assert 'violations' in summary
        
        # For truly safe text, should have no PII
        if not summary['pii_detected']:
            assert summary['pii_count'] == 0
            assert summary['masked_text'] == safe_text
    
    @given(
        response=st.text(min_size=10, max_size=100, alphabet='abcdefghijklmnopqrstuvwxyz '),
        context=st.text(min_size=10, max_size=100, alphabet='abcdefghijklmnopqrstuvwxyz ')
    )
    @settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow])
    def test_hallucination_detection_consistency(self, response, context):
        """
        Test that hallucination detection produces consistent results
        """
        # Test hallucination detection with context
        violations = self.safety_filters.check_content_safety(response, context)
        
        # Should always return a list
        assert isinstance(violations, list)
        
        # Check for hallucination violations
        hallucination_violations = [v for v in violations if v.violation_type == SafetyViolationType.HALLUCINATION_DETECTED]
        
        # If hallucination is detected, should have proper structure
        for violation in hallucination_violations:
            assert 0.0 <= violation.confidence <= 1.0
            assert isinstance(violation.description, str)
            assert isinstance(violation.details, dict)
            assert isinstance(violation.suggested_action, str)
    
    @given(st.lists(text_with_pii(), min_size=2, max_size=5))
    @settings(max_examples=5, suppress_health_check=[HealthCheck.too_slow])
    def test_batch_safety_processing(self, pii_texts):
        """
        Test that safety filters work consistently across multiple texts
        """
        all_violations = []
        all_summaries = []
        
        for pii_data in pii_texts:
            text = pii_data['text']
            
            # Process each text
            violations = self.safety_filters.check_content_safety(text)
            summary = self.safety_filters.get_safety_summary(text)
            
            all_violations.append(violations)
            all_summaries.append(summary)
            
            # Each text with PII should have violations
            assert len(violations) > 0
            assert not summary['is_safe']
            assert summary['pii_detected']
        
        # Verify consistency across batch
        assert len(all_violations) == len(pii_texts)
        assert len(all_summaries) == len(pii_texts)
        
        # All should have detected PII
        assert all(len(violations) > 0 for violations in all_violations)
        assert all(not summary['is_safe'] for summary in all_summaries)
    
    @given(
        text=st.text(min_size=5, max_size=100, alphabet='abcdefghijklmnopqrstuvwxyz '),
        context=st.one_of(st.none(), st.text(min_size=5, max_size=100, alphabet='abcdefghijklmnopqrstuvwxyz '))
    )
    @settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow])
    def test_safety_check_robustness(self, text, context):
        """
        Test that safety checks are robust and don't crash with various inputs
        """
        try:
            # Should not raise exceptions
            violations = self.safety_filters.check_content_safety(text, context)
            summary = self.safety_filters.get_safety_summary(text, context)
            masked_text, detections = self.safety_filters.mask_pii_in_text(text)
            is_safe = self.safety_filters.is_content_safe(text, context)
            
            # Verify return types
            assert isinstance(violations, list)
            assert isinstance(summary, dict)
            assert isinstance(masked_text, str)
            assert isinstance(detections, list)
            assert isinstance(is_safe, bool)
            
            # Verify summary structure
            required_keys = ['is_safe', 'violations', 'pii_detected', 'pii_count', 'masked_text']
            for key in required_keys:
                assert key in summary
            
            # Verify consistency between methods
            has_violations = len(violations) > 0
            assert summary['is_safe'] == (not has_violations)
            assert is_safe == summary['is_safe']
            
        except Exception as e:
            pytest.fail(f"Safety check failed with exception: {e}")


if __name__ == "__main__":
    pytest.main([__file__])