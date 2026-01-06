"""
Property-based tests for Prompt Versioning System

Tests the correctness properties of prompt template storage, versioning,
and usage tracking functionality.
"""
import pytest
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from hypothesis import given, strategies as st, settings, HealthCheck
import json

from app.services.prompt_versioning import PromptVersioningSystem, PromptTemplate


# Test data generators
@st.composite
def prompt_template_data(draw):
    """Generate valid prompt template data"""
    name = draw(st.text(min_size=1, max_size=20, alphabet='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'))
    content = draw(st.text(min_size=5, max_size=100, alphabet='abcdefghijklmnopqrstuvwxyz '))
    variables = draw(st.lists(st.text(min_size=1, max_size=10, alphabet='abcdefghijklmnopqrstuvwxyz'), min_size=0, max_size=3, unique=True))
    description = draw(st.text(max_size=50, alphabet='abcdefghijklmnopqrstuvwxyz '))
    created_by = draw(st.text(min_size=1, max_size=20, alphabet='abcdefghijklmnopqrstuvwxyz'))
    tags = draw(st.lists(st.text(min_size=1, max_size=10, alphabet='abcdefghijklmnopqrstuvwxyz'), min_size=0, max_size=3, unique=True))
    
    return {
        'name': name,
        'content': content,
        'variables': variables,
        'description': description,
        'created_by': created_by,
        'tags': tags
    }


class TestPromptVersioningProperties:
    """Property-based tests for prompt versioning system"""
    
    def setup_method(self):
        """Set up test environment with temporary database"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_prompt_versioning.db"
        self.system = PromptVersioningSystem(str(self.db_path))
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @given(template_data=prompt_template_data())
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_property_14_prompt_versioning_consistency(self, template_data):
        """
        Property 14: Prompt Versioning Consistency
        For any prompt usage, the system should maintain accurate version tracking 
        and allow retrieval of historical prompt versions
        
        Feature: enterprise-genai-platform, Property 14: Prompt Versioning Consistency
        Validates: Requirements 5.1
        """
        # Create a fresh system for each test to ensure isolation
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "isolated_test.db"
        system = PromptVersioningSystem(str(db_path))
        
        try:
            # Create a prompt template
            template = system.create_prompt_template(**template_data)
            
            # Verify template was created with correct data
            assert template.name == template_data['name']
            assert template.content == template_data['content']
            assert template.variables == template_data['variables']
            assert template.description == template_data['description']
            assert template.created_by == template_data['created_by']
            assert template.tags == template_data['tags']
            assert template.is_active is True
            
            # Retrieve the template by name (should get latest version)
            retrieved = system.get_template(template.name)
            assert retrieved is not None
            assert retrieved.template_id == template.template_id
            assert retrieved.name == template.name
            assert retrieved.version == template.version
            assert retrieved.content == template.content
            
            # Retrieve by specific version
            retrieved_by_version = system.get_template(template.name, template.version)
            assert retrieved_by_version is not None
            assert retrieved_by_version.template_id == template.template_id
            
            # Get all versions (should have exactly one)
            versions = system.get_template_versions(template.name)
            assert len(versions) == 1
            assert versions[0].template_id == template.template_id
            
        finally:
            # Clean up
            shutil.rmtree(temp_dir, ignore_errors=True)