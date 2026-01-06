"""
Prompt Versioning System for LLMOps Monitoring

This module provides prompt template storage, versioning, and usage tracking
to maintain prompt evolution history and enable prompt management.
"""
import logging
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import sqlite3
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@dataclass
class PromptTemplate:
    """Represents a versioned prompt template"""
    template_id: str
    name: str
    version: str
    content: str
    variables: List[str]
    description: str
    created_at: datetime
    created_by: str
    tags: List[str]
    is_active: bool = True

@dataclass
class PromptUsage:
    """Tracks usage of a specific prompt template"""
    usage_id: str
    template_id: str
    version: str
    input_variables: Dict[str, Any]
    rendered_prompt: str
    response: Optional[str]
    execution_time: float
    token_count: int
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None

class PromptVersioningSystem:
    """Manages prompt templates, versions, and usage tracking"""
    
    def __init__(self, db_path: str = "data/prompt_versioning.db"):
        """
        Initialize the prompt versioning system
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database with required tables"""
        with self._get_db_connection() as conn:
            # Create prompt templates table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prompt_templates (
                    template_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    content TEXT NOT NULL,
                    variables TEXT NOT NULL,  -- JSON array
                    description TEXT,
                    created_at TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    tags TEXT NOT NULL,  -- JSON array
                    is_active BOOLEAN DEFAULT TRUE,
                    UNIQUE(name, version)
                )
            """)
            
            # Create prompt usage table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prompt_usage (
                    usage_id TEXT PRIMARY KEY,
                    template_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    input_variables TEXT NOT NULL,  -- JSON object
                    rendered_prompt TEXT NOT NULL,
                    response TEXT,
                    execution_time REAL NOT NULL,
                    token_count INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    error_message TEXT,
                    FOREIGN KEY (template_id) REFERENCES prompt_templates (template_id)
                )
            """)
            
            # Create indexes for better query performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_template_name ON prompt_templates (name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_template_active ON prompt_templates (is_active)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_template ON prompt_usage (template_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON prompt_usage (timestamp)")
            
            conn.commit()
    
    @contextmanager
    def _get_db_connection(self):
        """Get a database connection with proper error handling"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()
    
    def create_prompt_template(
        self,
        name: str,
        content: str,
        variables: List[str],
        description: str = "",
        created_by: str = "system",
        tags: List[str] = None,
        version: str = None
    ) -> PromptTemplate:
        """
        Create a new prompt template
        
        Args:
            name: Template name
            content: Template content with variable placeholders
            variables: List of variable names used in template
            description: Template description
            created_by: Creator identifier
            tags: List of tags for categorization
            version: Version string (auto-generated if not provided)
            
        Returns:
            Created PromptTemplate instance
        """
        if tags is None:
            tags = []
        
        # Generate version if not provided
        if version is None:
            existing_versions = self.get_template_versions(name)
            version = f"v{len(existing_versions) + 1}.0"
        
        # Generate template ID
        template_id = self._generate_template_id(name, version)
        
        template = PromptTemplate(
            template_id=template_id,
            name=name,
            version=version,
            content=content,
            variables=variables,
            description=description,
            created_at=datetime.now(),
            created_by=created_by,
            tags=tags,
            is_active=True
        )
        
        # Store in database
        with self._get_db_connection() as conn:
            conn.execute("""
                INSERT INTO prompt_templates 
                (template_id, name, version, content, variables, description, 
                 created_at, created_by, tags, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                template.template_id,
                template.name,
                template.version,
                template.content,
                json.dumps(template.variables),
                template.description,
                template.created_at.isoformat(),
                template.created_by,
                json.dumps(template.tags),
                template.is_active
            ))
            conn.commit()
        
        logger.info(f"Created prompt template: {name} v{version}")
        return template
    
    def get_template(self, name: str, version: str = None) -> Optional[PromptTemplate]:
        """
        Get a specific prompt template
        
        Args:
            name: Template name
            version: Template version (latest active if not specified)
            
        Returns:
            PromptTemplate instance or None if not found
        """
        with self._get_db_connection() as conn:
            if version:
                cursor = conn.execute("""
                    SELECT * FROM prompt_templates 
                    WHERE name = ? AND version = ?
                """, (name, version))
            else:
                # Get latest active version
                cursor = conn.execute("""
                    SELECT * FROM prompt_templates 
                    WHERE name = ? AND is_active = TRUE
                    ORDER BY created_at DESC LIMIT 1
                """, (name,))
            
            row = cursor.fetchone()
            if row:
                return self._row_to_template(row)
            return None
    
    def get_template_versions(self, name: str) -> List[PromptTemplate]:
        """
        Get all versions of a template
        
        Args:
            name: Template name
            
        Returns:
            List of PromptTemplate instances ordered by creation date
        """
        with self._get_db_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM prompt_templates 
                WHERE name = ? 
                ORDER BY created_at DESC
            """, (name,))
            
            return [self._row_to_template(row) for row in cursor.fetchall()]
    
    def list_templates(self, active_only: bool = True, tags: List[str] = None) -> List[PromptTemplate]:
        """
        List all prompt templates
        
        Args:
            active_only: Only return active templates
            tags: Filter by tags (any of the provided tags)
            
        Returns:
            List of PromptTemplate instances
        """
        query = "SELECT * FROM prompt_templates WHERE 1=1"
        params = []
        
        if active_only:
            query += " AND is_active = TRUE"
        
        if tags:
            # Filter by tags (this is a simple implementation)
            tag_conditions = []
            for tag in tags:
                tag_conditions.append("tags LIKE ?")
                params.append(f"%{tag}%")
            query += f" AND ({' OR '.join(tag_conditions)})"
        
        query += " ORDER BY name, created_at DESC"
        
        with self._get_db_connection() as conn:
            cursor = conn.execute(query, params)
            return [self._row_to_template(row) for row in cursor.fetchall()]
    
    def deactivate_template(self, template_id: str) -> bool:
        """
        Deactivate a prompt template
        
        Args:
            template_id: Template ID to deactivate
            
        Returns:
            True if successful, False otherwise
        """
        with self._get_db_connection() as conn:
            cursor = conn.execute("""
                UPDATE prompt_templates 
                SET is_active = FALSE 
                WHERE template_id = ?
            """, (template_id,))
            conn.commit()
            
            success = cursor.rowcount > 0
            if success:
                logger.info(f"Deactivated prompt template: {template_id}")
            return success
    
    def render_template(self, template: PromptTemplate, variables: Dict[str, Any]) -> str:
        """
        Render a prompt template with provided variables
        
        Args:
            template: PromptTemplate instance
            variables: Dictionary of variable values
            
        Returns:
            Rendered prompt string
        """
        try:
            # Simple string formatting - could be enhanced with Jinja2
            rendered = template.content
            for var_name, var_value in variables.items():
                placeholder = f"{{{var_name}}}"
                rendered = rendered.replace(placeholder, str(var_value))
            
            return rendered
        except Exception as e:
            logger.error(f"Error rendering template {template.template_id}: {e}")
            raise
    
    def log_usage(
        self,
        template_id: str,
        version: str,
        input_variables: Dict[str, Any],
        rendered_prompt: str,
        response: Optional[str],
        execution_time: float,
        token_count: int,
        success: bool,
        error_message: Optional[str] = None
    ) -> str:
        """
        Log usage of a prompt template
        
        Args:
            template_id: Template ID
            version: Template version
            input_variables: Variables used for rendering
            rendered_prompt: Final rendered prompt
            response: LLM response
            execution_time: Execution time in seconds
            token_count: Total token count
            success: Whether execution was successful
            error_message: Error message if failed
            
        Returns:
            Usage ID
        """
        usage_id = self._generate_usage_id()
        
        usage = PromptUsage(
            usage_id=usage_id,
            template_id=template_id,
            version=version,
            input_variables=input_variables,
            rendered_prompt=rendered_prompt,
            response=response,
            execution_time=execution_time,
            token_count=token_count,
            timestamp=datetime.now(),
            success=success,
            error_message=error_message
        )
        
        with self._get_db_connection() as conn:
            conn.execute("""
                INSERT INTO prompt_usage 
                (usage_id, template_id, version, input_variables, rendered_prompt,
                 response, execution_time, token_count, timestamp, success, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                usage.usage_id,
                usage.template_id,
                usage.version,
                json.dumps(usage.input_variables),
                usage.rendered_prompt,
                usage.response,
                usage.execution_time,
                usage.token_count,
                usage.timestamp.isoformat(),
                usage.success,
                usage.error_message
            ))
            conn.commit()
        
        return usage_id
    
    def get_usage_stats(
        self,
        template_name: str = None,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> Dict[str, Any]:
        """
        Get usage statistics for prompt templates
        
        Args:
            template_name: Filter by template name
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            Dictionary with usage statistics
        """
        query = """
            SELECT 
                pt.name,
                pt.version,
                COUNT(*) as usage_count,
                AVG(pu.execution_time) as avg_execution_time,
                SUM(pu.token_count) as total_tokens,
                AVG(pu.token_count) as avg_tokens,
                SUM(CASE WHEN pu.success THEN 1 ELSE 0 END) as success_count,
                COUNT(*) - SUM(CASE WHEN pu.success THEN 1 ELSE 0 END) as error_count
            FROM prompt_usage pu
            JOIN prompt_templates pt ON pu.template_id = pt.template_id
            WHERE 1=1
        """
        params = []
        
        if template_name:
            query += " AND pt.name = ?"
            params.append(template_name)
        
        if start_date:
            query += " AND pu.timestamp >= ?"
            params.append(start_date.isoformat())
        
        if end_date:
            query += " AND pu.timestamp <= ?"
            params.append(end_date.isoformat())
        
        query += " GROUP BY pt.name, pt.version ORDER BY usage_count DESC"
        
        with self._get_db_connection() as conn:
            cursor = conn.execute(query, params)
            results = []
            for row in cursor.fetchall():
                results.append({
                    "name": row["name"],
                    "version": row["version"],
                    "usage_count": row["usage_count"],
                    "avg_execution_time": row["avg_execution_time"],
                    "total_tokens": row["total_tokens"],
                    "avg_tokens": row["avg_tokens"],
                    "success_count": row["success_count"],
                    "error_count": row["error_count"],
                    "success_rate": row["success_count"] / row["usage_count"] if row["usage_count"] > 0 else 0
                })
            
            return {
                "templates": results,
                "total_templates": len(results),
                "total_usage": sum(r["usage_count"] for r in results),
                "total_tokens": sum(r["total_tokens"] for r in results)
            }
    
    def _generate_template_id(self, name: str, version: str) -> str:
        """Generate a unique template ID"""
        content = f"{name}:{version}:{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _generate_usage_id(self) -> str:
        """Generate a unique usage ID"""
        content = f"usage:{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _row_to_template(self, row) -> PromptTemplate:
        """Convert database row to PromptTemplate instance"""
        return PromptTemplate(
            template_id=row["template_id"],
            name=row["name"],
            version=row["version"],
            content=row["content"],
            variables=json.loads(row["variables"]),
            description=row["description"],
            created_at=datetime.fromisoformat(row["created_at"]),
            created_by=row["created_by"],
            tags=json.loads(row["tags"]),
            is_active=bool(row["is_active"])
        )