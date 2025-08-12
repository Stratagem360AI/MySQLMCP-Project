# ==================================================
# nlp.py - Natural Language Processing with LLM
# ==================================================
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import logging
from typing import Any, Dict, List, Optional
from scripts.llm_client import LLMClient
from scripts.database import MySQLConnection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mysql_mcp_server.log')
    ]
)

logger = logging.getLogger(__name__)

class EnhancedNaturalLanguageProcessor:
    """Enhanced NLP processor using LLM for query generation"""
    
    def __init__(self, db_connection: MySQLConnection, llm_client: LLMClient):
        self.db = db_connection
        self.llm = llm_client
        self.schema_info = None
        self._load_schema_info()
    
    def _load_schema_info(self):
        """Load database schema information"""
        try:
            self.schema_info = self.db.get_database_schema()
            logger.info(f"Loaded schema for {len(self.schema_info.get('tables', []))} tables")
        except Exception as e:
            logger.error(f"Could not load schema info: {e}")
            self.schema_info = {"schema": {}, "tables": []}
    
    async def process_natural_query(self, natural_query: str) -> Dict[str, Any]:
        """Convert natural language query to SQL using LLM"""
        try:
            # Handle simple commands first
            simple_result = self._handle_simple_queries(natural_query)
            if simple_result:
                return {
                    "sql_query": simple_result,
                    "method": "pattern_matching",
                    "confidence": "high"
                }
            
            # Use LLM for complex queries
            sql_query = await self.llm.generate_sql(natural_query, self.schema_info)
            
            # Validate the generated SQL
            validation_result = self._validate_sql_query(sql_query)
            
            return {
                "sql_query": sql_query,
                "method": "llm_generated",
                "confidence": validation_result["confidence"],
                "warnings": validation_result.get("warnings", [])
            }
            
        except Exception as e:
            logger.error(f"Error processing natural query: {e}")
            return {
                "sql_query": f"-- Error: {str(e)}",
                "method": "error",
                "confidence": "none"
            }
    
    async def explain_sql_result(self, sql_query: str, result: Dict[str, Any]) -> str:
        """Explain SQL query and its results in natural language"""
        try:
            explanation = await self.llm.explain_query(sql_query, self.schema_info)
            
            if "data" in result:
                row_count = len(result["data"])
                explanation += f"\n\nThis query returned {row_count} row(s)."
                
                if row_count > 0 and row_count <= 5:
                    explanation += " Here are the results:"
                    for row in result["data"]:
                        explanation += f"\n  - {dict(row)}"
            
            return explanation
            
        except Exception as e:
            return f"Could not generate explanation: {str(e)}"
    
    def _handle_simple_queries(self, query: str) -> Optional[str]:
        """Handle simple queries with pattern matching"""
        query_lower = query.lower().strip()
        
        # Show tables
        if any(phrase in query_lower for phrase in ["show tables", "list tables", "what tables"]):
            return "SHOW TABLES"
        
        # Show databases
        if any(phrase in query_lower for phrase in ["show databases", "list databases"]):
            return "SHOW DATABASES"
        
        # Describe table
        if query_lower.startswith("describe") or "structure of" in query_lower:
            for table_name in self.schema_info.get("tables", []):
                if table_name.lower() in query_lower:
                    return f"DESCRIBE `{table_name}`"
        
        return None
    
    def _validate_sql_query(self, sql_query: str) -> Dict[str, Any]:
        """Validate generated SQL query"""
        warnings = []
        confidence = "medium"
        
        # Check if it's an error message
        if sql_query.startswith("--"):
            return {"confidence": "none", "warnings": ["Query generation failed"]}
        
        # Basic SQL validation
        sql_lower = sql_query.lower().strip()
        
        # Check for dangerous operations
        dangerous_keywords = ["drop", "delete", "truncate", "alter", "create"]
        if any(keyword in sql_lower for keyword in dangerous_keywords):
            warnings.append("Query contains potentially destructive operations")
            confidence = "low"
        
        # Check for table existence
        mentioned_tables = []
        for table in self.schema_info.get("tables", []):
            if table.lower() in sql_lower:
                mentioned_tables.append(table)
        
        if not mentioned_tables and "from" in sql_lower:
            warnings.append("Query references tables not found in schema")
            confidence = "low"
        
        # Check basic SQL syntax
        if not any(sql_lower.startswith(cmd) for cmd in ["select", "show", "describe", "explain"]):
            warnings.append("Query doesn't start with expected SQL command")
        
        return {"confidence": confidence, "warnings": warnings}