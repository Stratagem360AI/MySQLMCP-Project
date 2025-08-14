# ==================================================
# llm_client.py - LLM Integration
# ==================================================
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import logging
import json
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from config import LLMConfig

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

class LLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    @abstractmethod
    async def generate_sql(self, natural_query: str, schema_info: Dict[str, Any]) -> str:
        """Generate SQL from natural language query"""
        pass
    
    @abstractmethod
    async def explain_query(self, sql_query: str, schema_info: Dict[str, Any]) -> str:
        """Explain what a SQL query does in natural language"""
        pass

class OpenAIClient(LLMClient):
    """OpenAI LLM client"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        try:
            import openai
            self.client = openai.AsyncOpenAI(api_key=config.api_key)
        except ImportError:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
    
    async def generate_sql(self, natural_query: str, schema_info: Dict[str, Any]) -> str:
        """Generate SQL from natural language using OpenAI"""
        schema_text = self._format_schema(schema_info)
        
        prompt = f"""
You are a SQL expert. Convert the following natural language query into a MySQL SQL query.

Database Schema:
{schema_text}

Natural Language Query: {natural_query}

Rules:
1. Return ONLY the SQL query, no explanations
2. Use proper MySQL syntax
3. Use backticks for table/column names if needed
4. Be precise and efficient
5. If the query is ambiguous, make reasonable assumptions
6. For aggregations, include appropriate GROUP BY clauses

SQL Query:"""

        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            sql_query = response.choices[0].message.content.strip()
            # Clean up the response (remove markdown formatting if present)
            if sql_query.startswith('```'):
                sql_query = sql_query.split('\n', 1)[1]
            if sql_query.endswith('```'):
                sql_query = sql_query.rsplit('\n', 1)[0]
            
            return sql_query.strip()
            
        except Exception as e:
            logger.error(f"Error generating SQL with OpenAI: {e}")
            return f"-- Error: Could not generate SQL query: {str(e)}"
    
    async def explain_query(self, sql_query: str, schema_info: Dict[str, Any]) -> str:
        """Explain SQL query in natural language"""
        schema_text = self._format_schema(schema_info)
        
        prompt = f"""
Explain this SQL query in simple, natural language:

Database Schema:
{schema_text}

SQL Query: {sql_query}

Provide a clear, concise explanation of what this query does."""

        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.max_tokens,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error explaining query with OpenAI: {e}")
            return f"Error: Could not explain query: {str(e)}"
    
    def _format_schema(self, schema_info: Dict[str, Any]) -> str:
        """Format schema information for the LLM prompt"""
        if "schema" not in schema_info:
            return "No schema information available"
        
        formatted_tables = []
        for table_name, columns in schema_info["schema"].items():
            column_info = []
            for col in columns:
                col_desc = f"  - {col['Field']} ({col['Type']})"
                if col['Key'] == 'PRI':
                    col_desc += " [PRIMARY KEY]"
                if col['Null'] == 'NO':
                    col_desc += " [NOT NULL]"
                column_info.append(col_desc)
            
            formatted_tables.append(f"Table: {table_name}\n" + "\n".join(column_info))
        
        return "\n\n".join(formatted_tables)

class OllamaClient(LLMClient):
    """Ollama local LLM client"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = config.base_url or "http://localhost:11434"
        try:
            import httpx
            self.client = httpx.AsyncClient()
        except ImportError:
            raise ImportError("httpx package not installed. Install with: pip install httpx")
    
    async def generate_sql(self, natural_query: str, schema_info: Dict[str, Any]) -> str:
        """Generate SQL using Ollama"""
        schema_text = self._format_schema(schema_info)
        
        prompt = f"""Convert this natural language query to MySQL SQL:

Schema: {schema_text}
Query: {natural_query}

Return only the SQL query:"""

        try:
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.config.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                return f"-- Error: Ollama request failed with status {response.status_code}"
                
        except Exception as e:
            logger.error(f"Error with Ollama: {e}")
            return f"-- Error: Could not generate SQL query: {str(e)}"
    
    async def explain_query(self, sql_query: str, schema_info: Dict[str, Any]) -> str:
        """Explain query using Ollama"""
        prompt = f"Explain this SQL query in simple terms: {sql_query}"
        
        try:
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.config.model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                return f"Error: Could not explain query"
                
        except Exception as e:
            return f"Error: Could not explain query: {str(e)}"
    
    def _format_schema(self, schema_info: Dict[str, Any]) -> str:
        """Format schema for Ollama prompt"""
        if "schema" not in schema_info:
            return "No schema"
        
        tables = []
        for table_name, columns in schema_info["schema"].items():
            cols = [f"{col['Field']}({col['Type']})" for col in columns]
            tables.append(f"{table_name}({', '.join(cols)})")
        
        return "; ".join(tables)

def create_llm_client(config: LLMConfig) -> LLMClient:
    """Factory function to create appropriate LLM client"""
    if config.provider.lower() == "openai":
        return OpenAIClient(config)
    elif config.provider.lower() == "ollama":
        return OllamaClient(config)
    else:
        raise ValueError(f"Unsupported LLM provider: {config.provider}")
