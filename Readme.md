# **Objectives and Goals of this project**
To create a simple Model Context Protocol (MCP) server in Python that help you interact with your MySQL database using natural language or direct SQL queries.

# **🏗️Modular Architecture**
The code is now organized into logical modules:

***config.py*** - Configuration management with environment variable support
***database.py*** - MySQL connection and operations with caching
***llm_client.py*** - LLM integration supporting OpenAI and Ollama
***nlp.py*** - Enhanced natural language processing using LLMs
***server.py*** - MCP server implementation with advanced tools
***main.py*** - Application entry point with proper error handling

# **🤖 LLM Integration Features**
Multiple LLM Providers:

OpenAI (GPT-3.5/GPT-4)
Ollama (Local models like Llama2, Code Llama)
Extensible architecture for additional providers

# **Natural Language Processing:**

Context-aware SQL generation using database schema
Query validation and confidence scoring
Natural language explanations of results
Query optimization suggestions

# **🛠️ Available SQL operations**

execute_sql - Direct SQL execution with optional explanations
ask_database - Natural language database queries with AI
get_schema_info - Comprehensive schema exploration
optimize_query - AI-powered query optimization

**Security:**

SQL injection protection
Dangerous operation warnings
Input validation

**📦 Easy Setup**

Docker Compose for development environment
Sample database schema with test data
Environment variable templates
Requirements file for dependencies
Usage examples and testing code

# **Steps for building the MCP Implementation Project**
In VS Code setup Python project structure as below 

```
-- data
|
-- notebooks
|
-- scripts
|
-- results
|
-- Readme.md
```
On the VS Code Terminal, create the python virtual environment
***Note*** Assuming that python3 is installed
- Check for Python installation by running the following command

```
python --version
pip --version
```
- Now, install virtual environment for python development
A Python virtual environment is an isolated, self-contained directory that houses a specific Python interpreter and its own set of installed packages (libraries). It functions as a separate container for each Python project, preventing conflicts and ensuring consistent project dependencies.

``` 
python -m pip install virtualenv

python -m venv ./venv/
source ./venv/bin/activate
```
The above commands will install the libraries for setting up virtual environment and then create a .venv in the MCP implementation project structure (root folder). Pointing to the source location will activate the virtual environment for further use.

- The ***config.py*** script
This python script will isolate the configurations required for this project. We need a Database configuration and LLM Configuration. 
It will look for these configuration setup as environment variables. Optionally, these configurations can be provided in the config.py script itself.
***Note*** It is not a safe practice to have sensitive configurations like the openai API Key or DB passwords in a plain text file or a script like this. Setting env variable is comparatively safer option.
```

# ==================================================
# config.py - Configuration Management
# ==================================================

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 3306
    database: str = "test_db"
    username: str = "root"
    password: str = "password"
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        """Load configuration from environment variables"""
        return cls(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', '3306')),
            database=os.getenv('DB_NAME', 'test_db'),
            username=os.getenv('DB_USER', 'root'),
            password=os.getenv('DB_PASSWORD', 'password')
        )

@dataclass
class LLMConfig:
    """LLM configuration"""
    provider: str = "openai"  # openai, anthropic, ollama
    api_key: Optional[str] = None
    model: str = "gpt-3.5-turbo"
    base_url: Optional[str] = None  # For local models like Ollama
    max_tokens: int = 1000
    temperature: float = 0.1
    
    @classmethod
    def from_env(cls) -> 'LLMConfig':
        """Load LLM configuration from environment variables"""
        return cls(
            provider=os.getenv('LLM_PROVIDER', 'openai'),
            api_key=os.getenv('LLM_API_KEY'),
            model=os.getenv('LLM_MODEL', 'gpt-3.5-turbo'),
            base_url=os.getenv('LLM_BASE_URL'),
            max_tokens=int(os.getenv('LLM_MAX_TOKENS', '1000')),
            temperature=float(os.getenv('LLM_TEMPERATURE', '0.1'))
        )
```
The above code is self-explainatory. So I will not go in details.

- The ***database.py*** script.
This script will require a ***mysql-connector-python*** package. To install this,

```
pip install mysql-connector-python
```
Once this package is installed all the required dependencies will be resolved and this script can be executed successfully. 

```
# ==================================================
# database.py - Database Connection and Operations
# ==================================================
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import logging
from typing import Any, Dict, List, Optional
import mysql.connector
from mysql.connector import Error

logger = logging.getLogger(__name__)

from scripts.config import DatabaseConfig

class MySQLConnection:
    """MySQL database connection handler"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.connection = None
        self._schema_cache = {}
    
    def connect(self) -> bool:
        """Establish database connection"""
        try:
            self.connection = mysql.connector.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.username,
                password=self.config.password,
                autocommit=False
            )
            logger.info(f"Connected to MySQL database: {self.config.database}")
            self._load_schema_cache()
            return True
        except Error as e:
            logger.error(f"Error connecting to MySQL: {e}")
            return False
    
    def disconnect(self):
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("MySQL connection closed")
    
    def is_connected(self) -> bool:
        """Check if database is connected"""
        return self.connection and self.connection.is_connected()
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> Dict[str, Any]:
        """Execute SQL query and return results"""
        if not self.is_connected():
            return {"error": "Database not connected"}
        
        try:
            cursor = self.connection.cursor(dictionary=True, buffered=True)
            cursor.execute(query, params or ())
            
            # Handle different query types
            if query.strip().upper().startswith(('SELECT', 'SHOW', 'DESCRIBE', 'EXPLAIN')):
                results = cursor.fetchall()
                return {
                    "data": results, 
                    "row_count": len(results),
                    "columns": [desc[0] for desc in cursor.description] if cursor.description else []
                }
            else:
                self.connection.commit()
                return {"message": f"Query executed successfully. Affected rows: {cursor.rowcount}"}
                
        except Error as e:
            logger.error(f"Database error: {e}")
            return {"error": str(e)}
        finally:
            if 'cursor' in locals():
                cursor.close()
    
    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get table schema information"""
        if table_name in self._schema_cache:
            return {"data": self._schema_cache[table_name], "cached": True}
        
        query = "DESCRIBE `{}`".format(table_name.replace('`', '``'))
        result = self.execute_query(query)
        
        if "data" in result:
            self._schema_cache[table_name] = result["data"]
        
        return result
    
    def list_tables(self) -> Dict[str, Any]:
        """List all tables in the database"""
        return self.execute_query("SHOW TABLES")
    
    def get_database_schema(self) -> Dict[str, Any]:
        """Get complete database schema"""
        schema = {}
        tables_result = self.list_tables()
        
        if "data" in tables_result:
            for table_row in tables_result["data"]:
                table_name = list(table_row.values())[0]
                schema_result = self.get_table_schema(table_name)
                if "data" in schema_result:
                    schema[table_name] = schema_result["data"]
        
        return {"schema": schema, "tables": list(schema.keys())}
    
    def _load_schema_cache(self):
        """Load and cache database schema"""
        try:
            schema_info = self.get_database_schema()
            if "schema" in schema_info:
                self._schema_cache.update(schema_info["schema"])
            logger.info(f"Loaded schema for {len(self._schema_cache)} tables")
        except Exception as e:
            logger.warning(f"Could not load schema cache: {e}")
```
The above script initializes the db config class based on the configurations available from the config.py script which is used to import the DatabaseConfig class.
Once the class is initialized it will establish DB Connection. It will report any connection issues through the logger info or warning in the output window or terminal based on how the project is executed.

Other SQL operations like execute query. The input to this function is a query that is converted from natural language to SQL syntax using LLM. This is discussed further. 
This function will execute the query and display results. Similarly, there are other operations to describe table, database schema etc.

- The ***llm_client.py*** script.
Ideally, this script would not have exist or would have been optional in a traditional client/ server implementation. Traditionally, the client was known to send a syntactically correct SQL statement to the server for execution. However, the beauty of LLM integration and this script is that the client interaction with the MCP server can be in natural language and the MCP server then take advantage of LLM integration through this script to utilize either openai API or local Ollama model to translate natural language to a SQL syntax. 

```
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
from scripts.config import LLMConfig

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
```
The above script defines a factory function to create appropriate LLM client based on the configuration provided/ made available through config.py or test_example.py (discussed further).
Based on the established factory function, configurations are made available to respective class.
The LLM_Client class definition has two abstract method to generate a SQL query from natural language and expalnation of the SQL query.
Both the abstract methods are implemented within the OpenAI class or Ollama class depending on what is configured.
The implementation for both the LLM model is same. 
For the openai class, once the configurations are initialized, the prompt is set to either generate the SQL from natural language or to explain the SQL query. 
Both database schema and natural language query is passed as context to the prompt. Along with this, the role definition is required to set the context. In this case,
***You are a SQL expert. Convert the following natural language query into a MySQL SQL query.*** is the role definition.
There are Rules defined in the prompt to provide guardrails to govern the SQL transformation. This will provide the necessary content, prompt, role, user and context for the openai API to transform the natural language request into a SQL Query. 

- The ***nlp.py*** script
This is a critical script to further provide a wrapper for LLM_Client script. This script also governs what type of questions/ request the client is making and filters out sensitive statements like DROP, DELETE etc. It also ensure the requested table exists to avoid calling LLM APIs and incur cost. It ensures security by not allowing SQL injection prevention.

```
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
```
- The ***server.py*** script
Resolve all the dependencies like installation of mcp, typing, asyncio packages by running the below command inside the virtual environment.

```
pip install mcp
pip install asyncio
pip install typing
```
Try running the followin script.
```
# ==================================================
# server.py - MCP Server Implementation
# ==================================================

import asyncio
import json
from scripts.config import DatabaseConfig, LLMConfig
from scripts.database import MySQLConnection
from scripts.llm_client import create_llm_client
from scripts.nlp import EnhancedNaturalLanguageProcessor
import logging
from typing import Dict, Any
from mcp import ClientSession, StdioServerParameters
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest, CallToolResult, ListToolsRequest, ListToolsResult,
    Tool, TextContent, GetPromptRequest, GetPromptResult,
    ListPromptsRequest, ListPromptsResult, Prompt, PromptMessage, PromptArgument
)

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

class MySQLMCPServer:
    """Enhanced MCP Server for MySQL database operations with LLM integration"""
    
    def __init__(self, db_config: DatabaseConfig, llm_config: LLMConfig):
        self.db_config = db_config
        self.llm_config = llm_config
        self.db_connection = MySQLConnection(db_config)
        self.llm_client = None
        self.nlp_processor = None
        self.server = Server("mysql-mcp-server")
        self._setup_handlers()
    
    async def initialize(self):
        """Initialize database connection and LLM client"""
        # Connect to database
        if not self.db_connection.connect():
            raise Exception("Failed to connect to database")
        
        # Initialize LLM client
        try:
            self.llm_client = create_llm_client(self.llm_config)
            logger.info(f"Initialized {self.llm_config.provider} LLM client")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            raise
        
        # Initialize NLP processor
        self.nlp_processor = EnhancedNaturalLanguageProcessor(
            self.db_connection, self.llm_client
        )
    
    def _setup_handlers(self):
        """Setup MCP server handlers"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> ListToolsResult:
            """List available tools"""
            return ListToolsResult(
                tools=[
                    Tool(
                        name="execute_sql",
                        description="Execute SQL query directly on MySQL database",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "SQL query to execute"
                                },
                                "explain": {
                                    "type": "boolean",
                                    "description": "Whether to explain the query results",
                                    "default": False
                                }
                            },
                            "required": ["query"]
                        }
                    ),
                    Tool(
                        name="ask_database",
                        description="Ask questions about the database in natural language using AI",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "question": {
                                    "type": "string",
                                    "description": "Natural language question about the data"
                                },
                                "explain_results": {
                                    "type": "boolean",
                                    "description": "Whether to provide natural language explanation of results",
                                    "default": True
                                }
                            },
                            "required": ["question"]
                        }
                    ),
                    Tool(
                        name="get_schema_info",
                        description="Get database schema and structure information",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "table_name": {
                                    "type": "string",
                                    "description": "Specific table name (optional)"
                                },
                                "detailed": {
                                    "type": "boolean",
                                    "description": "Include detailed column information",
                                    "default": True
                                }
                            }
                        }
                    ),
                    Tool(
                        name="optimize_query",
                        description="Get suggestions to optimize a SQL query using AI",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "SQL query to optimize"
                                }
                            },
                            "required": ["query"]
                        }
                    )
                ]
            )
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> CallToolResult:
            """Handle tool calls"""
            
            if name == "execute_sql":
                query = arguments.get("query", "")
                explain = arguments.get("explain", False)
                
                result = self.db_connection.execute_query(query)
                
                response = {
                    "sql_query": query,
                    "result": result
                }
                
                if explain and "data" in result:
                    explanation = await self.nlp_processor.explain_sql_result(query, result)
                    response["explanation"] = explanation
                
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=json.dumps(response, indent=2, default=str)
                    )]
                )
            
            elif name == "ask_database":
                question = arguments.get("question", "")
                explain_results = arguments.get("explain_results", True)
                
                # Generate SQL from natural language
                query_info = await self.nlp_processor.process_natural_query(question)
                sql_query = query_info["sql_query"]
                
                # Execute the generated query
                result = self.db_connection.execute_query(sql_query)
                
                response = {
                    "natural_question": question,
                    "generated_sql": sql_query,
                    "generation_method": query_info.get("method", "unknown"),
                    "confidence": query_info.get("confidence", "medium"),
                    "result": result
                }
                
                if query_info.get("warnings"):
                    response["warnings"] = query_info["warnings"]
                
                # Add explanation if requested and successful
                if explain_results and "data" in result:
                    explanation = await self.nlp_processor.explain_sql_result(sql_query, result)
                    response["explanation"] = explanation
                
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=json.dumps(response, indent=2, default=str)
                    )]
                )
            
            elif name == "get_schema_info":
                table_name = arguments.get("table_name")
                detailed = arguments.get("detailed", True)
                
                if table_name:
                    result = self.db_connection.get_table_schema(table_name)
                else:
                    result = self.db_connection.get_database_schema()
                
                if not detailed and "schema" in result:
                    # Simplify the output
                    simplified = {}
                    for table, columns in result["schema"].items():
                        simplified[table] = [col["Field"] for col in columns]
                    result = {"tables": simplified}
                
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=json.dumps(result, indent=2, default=str)
                    )]
                )
            
            elif name == "optimize_query":
                query = arguments.get("query", "")
                
                # This would ideally use the LLM to suggest optimizations
                optimization_prompt = f"Suggest optimizations for this MySQL query: {query}"
                
                try:
                    suggestions = await self.llm_client.explain_query(
                        f"EXPLAIN {query}", 
                        self.nlp_processor.schema_info
                    )
                    
                    response = {
                        "original_query": query,
                        "optimization_suggestions": suggestions
                    }
                except Exception as e:
                    response = {
                        "original_query": query,
                        "error": f"Could not generate optimizations: {str(e)}"
                    }
                
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=json.dumps(response, indent=2, default=str)
                    )]
                )
            
            else:
                raise ValueError(f"Unknown tool: {name}")
        
        @self.server.list_prompts()
        async def handle_list_prompts() -> ListPromptsResult:
            """List available prompts"""
            return ListPromptsResult(
                prompts=[
                    Prompt(
                        name="database_assistant",
                        description="Get help with database operations and queries",
                        arguments=[
                            PromptArgument(
                                name="context",
                                description="Context or specific area you need help with",
                                required=False
                            )
                        ]
                    ),
                    Prompt(
                        name="schema_explorer",
                        description="Explore and understand the database schema",
                        arguments=[]
                    )
                ]
            )
        
        @self.server.get_prompt()
        async def handle_get_prompt(name: str, arguments: dict) -> GetPromptResult:
            """Handle prompt requests"""
            
            if name == "database_assistant":
                context = arguments.get("context", "general help")
                
                schema_info = self.db_connection.get_database_schema()
                tables_list = ", ".join(schema_info.get("tables", []))
                
                help_content = f"""
# MySQL Database Assistant

I can help you interact with your MySQL database using natural language or direct SQL queries.

## Available Tables:
{tables_list}

## What I can help you with:

### Natural Language Queries:
- "Show me all customers from New York"
- "Count how many orders were placed last month"
- "Find the top 5 products by sales"
- "List all employees in the marketing department"

### Direct SQL Operations:
- Execute any SQL query safely
- Get explanations of query results
- Optimize existing queries
- Explore database schema

### Database Schema:
- View table structures
- Understand relationships
- Get column information

## Context: {context}

What would you like to know about your database?
"""
                
                return GetPromptResult(
                    description=f"Database assistant for: {context}",
                    messages=[
                        PromptMessage(
                            role="assistant",
                            content=TextContent(type="text", text=help_content)
                        )
                    ]
                )
            
            elif name == "schema_explorer":
                schema_info = self.db_connection.get_database_schema()
                
                schema_content = "# Database Schema Explorer\n\n"
                
                if "schema" in schema_info:
                    for table_name, columns in schema_info["schema"].items():
                        schema_content += f"## Table: {table_name}\n\n"
                        for col in columns:
                            key_info = ""
                            if col['Key'] == 'PRI':
                                key_info = " 🔑 PRIMARY KEY"
                            elif col['Key'] == 'MUL':
                                key_info = " 🔗 INDEXED"
                            
                            null_info = "" if col['Null'] == 'YES' else " ❗ NOT NULL"
                            
                            schema_content += f"- **{col['Field']}** ({col['Type']}){key_info}{null_info}\n"
                        
                        schema_content += "\n"
                
                schema_content += "\nYou can ask questions about any of these tables or their relationships!"
                
                return GetPromptResult(
                    description="Database schema exploration",
                    messages=[
                        PromptMessage(
                            role="assistant",
                            content=TextContent(type="text", text=schema_content)
                        )
                    ]
                )
    
    async def start(self):
        """Start the MCP server"""
        await self.initialize()
        
        # Start the server
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="mysql-mcp-server",
                    server_version="2.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=None,
                        experimental_capabilities=None
                    )
                )
            )
    
    def cleanup(self):
        """Cleanup resources"""
        self.db_connection.disconnect()
```
This script is the core of this project. The MCP Server. It is actually a wrapper for all the other scripts. It contains the initializers and handler functions to manage the client request.
The MCP Server start function will initialize the server and wait for read/ write stream for client request. 
The get prompt function handles the client request and shows the help-content to guide the client to formulate a request that can be successfully responded back. 

- The ***main.py** script
This is the main entry point to initialize the MySQLMCPServer function of the MCP Server script.
```
# ==================================================
# main.py - Main Application Entry Point
# ==================================================

import logging
import sys
from pathlib import Path

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

async def main():
    """Main function to run the enhanced MCP server"""
    
    try:
        # Load configurations
        db_config = DatabaseConfig.from_env()
        llm_config = LLMConfig.from_env()
        
        # Validate required configurations
        if not llm_config.api_key and llm_config.provider == "openai":
            logger.warning("OpenAI API key not provided. Set LLM_API_KEY environment variable.")
            # You might want to fall back to a simpler implementation or exit
        
        logger.info(f"Starting MySQL MCP Server with {llm_config.provider} LLM")
        logger.info(f"Database: {db_config.host}:{db_config.port}/{db_config.database}")
        
        # Create and start the server
        server = MySQLMCPServer(db_config, llm_config)
        
        await server.start()
        
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if 'server' in locals():
            server.cleanup()
```
