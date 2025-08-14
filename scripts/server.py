# ==================================================
# server.py - MCP Server Implementation
# ==================================================

import asyncio
import json
from config import DatabaseConfig, LLMConfig
from database import MySQLConnection
from llm_client import create_llm_client
from nlp import EnhancedNaturalLanguageProcessor
import logging
import types
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
        notification_options = types.NoneType  # Example: completion notifications
        experimental_capabilities = {"myExperimentalFeature": True}

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="mysql-mcp-server",
                    server_version="2.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=notification_options,
                        experimental_capabilities=experimental_capabilities)
                )
            )
    
    def cleanup(self):
        """Cleanup resources"""
        self.db_connection.disconnect()