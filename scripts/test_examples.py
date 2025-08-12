# ==================================================
# Usage Examples and Testing
# ==================================================

# test_examples.py - Example usage and testing
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import asyncio
import logging
from config import DatabaseConfig, LLMConfig
from scripts.server import MySQLMCPServer

from scripts.main import main

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

async def test_server():
    # Test configuration
    db_config = DatabaseConfig(
        host="localhost",
        database="test_db",
        username="root",
        password="password"
    )
    
    llm_config = LLMConfig(
        provider="openai",
        api_key="API_KEY",
        model="gpt-3.5-turbo"
    )
    
    # Create server instance
    server = MySQLMCPServer(db_config, llm_config)
    await server.initialize()
     #await asyncio.sleep(1)
    print("Async task finished")


    # Test natural language processing
    nlp = server.nlp_processor
    
    # Test queries
    test_queries = [
        "Show me all users",
        "How many products do we have?",
        "List all completed orders",
        "Find the most expensive product",
        "Show users who have placed orders"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = await nlp.process_natural_query(query)
        print(f"SQL: {result['sql_query']}")
        
        # Execute the query
        db_result = server.db_connection.execute_query(result['sql_query'])
        print(f"Result: {db_result}")
    
    server.cleanup()

# If an event loop is already running (e.g., in Jupyter)
#    if __name__ == "__main__":
#        loop = asyncio.get_event_loop()
#    if loop.is_running():
#        loop.run_until_complete(test_server())
#   else:
#        asyncio.run(test_server())

if __name__ == "__main__":
    asyncio.run(test_server())
