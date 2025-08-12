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