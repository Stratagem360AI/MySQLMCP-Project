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

from config import DatabaseConfig

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
