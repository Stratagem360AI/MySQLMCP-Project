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

# **MCP Implementation Project**
In VS Code setup Python project structure as below 

```
|-- data
|-- notebooks
    |-- "all python notebooks"
|-- scripts
    |-- init.sql
    |-- .env
    |-- docker-compose.yml
    |-- requirements.txt
    |-- "all .py files"
    |-- test_examples.py
    |-- ollama_test.py
|-- results
|-- Readme.md
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

- The ***database.py*** script.
This script will require a ***mysql-connector-python*** package. To install this,

```
pip install mysql-connector-python
```
Once this package is installed all the required dependencies will be resolved and this script can be executed successfully. 

The above script initializes the db config class based on the configurations available from the config.py script which is used to import the DatabaseConfig class.
Once the class is initialized it will establish DB Connection. It will report any connection issues through the logger info or warning in the output window or terminal based on how the project is executed.

Other SQL operations like execute query. The input to this function is a query that is converted from natural language to SQL syntax using LLM. This is discussed further. 
This function will execute the query and display results. Similarly, there are other operations to describe table, database schema etc.

- The ***llm_client.py*** script.
Ideally, this script would not have exist or would have been optional in a traditional client/ server implementation. Traditionally, the client was known to send a syntactically correct SQL statement to the server for execution. However, the beauty of LLM integration and this script is that the client interaction with the MCP server can be in natural language and the MCP server then take advantage of LLM integration through this script to utilize either openai API or local Ollama model to translate natural language to a SQL syntax. 

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

- The ***server.py*** script
Resolve all the dependencies like installation of mcp, typing, asyncio packages by running the below command inside the virtual environment.

```
pip install mcp
pip install asyncio
pip install typing
```
This script is the core of this project. The MCP Server. It is actually a wrapper for all the other scripts. It contains the initializers and handler functions to manage the client request.
The MCP Server start function will initialize the server and wait for read/ write stream for client request. 
The get prompt function handles the client request and shows the help-content to guide the client to formulate a request that can be successfully responded back. 

- The ***main.py** script
This is the main entry point to initialize the MySQLMCPServer function of the MCP Server script.
Code here is self-explanatory.

## **Deployment using Docker**
- The requirements.txt has all the dependencies required for this project.
- The docker-compose.yml will be containerized deployment with mysql8 and ollama image running
```
docker compose up
```
If the init.sql is in the right path, it will also create the test_db and tables with data in it. If for some reason the init.sql is not found and initial schema is not available, this step can be manually perform using MySQL Workbench. 

## **Testing the project**
- The script folder has a python script ***test_examples.py*** for testing the project.
Create a vscode testing project to execute the test. ***Note-*** You will have to download all the dependencies required for the test like pytest.

## **Testing Ollama**
- First ensure that ollama container is up and running.
- Check if the following command lists any models
```
curl http://localhost:11434/api/tags
```
If no models are available, then run the following command.
```
docker exec -it <<container_ID>> ollama pull llama2
```
Once the model is successfully downloaded, run the ollama_test.py file to ensure that ollama is functioning properly.
