from langchain_community.tools import DuckDuckGoSearchResults
import os
from langchain_azure_dynamic_sessions import SessionsPythonREPLTool
from PIL import Image
import base64
from io import BytesIO


#########################################
# TOOLS #
#########################################
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b


def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b


def divide(a: int, b: int) -> float:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b


def web_search(question: str) -> str:
    """Searches the web for the question.

    Args:
        question: question to search for
    """
    search = DuckDuckGoSearchResults(output_format="list")
    return search.invoke(question)


def python_repl(code: str) -> str:
    """Executes the code in a python repl.

    Args:
        code: code to execute
    """
    python_repl = SessionsPythonREPLTool(pool_management_endpoint=os.getenv("POOL_MANAGEMENT_ENDPOINT"))
    result = python_repl.execute(code)
    print(result)
    if isinstance(result, dict) and "result" in result and result['result']['type'] == 'image':
        return result
    
    return result
