# ==================== utils.py ====================
import os
import random
import asyncio
from datetime import datetime
import yake
from langdetect import detect
from dotenv import load_dotenv
from pathlib import Path
from importlib import resources

from llama_index.core.tools import FunctionTool

from .providers import PROVIDER_KEY_MAPPING

load_dotenv()


def extract_keywords(text, max_keywords=10, score_threshold=0.1):
    """Extracts relevant keywords from a text"""
    try:
        language = detect(text)
        kw_extractor = yake.KeywordExtractor(lan=language, n=2, top=30)
        keywords = kw_extractor.extract_keywords(text)
        filtered = []
        for kw, score in keywords:
            if score <= score_threshold:
                filtered.append(kw)
            if len(filtered) >= max_keywords:
                break
        return filtered
    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return []


def format_text(text: str, spaces: int = 4) -> str:
    """
    Formats a text ensuring all lines have at least the specified indentation.

    Args:
        text (str): The text to be formatted
        spaces (int): Number of spaces desired for indentation (default: 4)

    Returns:
        str: Text formatted with the correct indentation
    """
    lines = text.split("\n")
    formatted_lines = []
    for line in lines:
        # Remove leading spaces and tabs
        clean_line = line.lstrip()
        # Add the desired indentation
        formatted_line = " " * spaces + clean_line
        formatted_lines.append(formatted_line)
    return "\n".join(formatted_lines)


def format_text_preserving_empty_lines(text: str, spaces: int = 4) -> str:
    """
    Formats a text ensuring the specified indentation, preserving empty lines.
    """
    lines = text.split("\n")
    formatted_lines = []
    for line in lines:
        if line.strip():  # If the line is not empty
            clean_line = line.lstrip()
            formatted_line = " " * spaces + clean_line
        else:
            formatted_line = ""  # Keep empty line
        formatted_lines.append(formatted_line)
    return "\n".join(formatted_lines)


def advanced_format_text(
    text: str, spaces: int = 4, preserve_empty: bool = True
) -> str:
    """
    Formats text with control over spaces and handling of empty lines.

    Args:
        text (str): Text to format
        spaces (int): Number of spaces for indentation
        preserve_empty (bool): If True, preserves empty lines

    Returns:
        str: Formatted text
    """
    lines = text.split("\n")
    formatted_lines = []
    for line in lines:
        if preserve_empty and not line.strip():
            formatted_lines.append("")
        else:
            clean_line = line.lstrip()
            formatted_line = " " * spaces + clean_line
            formatted_lines.append(formatted_line)
    return "\n".join(formatted_lines)


def get_api_key_from_provider(provider: str) -> str:
    """
    Retrieves the API key for a given provider from environment variables.

    Args:
        provider (str): The name of the provider (e.g., 'openai', 'anthropic')

    Returns:
        str: The API key for the specified provider, or empty string if not found.
    """
    provider_lower = provider.lower().strip()
    # If the provider exists in the mapping, try all possible keys
    if provider_lower in PROVIDER_KEY_MAPPING:
        for key_name in PROVIDER_KEY_MAPPING[provider_lower]:
            api_key = os.getenv(key_name)
            if api_key:
                return api_key
    # Fallback: try the standard PROVIDER_API_KEY
    return os.getenv(f"{provider.upper()}_API_KEY", "")


def get_random_api_key(api_keys):
    """Select a random API key from the available keys."""
    return random.choice(api_keys)


def get_model_from_provider(
    provider, model, api_key, temperature, debug, has_multiple_keys
):
    """
    Dynamically import and return the model based on the provider.

    Args:
        provider: The provider name (e.g., 'openai', 'anthropic')
        model: The model name
        api_key: The API key to use
        temperature: The temperature setting
        debug: Whether to print debug messages
        has_multiple_keys: Whether multiple API keys are available

    Returns:
        The initialized model instance
    """
    try:
        if has_multiple_keys and debug:
            # Mask the key for logging (show only first and last characters)
            masked_key = (
                f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
            )
            print(f"🔑 Using API key: {masked_key}")

        if provider == "openai":
            from llama_index.llms.openai import OpenAI

            return OpenAI(
                model=model,
                api_key=api_key,
                temperature=temperature,
            )
        elif provider == "azure":
            from llama_index.llms.azure_openai import AzureOpenAI

            return AzureOpenAI(
                model=model,
                api_key=api_key,
                temperature=temperature,
            )
        elif provider == "anthropic":
            from llama_index.llms.anthropic import Anthropic

            return Anthropic(
                model=model,
                api_key=api_key,
                temperature=temperature,
            )
        elif provider in ["gemini", "google"]:
            from llama_index.llms.google_genai import GoogleGenAI

            return GoogleGenAI(
                model=model,
                api_key=api_key,
                temperature=temperature,
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    except ImportError as e:
        raise ImportError(
            f"Failed to import model for provider {provider}. "
            f"Please ensure the required package is installed."
        ) from e


def create_agent_wrapper(agent_instance, debug):
    """
    Creates a wrapper function for an Agent instance.

    Args:
        agent_instance: An instance of Agent class
        debug: Whether to print debug messages

    Returns:
        A callable function that wraps the agent's execute method
    """

    def wrapper_function(query: str) -> str:
        """
        Consults the specialized agent with the given query.

        Args:
            query (str): The question or task to be processed by the agent

        Returns:
            str: The agent's response
        """
        if debug:
            print(f"\n🔄 Consulting agent '{agent_instance.name}'...")
        try:
            # Execute the agent synchronously
            response = asyncio.run(agent_instance.execute(input_data=query))
            return str(response)
        except Exception as e:
            return f"Error consulting {agent_instance.name}: {str(e)}"

    # Define function name and description based on the agent
    wrapper_function.__name__ = (
        f"consult_{agent_instance.name.lower().replace(' ', '_')}"
    )
    wrapper_function.__doc__ = format_text(
        f"""
        Consults the specialized agent '{agent_instance.name}'.
        
        Agent Description: {agent_instance.description}
        
        Args:
            query (str): The question or task to be processed by this specialist
            
        Returns:
            str: Detailed response from {agent_instance.name}
    """
    )

    return wrapper_function


def convert_tools_to_function_tools(tools, agent_instance):
    """
    Convert functions and Agent instances to FunctionTool objects automatically.

    Args:
        tools: List containing functions, FunctionTools, or Agent instances
        agent_instance: The Agent instance that owns these tools (for accessing debug flag)

    Returns:
        List of FunctionTool objects
    """
    # Import here to avoid circular dependency
    from hermes.core import Agent

    converted_tools = []

    for tool in tools:
        # If it's an Agent instance, create wrapper function
        if isinstance(tool, Agent):
            if agent_instance.debug:
                print(f"✨ Creating automatic tool for agent: {tool.name}")
            wrapper_func = create_agent_wrapper(tool, agent_instance.debug)

            # Create FunctionTool from wrapper function
            converted_tool = FunctionTool.from_defaults(
                fn=wrapper_func,
                name=wrapper_func.__name__,
                description=wrapper_func.__doc__,
            )
            converted_tools.append(converted_tool)

        # If it's a regular function (not FunctionTool)
        elif callable(tool) and not isinstance(tool, FunctionTool):
            tool_name = tool.__name__
            tool_description = tool.__doc__ or f"Tool for {tool_name}"
            converted_tool = FunctionTool.from_defaults(
                fn=tool, name=tool_name, description=tool_description
            )
            converted_tools.append(converted_tool)

        # If it's already a FunctionTool, use directly
        else:
            converted_tools.append(tool)

    return converted_tools


def get_enhanced_prompt(name, description, prompt, tools):
    """
    Generate an enhanced prompt with identity, instructions, and tools.

    Args:
        name: The agent's name
        description: The agent's description
        prompt: The agent's base prompt/instructions
        tools: List of tools available to the agent

    Returns:
        str: The enhanced prompt
    """
    current_datetime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    enhanced_prompt = format_text(
        f"""
        # Identity (Agent Profile):
        Your name is: '{name}'
        Your description is: '{description}'

        # Instructions (Behavior Guidelines):
        {prompt}

        # Thinking Process (Chain of Thought):
        1. First, analyze the user's query carefully to understand the intent and context
        2. Break down complex problems into smaller, manageable steps
        3. Consider available tools and resources that could help solve this problem
        4. Evaluate different approaches before selecting the best one
        5. Explain your reasoning step by step when appropriate
        6. Verify your solution makes sense before providing the final answer

        # Response Structure (How to Reply):
        - For complex queries: Show your thinking process before giving the final answer
        - For simple queries: Provide a direct response with optional brief explanation
        - Always maintain a helpful and professional tone

        # Current date and time (for context): 
        {current_datetime}
    """
    )

    if len(tools) > 0:
        enhanced_prompt += "\n# Available tools (to assist you):\n"
        for idx, tool in enumerate(tools, start=1):
            enhanced_prompt += (
                f"{idx}. {tool.metadata.name}: {tool.metadata.description}\n"
            )

    return enhanced_prompt


def enhance_input_data(input_data, debug):
    """
    Enhance input data with extracted keywords.

    Args:
        input_data: The raw input from the user
        debug: Whether to print debug messages

    Returns:
        str: Enhanced input data
    """
    keywords = extract_keywords(input_data)

    enhanced_input = f"""
        # User Input (to be processed):
        {input_data}

        # Extracted Keywords For User Input (to focus on):
        {', '.join(keywords)}
    """.strip()

    if debug:
        print(f"\n🛠️ Enhanced Input Data:\n{format_text(enhanced_input)}\n")

    return format_text(enhanced_input)


# async def execute_web_interface(port=8000, directory="hermes/web_interface"):
#     from hermes.web import serve_static_fastapi
#     import hermes

#     try:
#         # Absolute path inside the installed package
#         package_dir = Path(resources.files(hermes) / "web_interface")

#         # Check if it exists inside the installed package
#         if package_dir.exists():
#             await serve_static_fastapi(port=port, directory=str(package_dir))
#             return

#     except Exception:
#         pass  # Fallback if package is not packaged yet (dev mode)

#     # Dev mode: try relative path (useful when running directly from repo)
#     relative_dir = Path(__file__).resolve().parent / "web_interface"
#     if relative_dir.exists():
#         await serve_static_fastapi(port=port, directory=str(relative_dir))
#     else:
#         raise RuntimeError(f"Could not find web_interface folder at: {relative_dir}")
