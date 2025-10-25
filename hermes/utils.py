import os
import yake
from langdetect import detect
from .providers import PROVIDER_KEY_MAPPING
from dotenv import load_dotenv

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


# Alternative version that preserves empty lines
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


# More flexible version that allows different levels of indentation
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

    # Se o provedor existe no mapeamento, tenta todas as possíveis keys
    if provider_lower in PROVIDER_KEY_MAPPING:
        for key_name in PROVIDER_KEY_MAPPING[provider_lower]:
            api_key = os.getenv(key_name)
            if api_key:
                return api_key

    # Fallback: tenta o padrão PROVIDER_API_KEY
    return os.getenv(f"{provider.upper()}_API_KEY", "")
