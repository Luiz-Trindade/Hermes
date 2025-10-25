from typing import Dict

# Mapeamento de provedores para possíveis nomes de variáveis de ambiente
PROVIDER_KEY_MAPPING: Dict[str, list] = {
    "openai": [
        "OPENAI_API_KEY",
        "OPENAI_KEY",
        "OPEN_AI_KEY",
        "OPENAI_TOKEN",
    ],
    "azure": [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_KEY",
        "AZURE_API_KEY",
        "AZURE_KEY",
    ],
    "anthropic": [
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_KEY",
        "CLAUDE_API_KEY",
        "CLAUDE_KEY",
    ],
    "google": [
        "GOOGLE_API_KEY",
        "GOOGLE_KEY",
        "GEMINI_API_KEY",
        "GEMINI_KEY",
        "GOOGLE_AI_KEY",
    ],
    "gemini": [
        "GEMINI_API_KEY",
        "GEMINI_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_KEY",
    ],
    "cohere": [
        "COHERE_API_KEY",
        "COHERE_KEY",
        "CO_API_KEY",
    ],
    "huggingface": [
        "HUGGINGFACE_API_KEY",
        "HUGGINGFACE_TOKEN",
        "HF_API_KEY",
        "HF_TOKEN",
        "HUGGING_FACE_KEY",
    ],
    "replicate": [
        "REPLICATE_API_KEY",
        "REPLICATE_TOKEN",
        "REPLICATE_KEY",
    ],
    "together": [
        "TOGETHER_API_KEY",
        "TOGETHER_KEY",
        "TOGETHERAI_KEY",
    ],
    "mistral": [
        "MISTRAL_API_KEY",
        "MISTRAL_KEY",
        "MISTRALAI_KEY",
    ],
    "groq": [
        "GROQ_API_KEY",
        "GROQ_KEY",
    ],
    "perplexity": [
        "PERPLEXITY_API_KEY",
        "PERPLEXITY_KEY",
        "PPLX_API_KEY",
    ],
    "deepseek": [
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_KEY",
    ],
}
