# Hermes AI Agent Framework

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/LlamaIndex-0.14.5+-orange.svg" alt="LlamaIndex">
  <img src="https://img.shields.io/badge/OpenAI-GPT--4-green.svg" alt="OpenAI">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
  <img src="https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg" alt="Status">
</p>

## 🚀 Smart LlamaIndex Abstraction for AI Agents

Hermes is a lightweight, powerful abstraction layer over LlamaIndex that simplifies building production-ready AI agents with essential utilities and multi-agent capabilities.

## ✨ Features

- **🛠️ Automatic Tool Integration** - Convert Python functions to tools effortlessly
- **🤖 Multi-Agent Orchestration** - Coordinate multiple specialized agents
- **🧠 Enhanced Prompt Management** - Built-in Chain of Thought and context awareness
- **💬 Smart Memory Management** - Automatic chat history handling with configurable limits
- **🔌 Multi-Provider Support** - OpenAI, Azure, Anthropic, and more
- **📊 Text Processing Utilities** - Keyword extraction, formatting, and enhancement

## 🚀 Quick Start

```bash
pip install hermes-ai
```

```python
import asyncio
from hermes.core import Agent

async def main():
    # Create a simple tool
    def get_weather(location: str) -> str:
        """Get weather information for a location."""
        return f"Weather in {location}: Sunny, 25°C"

    # Initialize your agent
    agent = Agent(
        provider="openai",
        model="gpt-4o-mini",
        name="WeatherAssistant",
        description="Helps with weather information",
        prompt="Provide accurate and helpful weather updates.",
        tools=[get_weather]
    )

    # Execute the agent
    response = await agent.execute("What's the weather in Tokyo?")
    print(response)

asyncio.run(main())
```

## 🏗️ Multi-Agent System

```python
# Create specialized agents
market_agent = Agent(
    name="FinancialAnalyst",
    description="Stock market expert",
    tools=[get_market_data]
)

investment_agent = Agent(
    name="InvestmentAdvisor", 
    description="Investment strategy expert",
    tools=[calculate_returns]
)

# Coordinate with master agent
coordinator = Agent(
    name="FinanceCoordinator",
    description="Coordinates financial experts",
    tools=[market_agent.execute, investment_agent.execute]
)
```

## 🔧 Core Components

### Agent Configuration
```python
Agent(
    provider="openai",           # LLM provider
    model="gpt-4",              # Model name
    name="Assistant",           # Agent identity
    description="Helpful AI",   # Agent purpose
    prompt="Behavior guidelines", # System prompt
    tools=[function1, function2], # Available tools
    temperature=0.7,            # Creativity control
    max_chat_history_length=20  # Memory management
)
```

### Built-in Utilities
- **Automatic tool conversion** from Python functions
- **Enhanced prompt formatting** with current context
- **Chat history management** with configurable limits
- **Input enhancement** with keyword extraction
- **Multi-provider LLM support**

## 📁 Project Structure

```
hermes/
├── core.py          # Main Agent class
├── utils.py         # Text processing utilities
├── tools.py         # Tool creation helpers
└── examples/        # Usage examples
```

## 🎯 Use Cases

- **Customer Support Agents**
- **Multi-Domain Expert Systems** 
- **Research Assistants**
- **Data Analysis Tools**
- **Content Generation Systems**

## 🔮 Roadmap

- [ ] Vector database integration
- [ ] Advanced memory backends
- [ ] Streaming responses
- [ ] Plugin system
- [ ] Web interface

## 💡 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

<p align="center">
  <em>Build intelligent agents faster with Hermes - The messenger of AI capabilities</em>
</p>