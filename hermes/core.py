import asyncio
import random
from collections import deque
from datetime import datetime
from itertools import count

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool

from hermes.utils import extract_keywords, format_text, get_api_key_from_provider


class Agent:
    def __init__(
        self,
        provider="",
        model="",
        api_key="",
        name="",
        description="",
        prompt="",
        tools=[],
        temperature=0.7,
        max_chat_history_length=20,
        debug=False,
    ):
        """Initialize the Agent with the given parameters."""
        self.provider = provider
        self.model = model
        self.debug = debug

        # transform api_key into list, remove spaces and empty strings
        self.api_keys = [k.strip() for k in (api_key or "").split(",") if k.strip()]

        # if no keys provided, fetch from provider
        if not self.api_keys:
            self.api_keys = [get_api_key_from_provider(provider)]

        self.name = name
        self.description = description
        self.prompt = prompt
        self.temperature = max(0.0, min(temperature, 1.0))
        self.max_chat_history_length = max_chat_history_length

        # Convert tools and agents to FunctionTools
        self.tools = self._convert_tools(tools)

        # Initialize the agent with the first key (will be replaced in execute)
        self.agent = None
        self._initialize_agent()

    def _initialize_agent(self):
        """Initialize or reinitialize the agent with current API key."""
        self.agent = FunctionAgent(
            name=self.name,
            description=self.description,
            system_prompt=self._get_enhanced_prompt(),
            tools=self.tools,
            llm=self._get_model_from_provider(),
        )

    def _get_random_api_key(self):
        """Select a random API key from the available keys."""
        return random.choice(self.api_keys)

    def _create_agent_wrapper(self, agent_instance):
        """
        Creates a wrapper function for an Agent instance.

        Args:
            agent_instance: An instance of Agent class

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
            if self.debug:
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

    def _convert_tools(self, tools):
        """
        Convert functions and Agent instances to FunctionTool objects automatically.

        Args:
            tools: List containing functions, FunctionTools, or Agent instances

        Returns:
            List of FunctionTool objects
        """
        converted_tools = []

        for tool in tools:
            # If it's an Agent instance, create wrapper function
            if isinstance(tool, Agent):
                if self.debug:
                    print(f"✨ Creating automatic tool for agent: {tool.name}")
                wrapper_func = self._create_agent_wrapper(tool)

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

    def _get_model_from_provider(self):
        """Dynamically import and return the model based on the provider."""
        try:
            # Select a random API key if there are multiple
            current_api_key = self._get_random_api_key()

            if len(self.api_keys) > 1 and self.debug:
                # Mask the key for logging (show only first and last characters)
                masked_key = (
                    f"{current_api_key[:8]}...{current_api_key[-4:]}"
                    if len(current_api_key) > 12
                    else "***"
                )
                print(f"🔑 Using API key: {masked_key}")

            if self.provider == "openai":
                from llama_index.llms.openai import OpenAI

                model = OpenAI(
                    model=self.model,
                    api_key=current_api_key,
                    temperature=self.temperature,
                )
            elif self.provider == "azure":
                from llama_index.llms.azure_openai import AzureOpenAI

                model = AzureOpenAI(
                    model=self.model,
                    api_key=current_api_key,
                    temperature=self.temperature,
                )
            elif self.provider == "anthropic":
                from llama_index.llms.anthropic import Anthropic

                model = Anthropic(
                    model=self.model,
                    api_key=current_api_key,
                    temperature=self.temperature,
                )
            elif self.provider in ["gemini", "google"]:
                from llama_index.llms.google_genai import GoogleGenAI

                model = GoogleGenAI(
                    model=self.model,
                    api_key=current_api_key,
                    temperature=self.temperature,
                )
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            return model
        except ImportError as e:
            raise ImportError(
                f"Failed to import model for provider {self.provider}. "
                f"Please ensure the required package is installed."
            ) from e

    def _get_enhanced_prompt(self):
        """Generate an enhanced prompt with identity, instructions, and tools."""
        current_datetime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        enhanced_prompt = format_text(
            f"""
            # Identity (Agent Profile):
            Your name is '{self.name}'
            You are described as: '{self.description}'

            # Instructions (Behavior Guidelines):
            {self.prompt}

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

        if len(self.tools) > 0:
            enhanced_prompt += "\n# Available tools (to assist you):\n"
            for idx, tool in enumerate(self.tools, start=1):
                enhanced_prompt += (
                    f"{idx}. {tool.metadata.name}: {tool.metadata.description}\n"
                )

        return enhanced_prompt

    def _enhance_input_data(self, input_data):
        """Enhance input data if necessary (placeholder for future enhancements)."""
        keywords = extract_keywords(input_data)

        enhanced_input = f"""
            # User Input (to be processed):
            {input_data}

            # Extracted Keywords For User Input (to focus on):
            {', '.join(keywords)}
        """.strip()

        if self.debug:
            print(f"\n🛠️ Enhanced Input Data:\n{format_text(enhanced_input)}\n")
        return format_text(enhanced_input)

    async def execute(self, input_data=None, chat_history=None):
        """Execute the agent with the given input data and chat history."""
        try:
            # Reinitialize the agent with a new random API key on each execution
            if len(self.api_keys) > 1:
                self._initialize_agent()

            response = await self.agent.run(
                user_msg=self._enhance_input_data(input_data),
                chat_history=deque(
                    chat_history or [], maxlen=self.max_chat_history_length
                ),
            )
            return response
        except Exception as e:
            return f"Error executing agent: {str(e)}"
