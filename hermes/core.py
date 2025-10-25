from datetime import datetime
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from hermes.utils import extract_keywords, format_text


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
        agents=[],
        temperature=0.0,
        max_chat_history_length=20,
    ):
        """Initialize the Agent with the given parameters."""
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.name = name
        self.description = description
        self.prompt = prompt
        self.tools = self._convert_tools(tools)
        self.agents = agents
        self.temperature = max(0.0, min(temperature, 1.0))
        self.max_chat_history_length = max_chat_history_length
        self.agent = FunctionAgent(
            name=self.name,
            description=self.description,
            system_prompt=self._get_enhanced_prompt(),
            tools=self.tools,
            llm=self._get_model_from_provider(),
        )

    def _convert_tools(self, tools):
        """Convert functions to FunctionTool objects automatically."""
        converted_tools = []
        for tool in tools:
            if callable(tool) and not isinstance(tool, FunctionTool):
                # Se for uma função, criar FunctionTool
                tool_name = tool.__name__
                tool_description = tool.__doc__ or f"Tool for {tool_name}"
                converted_tool = FunctionTool.from_defaults(
                    fn=tool, name=tool_name, description=tool_description
                )
                converted_tools.append(converted_tool)
            else:
                # Se já for FunctionTool, usar diretamente
                converted_tools.append(tool)
        return converted_tools

    def _get_model_from_provider(self):
        """Dynamically import and return the model based on the provider."""
        try:
            if self.provider == "openai":
                from llama_index.llms.openai import OpenAI

                model = OpenAI(
                    model=self.model, api_key=self.api_key, temperature=self.temperature
                )
            elif self.provider == "azure":
                from llama_index.llms.azure_openai import AzureOpenAI

                model = AzureOpenAI(
                    model=self.model, api_key=self.api_key, temperature=self.temperature
                )
            elif self.provider == "anthropic":
                from llama_index.llms.anthropic import Anthropic

                model = Anthropic(
                    model=self.model, api_key=self.api_key, temperature=self.temperature
                )
            elif self.provider in ["gemini", "google"]:
                from llama_index.llms.google_genai import GoogleGenAI

                model = GoogleGenAI(
                    model=self.model,
                    api_key=self.api_key,
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
            enhanced_prompt += "\n# Available tools (to assist you):"
            for idx, tool in enumerate(self.tools, start=1):
                enhanced_prompt += (
                    f"{idx}. {tool.metadata.name}: {tool.metadata.description}\n"
                )

        return enhanced_prompt

    def _auto_adjust_chat_history(self, chat_history):
        """Automatically adjust chat history to maintain the maximum length."""
        if chat_history and len(chat_history) > self.max_chat_history_length:
            return chat_history[-self.max_chat_history_length :]
        return chat_history or []

    def _enhance_input_data(self, input_data):
        """Enhance input data if necessary (placeholder for future enhancements)."""
        keywords = extract_keywords(input_data)

        enhanced_input = f"""
            # User Input (to be processed):
            {input_data}

            # Extracted Keywords For User Input (to focus on):
            {', '.join(keywords)}
        """.strip()

        print(f"\n🛠️ Enhanced Input Data:\n{format_text(enhanced_input)}\n")
        return format_text(enhanced_input)

    async def execute(self, input_data=None, chat_history=None):
        """Execute the agent with the given input data and chat history."""
        try:
            response = await self.agent.run(
                user_msg=self._enhance_input_data(input_data),
                chat_history=self._auto_adjust_chat_history(chat_history),
            )
            return response
        except Exception as e:
            return f"Error executing agent: {str(e)}"
