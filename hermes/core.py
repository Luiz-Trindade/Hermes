# ==================== core.py ====================
from collections import deque

from llama_index.core.agent.workflow import FunctionAgent

# --- hotfix: corrige erro "unhashable type: 'FunctionAgent'" ---
FunctionAgent.__hash__ = lambda self: id(self)
# ---------------------------------------------------------------

from hermes.utils import (
    get_api_key_from_provider,
    get_model_from_provider,
    get_random_api_key,
    convert_tools_to_function_tools,
    get_enhanced_prompt,
    enhance_input_data,
)


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
        """
        Initialize the Agent with the given parameters.

        Args:
            provider (str): The name of the provider for the agent (e.g., "OpenAI", "Azure").
            model (str): The model identifier to be used by the agent (e.g., "gpt-3.5-turbo").
            api_key (str): A comma-separated string of API keys for authentication. If not provided,
                           the API key will be fetched from the provider.
            name (str): The name of the agent, used for identification purposes.
            description (str): A brief description of the agent's purpose or functionality.
            prompt (str): The initial prompt or context to be used by the agent.
            tools (list): A list of tools or functions that the agent can use during execution.
            temperature (float): A value between 0.0 and 1.0 that controls the randomness of the
                                 agent's responses. Higher values result in more randomness.
            max_chat_history_length (int): The maximum number of messages to retain in the chat history.
            debug (bool): A flag indicating whether debug mode is enabled. If True, additional
                          debugging information may be logged.
        """
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
        self.tools = convert_tools_to_function_tools(tools, self)

        # Initialize the agent with the first key (will be replaced in execute)
        self.agent = None
        self._initialize_agent()

    def _initialize_agent(self):
        """Initialize or reinitialize the agent with current API key."""
        self.agent = FunctionAgent(
            name=self.name,
            description=self.description,
            system_prompt=get_enhanced_prompt(
                self.name, self.description, self.prompt, self.tools
            ),
            tools=self.tools,
            llm=get_model_from_provider(
                self.provider,
                self.model,
                get_random_api_key(self.api_keys),
                self.temperature,
                self.debug,
                len(self.api_keys) > 1,
            ),
        )

    async def execute(self, input_data=None, chat_history=None):
        """Execute the agent with the given input data and chat history."""
        try:
            # Reinitialize the agent with a new random API key on each execution
            if len(self.api_keys) > 1:
                self._initialize_agent()

            response = await self.agent.run(
                user_msg=enhance_input_data(input_data, self.debug),
                chat_history=deque(
                    chat_history or [], maxlen=self.max_chat_history_length
                ),
            )
            return response
        except Exception as e:
            return f"Error executing agent: {str(e)}"

    # async def execute_web_interface(self):
    #     """Serve the agent's web interface."""
    #     from hermes.utils import execute_web_interface

    #     await execute_web_interface(directory="hermes/web_interface")
