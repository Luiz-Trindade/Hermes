# ==================== core.py ====================
from collections import deque

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.memory import ChatSummaryMemoryBuffer
from llama_index.core.llms import ChatMessage, MessageRole

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
        token_limit=2000,
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
            token_limit (int): Maximum number of tokens for the memory buffer. When exceeded,
                              older messages will be summarized automatically.
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
        self.token_limit = token_limit

        # Convert tools and agents to FunctionTools
        self.tools = convert_tools_to_function_tools(tools, self)

        # Initialize LLM and tokenizer
        self.llm, self.tokenize_fn = get_model_from_provider(
            self.provider,
            self.model,
            get_random_api_key(self.api_keys),
            self.temperature,
            self.debug,
            len(self.api_keys) > 1,
        )

        # Initialize memory buffer for chat history management
        self.memory = None
        self._initialize_memory()

        # Initialize the agent
        self.agent = None
        self._initialize_agent()

    def _initialize_memory(self):
        """Initialize the ChatSummaryMemoryBuffer with current LLM and tokenizer."""
        self.memory = ChatSummaryMemoryBuffer.from_defaults(
            llm=self.llm,
            token_limit=self.token_limit,
            tokenizer_fn=self.tokenize_fn,
        )

    def _initialize_agent(self):
        """Initialize or reinitialize the agent with current LLM."""
        self.agent = FunctionAgent(
            name=self.name,
            description=self.description,
            system_prompt=get_enhanced_prompt(
                self.name, self.description, self.prompt, self.tools
            ),
            tools=self.tools,
            llm=self.llm,
        )

    def _update_llm(self):
        """Update LLM with a new random API key."""
        if len(self.api_keys) > 1:
            self.llm, self.tokenize_fn = get_model_from_provider(
                self.provider,
                self.model,
                get_random_api_key(self.api_keys),
                self.temperature,
                self.debug,
                len(self.api_keys) > 1,
            )
            # Reinitialize memory with new LLM and tokenizer
            self._initialize_memory()

    def _convert_chat_history_to_messages(self, chat_history):
        """
        Convert chat history from dict format to ChatMessage objects.

        Args:
            chat_history: List of dicts with 'role' and 'content' keys
                         Example: [{"role": "user", "content": "Hello"}, ...]

        Returns:
            List of ChatMessage objects
        """
        if not chat_history:
            return []

        messages = []
        for msg in chat_history:
            # Determina o role
            role_str = msg.get("role", "user").lower()

            if role_str == "user":
                role = MessageRole.USER
            elif role_str == "assistant":
                role = MessageRole.ASSISTANT
            elif role_str == "system":
                role = MessageRole.SYSTEM
            else:
                role = MessageRole.USER  # Default para user

            # Cria o ChatMessage
            messages.append(ChatMessage(role=role, content=msg.get("content", "")))

        return messages

    def _has_summarization(self, messages):
        """
        Check if the chat history has been summarized.

        Args:
            messages: List of ChatMessage objects

        Returns:
            bool: True if there's a SYSTEM message (indicating summarization)
        """
        return any(msg.role == MessageRole.SYSTEM for msg in messages)

    def _count_tokens(self, messages):
        """
        Count total tokens in the message list.

        Args:
            messages: List of ChatMessage objects

        Returns:
            int: Total token count
        """
        total_tokens = 0
        for msg in messages:
            try:
                total_tokens += len(self.tokenize_fn(msg.content))
            except Exception:
                # Fallback: estimate 4 characters per token
                total_tokens += len(msg.content) // 4
        return total_tokens

    def _print_chat_history(self, title="📜 Chat History"):
        """
        Print the chat history in a formatted way.

        Args:
            title: Title for the history display
        """
        print(f"\n{'='*70}")
        print(f"{title}")
        print("=" * 70)

        messages = self.memory.get()
        has_summary = self._has_summarization(messages)
        token_count = self._count_tokens(messages)

        # Header info
        print(f"📊 Total messages: {len(messages)}")
        print(f"🎯 Token limit: {self.token_limit}")
        print(f"💾 Current tokens: ~{token_count}")

        if has_summary:
            print(f"✅ Status: SUMMARIZED (messages were compressed to save tokens)")
        else:
            print(f"📝 Status: FULL HISTORY (no summarization yet)")

        print("-" * 70)

        # Print each message
        for idx, msg in enumerate(messages, 1):
            role = msg.role.value.upper()
            content = msg.content

            # Truncate long messages for display
            display_content = content[:300] + "..." if len(content) > 300 else content

            # Special formatting for SYSTEM messages (summaries)
            if msg.role == MessageRole.SYSTEM:
                print(f"\n[{idx}] 🤖 {role} (SUMMARY)")
                print(f"    📝 {display_content}")
            elif msg.role == MessageRole.USER:
                print(f"\n[{idx}] 👤 {role}")
                print(f"    💬 {display_content}")
            elif msg.role == MessageRole.ASSISTANT:
                print(f"\n[{idx}] 🤖 {role}")
                print(f"    💡 {display_content}")
            else:
                print(f"\n[{idx}] ❓ {role}")
                print(f"    {display_content}")

        print("\n" + "=" * 70)

    def _update_memory_with_history(self, chat_history):
        """
        Update memory buffer with chat history.

        Args:
            chat_history: List of dicts with 'role' and 'content' keys
        """
        if not chat_history:
            return

        # Convert to ChatMessage objects
        messages = self._convert_chat_history_to_messages(chat_history)

        # Clear current memory and add new messages
        self.memory = ChatSummaryMemoryBuffer.from_defaults(
            chat_history=messages,
            llm=self.llm,
            token_limit=self.token_limit,
            tokenizer_fn=self.tokenize_fn,
        )

        if self.debug:
            has_summary = self._has_summarization(self.memory.get())
            print(f"\n📚 Memory updated with {len(messages)} input messages")
            print(f"   Token limit: {self.token_limit}")
            print(f"   Current memory size: {len(self.memory.get())} messages")
            if has_summary:
                print(f"   ⚠️  Summarization ACTIVE - older messages were compressed!")
            else:
                print(f"   ✅ No summarization needed yet")

    def get_chat_history(self):
        """
        Get the current chat history from memory in dict format.

        Returns:
            List of dicts with 'role' and 'content' keys
        """
        messages = self.memory.get()
        return [{"role": msg.role.value, "content": msg.content} for msg in messages]

    async def execute(self, input_data=None, chat_history=None):
        """
        Execute the agent with the given input data and chat history.

        Args:
            input_data: The user's input/query
            chat_history: List of dicts with 'role' and 'content' keys
                        Example: [
                            {"role": "user", "content": "Hello"},
                            {"role": "assistant", "content": "Hi there!"},
                        ]

        Returns:
            The agent's response
        """
        try:
            # Update LLM with a new random API key on each execution
            if len(self.api_keys) > 1:
                self._update_llm()
                self._initialize_agent()

            # Update memory with provided chat history (including current user message)
            if chat_history:
                self._update_memory_with_history(chat_history)

            # Print full chat history BEFORE execution if debug is enabled
            if self.debug:
                self._print_chat_history(title="📜 Chat History BEFORE Execution")

            # Get managed chat history from memory
            managed_history = self.memory.get()

            # Limit to max_chat_history_length (applied after summarization)
            if len(managed_history) > self.max_chat_history_length:
                managed_history = managed_history[-self.max_chat_history_length :]

            if self.debug:
                print(f"\n💬 Executing agent '{self.name}':")
                print(
                    f"   Input: {input_data[:100]}..."
                    if len(input_data) > 100
                    else f"   Input: {input_data}"
                )
                print(
                    f"   History length sent to agent: {len(managed_history)} messages"
                )

            # Execute the agent
            response = await self.agent.run(
                user_msg=enhance_input_data(input_data, self.debug),
                chat_history=deque(
                    managed_history, maxlen=self.max_chat_history_length
                ),
            )

            # DON'T add to memory here - let main.py manage the full history
            # The next call will receive the updated chat_history from main.py

            # Print full chat history AFTER execution if debug is enabled
            # (will show the same as BEFORE since we're not adding here)
            if self.debug:
                print(f"\n✅ Agent '{self.name}' execution completed!")
                print(f"💡 Response will be added to history by caller")

            return response

        except Exception as e:
            error_msg = f"Error executing agent: {str(e)}"
            if self.debug:
                import traceback

                print(f"\n❌ {error_msg}")
                print(traceback.format_exc())
            return error_msg

    # async def execute_web_interface(self):
    #     """Serve the agent's web interface."""
    #     from hermes.utils import execute_web_interface

    #     await execute_web_interface(directory="hermes/web_interface")
