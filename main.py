import asyncio
from hermes.core import Agent
from llama_index.core.llms import ChatMessage, MessageRole


async def main():
    # Ferramentas para os especialistas
    def get_market_info(query: str) -> str:
        """
        Provides financial market updates based on the query.

        Args:
            query (str): The query about the financial market.

        Returns:
            str: A summary of market conditions.
        """
        print("\n🔄 Obtendo informações do mercado financeiro...")
        return "O mercado financeiro está otimista hoje, com os principais índices em alta de 2%. A cotação do dólar está em R$ 5,10."

    def calculate_investment(amount: float, period: int) -> str:
        """
        Calculate investment returns and provide recommendations.

        Args:
            amount (float): Investment amount
            period (int): Investment period in months

        Returns:
            str: Investment analysis and recommendations
        """
        print("\n🔄 Calculando retorno de investimento...")
        expected_return = amount * (1 + 0.12) ** (period / 12)
        return (
            f"💰 Investimento de R$ {amount:,.2f} por {period} meses:\n"
            f"• Retorno esperado: R$ {expected_return:,.2f}\n"
            f"• Rentabilidade: 12% ao ano\n"
            f"• Recomendação: Fundos imobiliários e ações blue chips"
        )

    # Agente Especialista em Mercado
    market_agent = Agent(
        provider="openai",
        model="gpt-4o-mini",
        name="AnalistaFinanceiro",
        description="Especialista em análise de mercado e cotações",
        prompt="Forneça análises detalhadas do mercado financeiro, tendências e cotações atuais. Seja preciso e use dados concretos.",
        tools=[get_market_info],
        temperature=1.0,
        debug=True,
    )

    # Agente Especialista em Investimentos
    investment_agent = Agent(
        provider="openai",
        model="gpt-4o-mini",
        name="ConsultorInvestimentos",
        description="Especialista em planejamento de investimentos",
        prompt="Ajude com estratégias de investimento, cálculo de retornos e recomendações baseadas no perfil do cliente.",
        tools=[calculate_investment],
        temperature=1.0,
        debug=True,
    )

    # Agente Coordenador
    coordinator_agent = Agent(
        provider="openai",
        model="gpt-4o-mini",
        name="Coordenador",
        description="Coordena e direciona perguntas para os especialistas apropriados",
        prompt="""Analise a pergunta do usuário e decida qual especialista consultar:
        - Para perguntas sobre mercado, cotações, tendências: use o AnalistaFinanceiro
        - Para perguntas sobre investimentos, retornos, planejamento: use o ConsultorInvestimentos
        - Para perguntas complexas que envolvam ambos: consulte os dois especialistas
        
        Sempre explique qual especialista está sendo consultado e por quê.""",
        tools=[market_agent, investment_agent],
        temperature=0.5,
        debug=True,
    )

    chat_history = []

    print("=== SISTEMA MULTI-AGENTE FINANCEIRO (VERSÃO SIMPLIFICADA) ===")
    print("Digite 'exit' para sair\n")

    while True:
        input_data = input("👤 Sua pergunta: ")

        if input_data.lower() == "exit":
            break

        response = await coordinator_agent.execute(
            input_data=input_data, chat_history=chat_history
        )

        # Atualizar histórico
        chat_history.append(ChatMessage(role=MessageRole.USER, content=input_data))
        chat_history.append(
            ChatMessage(role=MessageRole.ASSISTANT, content=str(response))
        )

        print(f"🤖 Resposta: {response}")
        print("-" * 80)
        print("")


if __name__ == "__main__":
    asyncio.run(main())
