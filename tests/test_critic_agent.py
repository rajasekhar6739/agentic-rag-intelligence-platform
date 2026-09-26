from agents.critic_agent import CriticAgent

# USE THE SAME IMPORT THAT test_critic.py ALREADY USES
from agents.critic import Critic


critic = Critic()

agent_critic = CriticAgent(
    critic
)

result = agent_critic.evaluate(
    query="Why did API latency increase?",
    answer=(
        "API latency increased because "
        "database connections were exhausted."
    ),
    evidence=[
        "Database connections were exhausted."
    ]
)

print("\n==============================")
print("CRITIC AGENT")
print("==============================")

print(result)