from agents.critic import Critic


critic = Critic()
evidence = [
    {
        "source": "incident.pdf",
        "page": 10,
        "text": (
            "Database connections were exhausted during "
            "the production incident. The exhausted "
            "connection pool caused API requests to wait "
            "for available database connections, resulting "
            "in increased API latency."
        )
    }
]


answer = (
    "API latency increased because the database "
    "connection pool was exhausted, causing API "
    "requests to wait for available connections."
)

result = critic.evaluate(
    answer,
    evidence
)


print("\n==============================")
print("CRITIC RESULT")
print("==============================")

print(result)