
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.llm import llm


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI research assistant. "
            "Explain technical concepts clearly, accurately, "
            "and with practical examples.",
        ),
        ("human", "{question}"),
    ]
)

# Chain: Prompt -> LLM -> String output
chain = prompt | llm | StrOutputParser()


def main():
    question = (
        "Explain the difference between an AI agent, "
        "an LLM chain, and a predefined workflow."
    )

    answer = chain.invoke({"question": question})

    print("\n--- Research Assistant Response ---\n")
    print(answer)


if __name__ == "__main__":
    main()
