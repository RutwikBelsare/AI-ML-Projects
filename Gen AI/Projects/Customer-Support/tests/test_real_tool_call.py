import asyncio

from backend.services.ollama_service import OllamaService
from backend.services.tool_calling_service import ToolCallingService


async def main():

    llm_service = OllamaService()

    service = ToolCallingService(
        llm_service=llm_service
    )

    result = await service.process_message(
        "Where is my order ORD1001?"
    )

    print()
    print("========== FINAL RESPONSE ==========")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())