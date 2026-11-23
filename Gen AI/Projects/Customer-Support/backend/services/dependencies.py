from backend.core.config import get_settings
from backend.services.llm_service import LLMService
from backend.services.ollama_service import OllamaService
from backend.services.support_services import SupportService
from backend.services.tool_calling_service import ToolCallingService

def get_llm_service():

    settings = get_settings()

    if settings.llm_provider.lower() == "ollama":
        return OllamaService()

    return LLMService()


def get_support_service():

    llm_service = get_llm_service()

    return SupportService(
        llm_service=llm_service
    )

def get_tool_calling_service():
    llm_service = get_llm_service()
    return ToolCallingService(
        llm_service=llm_service
    )