class AppExceptions(Exception):
    """Base class for application-specific exceptions."""

    def __init__(self, code: str, message:str, status_code:int = 500):
        self.code = code
        self.message =message
        self.status_code = status_code

        super().__init__(self.message)

class LLMException(AppExceptions):
    """Raised when there is failure in the LLM Service."""

    def __init__(self, message :str = "the AI service is temporarily unavailable."):
        super().__init__(code = "LLM_ERROR",message = message, status_code = 502)

class LLMTimeoutException(AppExceptions):
    """Raise when the LLm takes too long to respond."""

    def __init__(self, message:str = "The AI service is taking too long to respond."):
        super().__init__(code = "LLM_TIMEOUT", message = message,status_code = 504)
        