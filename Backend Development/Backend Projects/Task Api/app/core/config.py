from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_NAME :  str = "Api Management System"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION:str = "Secure Api Management System"
    EVNIRONMENT:str = "developement"
    DEBUG: bool = True
    API_V1_PREFIX: str = "/api/v1"
    JWT_SECRETE_KEY: str = "23ghjbdsfn238eghjsadnSHAGSdHG837enjasbdsad8234:#@sadsd"
    JWT_ALGORITHM:str = "HS256 "
    ACCESS_TOKEN_EXPIRE_MINUTES:int = 30
    RATE_LIMIT_REQUEST: int = 100
    RATE_LIMIT_WINDOW_SECONDS:int = 60

    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding = "utf-8",
        case_sensitive = True,
    )
settings = Settings()



















