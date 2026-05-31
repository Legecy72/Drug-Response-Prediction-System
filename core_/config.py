from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"


class Settings(BaseSettings):
    app_name: str = "Drug Response Prediction API"
    regression_model_name: str = "lightgbm"
    classification_model_name: str = "xgboost"
    cors_origins: str = "*"

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def regression_dir(self) -> Path:
        return MODELS_DIR / "regression"

    @property
    def classification_dir(self) -> Path:
        return MODELS_DIR / "classification"

    @property
    def allowed_origins(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()

