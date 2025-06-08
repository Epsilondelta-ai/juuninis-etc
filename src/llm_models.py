from os import getenv

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI


def get_api_key(provider: str) -> str:
    if provider == "google":
        return getenv("GOOGLE_API_KEY")
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def get_llm(provider: str, model: str, api_key: str, **kwargs) -> BaseChatModel:
    if provider == "google":
        return ChatGoogleGenerativeAI(
            model=model,
            api_key=api_key,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")
