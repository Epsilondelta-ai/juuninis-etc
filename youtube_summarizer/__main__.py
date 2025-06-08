from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from youtube_summarizer.prompt import SYSTEM_PROMPT

import uvicorn
from fastapi import FastAPI, Request

from src.llm_models import get_llm, get_api_key

load_dotenv()

app = FastAPI()


@app.post("/")
async def summarize(request: Request):
    data = await request.json()
    youtube_url = data.get("youtube_url")
    llm_provider = data.get("llm_provider")
    llm_model = data.get("llm_model")

    return {"summary": youtube_summarize(youtube_url, llm_provider, llm_model)}


def youtube_summarize(
    youtube_url: str,
    llm_provider: str = "google",
    llm_model: str = "gemini-2.5-pro-preview-06-05",
) -> str:
    llm_api_key = get_api_key(provider=llm_provider)

    llm = get_llm(provider=llm_provider, model=llm_model, api_key=llm_api_key)

    response = llm.invoke(
        input=[
            SystemMessage(SYSTEM_PROMPT),
            HumanMessage(
                content=[
                    {
                        "type": "media",
                        "mime_type": "video/youtube",
                        "file_uri": youtube_url,
                    }
                ],
            ),
        ],
        generation_config=dict(thinking_config={"thinking_budget": 100}),
    )

    return response.content


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)
