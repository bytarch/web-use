import base64
import os
import time
from pathlib import Path
from typing import Dict, Optional
import requests

from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI, ChatOpenAI
import gradio as gr

from .llm import DeepSeekR1ChatOpenAI, DeepSeekR1ChatOllama

PROVIDER_DISPLAY_NAMES = {
    "bytarch": "bytarch",

}

def get_llm_model(provider: str, **kwargs):
    """
    获取LLM 模型
    :param provider: 模型类型
    :param kwargs:
    :return:
    """
    if provider not in ["ollama"]:
        env_var = "GOOGLE_API_KEY" if provider == "gemini" else f"{provider.upper()}_API_KEY"
        api_key = kwargs.get("api_key", "")
        if not api_key:
            handle_api_key_error(provider, env_var)
        kwargs["api_key"] = api_key

    if provider == "bytarch":
        if not kwargs.get("base_url", ""):
            base_url = "https://bytarch.taile6b163.ts.net/back-end/v1"
        else:
            base_url = kwargs.get("base_url")

        return ChatOpenAI(
            model=kwargs.get("model_name", "gpt-4o"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=base_url,
            api_key=api_key,
        )
   
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    
def get_model_names():
    url = "https://bytarch.taile6b163.ts.net/back-end/v1/models"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # Extract models owned by "bytarch" and explicitly exclude "deepresearch"
        models = [
            model["id"] for model in data.get("data", []) 
            if model.get("owned_by") == "bytarch" and model.get("owned_by") != "deepresearch"
        ]

        return {"bytarch": models}

    except requests.RequestException as e:
        print(f"Error fetching models: {e}")
        return {"bytarch": []}  # Return empty list in case of failure

# Predefined model names for common providers
model_names = get_model_names()

# Callback to update the model name dropdown based on the selected provider
def update_model_dropdown(llm_provider, api_key=None, base_url=None):
    """
    Update the model name dropdown with predefined models for the selected provider.
    """
    # Use API keys from .env if not provided
    if not api_key:
        api_key = os.getenv(f"{llm_provider.upper()}_API_KEY", "")
    if not base_url:
        base_url = os.getenv(f"{llm_provider.upper()}_BASE_URL", "")

    # Use predefined models for the selected provider
    if llm_provider in model_names:
        return gr.Dropdown(choices=model_names[llm_provider], value=model_names[llm_provider][0], interactive=True)
    else:
        return gr.Dropdown(choices=[], value="", interactive=True, allow_custom_value=True)

def handle_api_key_error(provider: str, env_var: str):
    """
    Handles the missing API key error by raising a gr.Error with a clear message.
    """
    provider_display = PROVIDER_DISPLAY_NAMES.get(provider, provider.upper())
    raise gr.Error(
        f"💥 {provider_display} API KEY not found or invalid! \n\n"
        f"Enter in a valid BytArch API Key"
    )

def encode_image(img_path):
    if not img_path:
        return None
    with open(img_path, "rb") as fin:
        image_data = base64.b64encode(fin.read()).decode("utf-8")
    return image_data


def get_latest_files(directory: str, file_types: list = ['.webm', '.zip']) -> Dict[str, Optional[str]]:
    """Get the latest recording and trace files"""
    latest_files: Dict[str, Optional[str]] = {ext: None for ext in file_types}
    
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        return latest_files

    for file_type in file_types:
        try:
            matches = list(Path(directory).rglob(f"*{file_type}"))
            if matches:
                latest = max(matches, key=lambda p: p.stat().st_mtime)
                # Only return files that are complete (not being written)
                if time.time() - latest.stat().st_mtime > 1.0:
                    latest_files[file_type] = str(latest)
        except Exception as e:
            print(f"Error getting latest {file_type} file: {e}")
            
    return latest_files
async def capture_screenshot(browser_context):
    """Capture and encode a screenshot"""
    # Extract the Playwright browser instance
    playwright_browser = browser_context.browser.playwright_browser  # Ensure this is correct.

    # Check if the browser instance is valid and if an existing context can be reused
    if playwright_browser and playwright_browser.contexts:
        playwright_context = playwright_browser.contexts[0]
    else:
        return None

    # Access pages in the context
    pages = None
    if playwright_context:
        pages = playwright_context.pages

    # Use an existing page or create a new one if none exist
    if pages:
        active_page = pages[0]
        for page in pages:
            if page.url != "about:blank":
                active_page = page
    else:
        return None

    # Take screenshot
    try:
        screenshot = await active_page.screenshot(
            type='jpeg',
            quality=75,
            scale="css"
        )
        encoded = base64.b64encode(screenshot).decode('utf-8')
        return encoded
    except Exception as e:
        return None
