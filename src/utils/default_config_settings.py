import os
import pickle
import uuid
import gradio as gr


def default_config():
    """Prepare the default configuration"""
    current_directory = os.getcwd()  # Get the current working directory
    return {
        "agent_type": "custom",
        "max_steps": 100,
        "max_actions_per_step": 10,
        "use_vision": False,
        "tool_calling_method": "auto",
        "llm_provider": "bytarch",
        "llm_model_name": "gpt-4o",
        "llm_temperature": 1.0,
        "llm_base_url": "https://bytarch.taile6b163.ts.net/back-end/v1",
        "llm_api_key": "",
        "use_own_browser": os.getenv("CHROME_PERSISTENT_SESSION", "true").lower() == "true",
        "keep_browser_open": True,
        "headless": False,
        "disable_security": True,
        "enable_recording": False,
        "window_w": 1280,
        "window_h": 1100,
        "save_recording_path": os.path.join(current_directory, "recordings"),  # Use current directory
        "save_trace_path": os.path.join(current_directory, "recordings", "traces"),  # Use current directory
        "save_agent_history_path": os.path.join(current_directory, "recordings", "agent_history"),  # Use current directory
        "task": "go to google.com and search for bytarch llc",
    }


def load_config_from_file(config_file):
    """Load settings from a UUID.pkl file."""
    try:
        with open(config_file, 'rb') as f:
            settings = pickle.load(f)
        return settings
    except Exception as e:
        return f"Error loading configuration: {str(e)}"


def save_config_to_file(settings, save_dir="./tmp/webui_settings"):
    """Save the current settings to a UUID.pkl file with a UUID name."""
    os.makedirs(save_dir, exist_ok=True)
    config_file = os.path.join(save_dir, f"{uuid.uuid4()}.pkl")
    with open(config_file, 'wb') as f:
        pickle.dump(settings, f)
    return f"Configuration saved to {config_file}"


def save_current_config(*args):
    current_config = {
        "agent_type": args[0],
        "max_steps": args[1],
        "max_actions_per_step": args[2],
        "use_vision": args[3],
        "tool_calling_method": args[4],
        "llm_provider": args[5],
        "llm_model_name": args[6],
        "llm_temperature": args[7],
        "llm_base_url": args[8],
        "llm_api_key": args[9],
        "use_own_browser": args[10],
        "keep_browser_open": args[11],
        "headless": args[12],
        "disable_security": args[13],
        "enable_recording": args[14],
        "window_w": args[15],
        "window_h": args[16],
        "save_recording_path": args[17],
        "save_trace_path": args[18],
        "save_agent_history_path": args[19],
        "task": args[20],
    }
    return save_config_to_file(current_config)


def update_ui_from_config(config_file):
    if config_file is not None:
        loaded_config = load_config_from_file(config_file.name)
        if isinstance(loaded_config, dict):
            current_directory = os.getcwd()  # Get current directory
             # Update paths if necessary
            save_recording_path = os.path.join(current_directory, "recordings")
            save_trace_path = os.path.join(current_directory, "recordings", "traces")
            save_agent_history_path = os.path.join(current_directory, "recordings", "agent_history")

            return (
                gr.update(value=loaded_config.get("agent_type", "custom")),
                gr.update(value=loaded_config.get("max_steps", 100)),
                gr.update(value=loaded_config.get("max_actions_per_step", 10)),
                gr.update(value=loaded_config.get("use_vision", False)),
                gr.update(value=loaded_config.get("tool_calling_method", True)),
                gr.update(value=loaded_config.get("llm_provider", "bytarch")),
                gr.update(value=loaded_config.get("llm_model_name", "02::deminji-computer-use")),
                gr.update(value=loaded_config.get("llm_temperature", 1.0)),
                gr.update(value=loaded_config.get("llm_base_url", "")),
                gr.update(value=loaded_config.get("llm_api_key", "")),
                gr.update(value=loaded_config.get("use_own_browser", True)),
                gr.update(value=loaded_config.get("keep_browser_open", False)),
                gr.update(value=loaded_config.get("headless", False)),
                gr.update(value=loaded_config.get("disable_security", True)),
                gr.update(value=loaded_config.get("enable_recording", True)),
                gr.update(value=loaded_config.get("window_w", 1280)),
                gr.update(value=loaded_config.get("window_h", 1100)),
                gr.update(value=save_recording_path),
                gr.update(value=save_trace_path),
                gr.update(value=save_agent_history_path),
                gr.update(value=loaded_config.get("task", "")),
                "Configuration loaded successfully."
            )
        else:
            return (
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), "Error: Invalid configuration file."
            )
    return (
        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
        gr.update(), "No file selected."
    )
