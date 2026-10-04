# chronicleweave/modules/llm_processor.py

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

# New unified Google Gen AI SDK (replaces the EOL google-generativeai package).
# We import lazily-friendly: the client is constructed inside _call_gemini_api,
# but the safety-setting enums are needed at LLMConfig instance time.
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

log = logging.getLogger(__name__)

MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPTS_DIR = MODULE_DIR / "prompts"

DEFAULT_NARRATIVE_PROMPT_FILENAME = "default_narrative_prompt.txt"
DEFAULT_SUMMARY_PROMPT_FILENAME = "default_summary_prompt.txt"
DEFAULT_CUMULATIVE_SUMMARY_PROMPT_FILENAME = "default_cumulative_summary_prompt.txt"


def _default_safety_settings() -> List[genai_types.SafetySetting]:
    block = genai_types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    return [
        genai_types.SafetySetting(category=genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=block),
        genai_types.SafetySetting(category=genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=block),
        genai_types.SafetySetting(category=genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=block),
        genai_types.SafetySetting(category=genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=block),
    ]


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for LLM processing, including cumulative summary logic."""
    api_provider: str = "gemini"
    gemini_api_key_env_var: str = "GEMINI_API_KEY"
    # Gemini 1.5 endpoints are being shut down; defaults bumped to 2.5 (mid-2025 release).
    model_name_summary: str = "gemini-2.5-flash"
    model_name_narrative: str = "gemini-2.5-pro"

    language: str = "en"
    narrative_prompt_filename: str = DEFAULT_NARRATIVE_PROMPT_FILENAME
    summary_prompt_filename: str = DEFAULT_SUMMARY_PROMPT_FILENAME
    cumulative_summary_prompt_filename: str = DEFAULT_CUMULATIVE_SUMMARY_PROMPT_FILENAME

    output_summary_filename: str = "llm_current_session_summary.txt"
    output_narrative_filename: str = "llm_current_session_narrative.txt"
    output_cumulative_meta_summary_filename: str = "llm_campaign_cumulative_summary.txt"

    max_output_tokens_summary: int = 1000
    max_output_tokens_narrative: int = 4000
    max_output_tokens_cumulative: int = 2000
    temperature: float = 0.7
    safety_settings: Optional[List[genai_types.SafetySetting]] = field(
        default_factory=_default_safety_settings
    )

    enable_summary: bool = True
    enable_narrative: bool = True
    enable_cumulative_summary: bool = True

    fallback_narrative_prompt: str = (
        "Narrate this session script: {current_session_script}\n"
        "Context from previous sessions: {cumulative_previous_sessions_summary}"
    )
    fallback_summary_prompt: str = (
        "Summarize this session script: {current_session_script}\n"
        "Context from previous sessions: {cumulative_previous_sessions_summary}"
    )
    fallback_cumulative_summary_prompt: str = (
        "Update campaign summary.\n"
        "Previous Overall Campaign Summary:\n{previous_cumulative_summary}\n\n"
        "Summary of the Latest Session to Integrate:\n{current_session_concise_summary}\n\n"
        "Produce the UPDATED AND INTEGRATED campaign summary below:"
    )

    initial_campaign_context: str = (
        "This is the very beginning of the campaign. No previous events have been recorded."
    )

    def __post_init__(self) -> None:
        if not self.model_name_summary:
            raise ValueError("LLMConfig: model_name_summary cannot be empty.")
        if not self.model_name_narrative:
            raise ValueError("LLMConfig: model_name_narrative cannot be empty.")
        if self.max_output_tokens_summary <= 0:
            raise ValueError("LLMConfig: max_output_tokens_summary must be positive.")
        if self.max_output_tokens_narrative <= 0:
            raise ValueError("LLMConfig: max_output_tokens_narrative must be positive.")
        if self.max_output_tokens_cumulative <= 0:
            raise ValueError("LLMConfig: max_output_tokens_cumulative must be positive.")
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError("LLMConfig: temperature must be between 0.0 and 2.0.")
        if not self.language:
            raise ValueError("LLMConfig: language cannot be empty.")


DEFAULT_LLM_CONFIG = LLMConfig()


def _load_api_key(env_var_name: str) -> Optional[str]:
    api_key = os.getenv(env_var_name)
    if not api_key:
        log.warning(f"API key environment variable '{env_var_name}' not found or is empty.")
    return api_key


def _load_prompt_template(
    base_prompts_dir: Path, language: str, prompt_filename: str, fallback_prompt: str
) -> str:
    prompt_file_path = base_prompts_dir / language / prompt_filename
    try:
        if prompt_file_path.is_file():
            log.info(f"Loading prompt from: {prompt_file_path}")
            return prompt_file_path.read_text(encoding="utf-8")
        log.warning(f"Prompt file not found: {prompt_file_path}. Using fallback prompt.")
        return fallback_prompt
    except OSError as e:
        log.error(f"Error reading prompt file {prompt_file_path}: {e}. Using fallback prompt.")
        return fallback_prompt


def _call_gemini_api(
    api_key: str,
    model_name: str,
    prompt_content: str,
    max_tokens: int,
    temperature: float,
    safety_settings: Optional[List[genai_types.SafetySetting]],
) -> str:
    """Call Gemini via the unified google-genai SDK and return generated text.

    Raises RuntimeError on API/blocked failures or empty text, ValueError on bad inputs.
    """
    log.info(f"Calling Gemini API with model: {model_name}")
    log.debug(f"Prompt content (first 200 chars): {prompt_content[:200]}...")

    config = genai_types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
        candidate_count=1,
        safety_settings=safety_settings,
    )

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt_content,
            config=config,
        )
    except genai_errors.APIError as e:
        # Single broad catch: status code distinguishes auth (401/403), quota (429),
        # bad request (400), not-found (404), server (5xx). Hand-rolling a branch per
        # case — like the old code did — is duplication for almost no signal.
        log.error(f"Gemini API error for '{model_name}' (code={getattr(e, 'code', 'n/a')}): {e}")
        raise RuntimeError(f"Gemini API call for '{model_name}' failed: {e}") from e
    except Exception as e:
        log.exception(f"Unexpected error during Gemini API call with model '{model_name}'")
        raise RuntimeError(f"Gemini API call for '{model_name}' failed unexpectedly: {e}") from e

    # Was the prompt blocked outright?
    if not response.candidates:
        feedback = getattr(response, "prompt_feedback", None)
        block_reason = getattr(feedback, "block_reason", None)
        reason_name = getattr(block_reason, "name", str(block_reason) if block_reason else "unknown")
        log.error(f"Gemini '{model_name}' returned no candidates. Block reason: {reason_name}")
        raise RuntimeError(f"Gemini API '{model_name}' produced no candidates (reason: {reason_name}).")

    text = response.text or ""
    if not text.strip():
        candidate = response.candidates[0]
        finish = getattr(candidate, "finish_reason", None)
        finish_name = getattr(finish, "name", str(finish) if finish else "UNKNOWN")
        log.error(f"Gemini '{model_name}' returned empty text. Finish reason: {finish_name}")
        raise RuntimeError(f"Gemini '{model_name}' returned empty text (finish reason: {finish_name}).")
    return text


def _require_text(text: str, what: str) -> str:
    """An empty generation is a failure: raise before any output file is written or overwritten."""
    if not (text or "").strip():
        raise RuntimeError(f"the model returned empty text for the {what}")
    return text


if TYPE_CHECKING:
    from chronicleweave.pipeline import PipelineConfig  # noqa: F401
else:
    @dataclass
    class PipelineConfig:
        base_path: Path = field(default_factory=Path.cwd)


def process_with_llm(
    final_script_file: Path,
    output_dir: Path,
    llm_config: LLMConfig,
    pipeline_config: "PipelineConfig",
) -> None:
    log.info("Starting LLM processing with cumulative summary logic...")
    session_base_path = getattr(pipeline_config, "base_path", Path("."))
    session_name = session_base_path.name
    log.info(f"LLM processing for session: {session_name} (Language: {llm_config.language})")

    api_key = _load_api_key(llm_config.gemini_api_key_env_var)
    if not api_key:
        raise ValueError(f"Missing Gemini API key (env var: {llm_config.gemini_api_key_env_var})")
    failures: List[str] = []

    try:
        current_session_script_content = final_script_file.read_text(encoding="utf-8")
        if not current_session_script_content.strip():
            log.warning(f"Input script '{final_script_file}' is empty. LLM results may be poor.")
    except FileNotFoundError:
        log.error(f"Final script file for LLM input not found: {final_script_file}")
        raise
    except OSError as e:
        log.exception(f"Error reading final script file {final_script_file}")
        raise IOError(f"Could not read final script file: {final_script_file}") from e

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load Previous Cumulative Summary ---
    cumulative_summary_file_path = output_dir / llm_config.output_cumulative_meta_summary_filename
    if cumulative_summary_file_path.is_file():
        try:
            previous_cumulative_summary_text = cumulative_summary_file_path.read_text(encoding="utf-8")
            log.info(f"Loaded previous cumulative summary from: {cumulative_summary_file_path}")
        except OSError as e:
            log.warning(
                f"Could not read previous cumulative summary {cumulative_summary_file_path}: {e}. "
                "Using initial context."
            )
            previous_cumulative_summary_text = llm_config.initial_campaign_context
    else:
        log.info(f"No previous cumulative summary at {cumulative_summary_file_path}. Using initial context.")
        previous_cumulative_summary_text = llm_config.initial_campaign_context

    current_session_concise_summary_text = ""

    # --- 2. Generate Current Session Concise Summary (Prompt B) ---
    if llm_config.enable_summary:
        log.info(f"Generating current session concise summary (model: {llm_config.model_name_summary})...")
        prompt_template_b = _load_prompt_template(
            DEFAULT_PROMPTS_DIR, llm_config.language,
            llm_config.summary_prompt_filename, llm_config.fallback_summary_prompt,
        )
        prompt_b_params = {
            "current_session_script": current_session_script_content,
            "cumulative_previous_sessions_summary": previous_cumulative_summary_text,
        }
        try:
            formatted_prompt_b = prompt_template_b.format(**prompt_b_params)
            current_session_concise_summary_text = _require_text(_call_gemini_api(
                api_key, llm_config.model_name_summary, formatted_prompt_b,
                llm_config.max_output_tokens_summary, llm_config.temperature, llm_config.safety_settings,
            ), "session summary")
            (output_dir / llm_config.output_summary_filename).write_text(
                current_session_concise_summary_text, encoding="utf-8"
            )
            log.info(f"Current session concise summary saved to: {output_dir / llm_config.output_summary_filename}")
        except KeyError as e:
            log.error(f"Prompt B is missing an expected key for formatting: {e}")
            failures.append(f"summary prompt: missing key {e}")
        except Exception as e:
            log.error(f"Failed to generate or save current session concise summary: {e}")
            failures.append(f"summary: {e}")

    # --- 3. Generate Current Session Narrative (Prompt A) ---
    if llm_config.enable_narrative:
        log.info(f"Generating current session narrative (model: {llm_config.model_name_narrative})...")
        prompt_template_a = _load_prompt_template(
            DEFAULT_PROMPTS_DIR, llm_config.language,
            llm_config.narrative_prompt_filename, llm_config.fallback_narrative_prompt,
        )
        prompt_a_params = {
            "current_session_script": current_session_script_content,
            "cumulative_previous_sessions_summary": previous_cumulative_summary_text,
        }
        try:
            formatted_prompt_a = prompt_template_a.format(**prompt_a_params)
            narrative_text = _require_text(_call_gemini_api(
                api_key, llm_config.model_name_narrative, formatted_prompt_a,
                llm_config.max_output_tokens_narrative, llm_config.temperature, llm_config.safety_settings,
            ), "session narrative")
            (output_dir / llm_config.output_narrative_filename).write_text(narrative_text, encoding="utf-8")
            log.info(f"Current session narrative saved to: {output_dir / llm_config.output_narrative_filename}")
        except KeyError as e:
            log.error(f"Prompt A is missing an expected key for formatting: {e}")
            failures.append(f"narrative prompt: missing key {e}")
        except Exception as e:
            log.error(f"Failed to generate or save current session narrative: {e}")
            failures.append(f"narrative: {e}")

    # --- 4. Generate NEW Cumulative Meta-Summary (Prompt C) ---
    if llm_config.enable_cumulative_summary:
        if not current_session_concise_summary_text.strip():
            log.warning(
                "Skipping cumulative meta-summary generation: current session's concise "
                "summary is empty or its generation failed."
            )
        else:
            log.info(f"Generating new cumulative campaign summary (model: {llm_config.model_name_summary})...")
            prompt_template_c = _load_prompt_template(
                DEFAULT_PROMPTS_DIR, llm_config.language,
                llm_config.cumulative_summary_prompt_filename,
                llm_config.fallback_cumulative_summary_prompt,
            )
            prompt_c_params = {
                "previous_cumulative_summary": previous_cumulative_summary_text,
                "current_session_concise_summary": current_session_concise_summary_text,
            }
            try:
                formatted_prompt_c = prompt_template_c.format(**prompt_c_params)
                new_cumulative_meta_summary_text = _require_text(_call_gemini_api(
                    api_key, llm_config.model_name_summary, formatted_prompt_c,
                    llm_config.max_output_tokens_cumulative, llm_config.temperature, llm_config.safety_settings,
                ), "cumulative campaign summary")
                if cumulative_summary_file_path.is_file():
                    backup_path = cumulative_summary_file_path.with_suffix(
                        cumulative_summary_file_path.suffix + ".bak"
                    )
                    try:
                        shutil.copy2(cumulative_summary_file_path, backup_path)
                        log.info(f"Backed up existing cumulative summary to: {backup_path}")
                    except OSError as e_backup:
                        log.warning(
                            f"Could not back up existing cumulative summary {cumulative_summary_file_path}: {e_backup}"
                        )
                cumulative_summary_file_path.write_text(new_cumulative_meta_summary_text, encoding="utf-8")
                log.info(f"New cumulative campaign summary saved to: {cumulative_summary_file_path}")
            except KeyError as e:
                log.error(f"Prompt C is missing an expected key for formatting: {e}")
                failures.append(f"cumulative summary prompt: missing key {e}")
            except Exception as e:
                log.error(f"Failed to generate or save new cumulative campaign summary: {e}")
                failures.append(f"cumulative summary: {e}")

    if failures:
        raise RuntimeError("LLM processing failed: " + "; ".join(failures))
    log.info("LLM processing with cumulative summary logic finished.")
