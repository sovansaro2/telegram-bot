# src/ai_solver.py

import asyncio
import logging
import urllib.parse
from io import BytesIO
from typing import Optional

import aiohttp
from google import genai
from google.genai import types
from google.genai.errors import APIError, ServerError
from PIL import Image

from src.config import GEMINI_API_KEY


logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================

CODECOGS_LATEX_URL = "https://latex.codecogs.com/png.image"

# Primary → fallback models.
#
# All are current Gemini 3.x models intended for the current
# Gemini API generation flow.
#
# Primary:
#   gemini-3.6-flash
#
# Fallback:
#   gemini-3.7-flash
#   gemini-3.5-flash
#
# Do NOT put old Gemini 1.x / 2.0 models here.
FALLBACK_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash",
]


# Maximum number of characters returned to Telegram is handled
# by handlers.py. This value is only used to prevent obviously
# empty/invalid responses here.
MIN_RESPONSE_LENGTH = 1


# ============================================================
# Errors
# ============================================================

class HomeworkSolverError(RuntimeError):
    """
    Safe application-level error for the homework solver.

    message:
        Technical/internal error.

    user_message:
        Khmer message safe to show to Telegram users.
    """

    def __init__(
        self,
        message: str,
        user_message: str | None = None,
    ):
        super().__init__(message)

        self.user_message = user_message or (
            "❌ សេវា AI មិនអាចដោះស្រាយលំហាត់បាននៅពេលនេះទេ។ "
            "សូមព្យាយាមម្ដងទៀត។"
        )


# ============================================================
# System Prompt
# ============================================================

SYSTEM_PROMPT = """
អ្នកជាគ្រូបង្រៀនខ្មែរដែលមានបទពិសោធន៍
សម្រាប់សិស្សវិទ្យាល័យកម្ពុជា ថ្នាក់ទី ១០ ដល់ ១២។

គោលបំណងរបស់អ្នកគឺជួយសិស្សយល់ពីលំហាត់
មិនមែនគ្រាន់តែផ្តល់ចម្លើយចុងក្រោយប៉ុណ្ណោះទេ។

ច្បាប់សំខាន់ៗ៖

1. ឆ្លើយជាភាសាខ្មែរ។
2. ប្រើពាក្យសាមញ្ញ ងាយយល់ សមស្របសម្រាប់សិស្ស។
3. ដោះស្រាយជាជំហានៗ។
4. បង្ហាញរូបមន្ត និងការគណនាឱ្យច្បាស់។
5. ពិនិត្យចម្លើយចុងក្រោយមុននឹងបញ្ចប់។
6. ប្រសិនបើសំណួរមិនច្បាស់ ឬរូបភាពមិនអាចអានបាន
   ត្រូវប្រាប់អ្នកប្រើឱ្យផ្ញើរូបភាពដែលច្បាស់ជាងមុន។
7. កុំស្មានទិន្នន័យដែលមិនមាននៅក្នុងសំណួរ។
8. ប្រសិនបើមានទិន្នន័យមិនគ្រប់គ្រាន់
   ត្រូវប្រាប់ថាតើទិន្នន័យណាដែលខ្វះ។
9. កុំសរសេរសួរសុខទុក្ខ ឬសេចក្តីផ្តើមទូទៅ។
10. ចាប់ផ្តើមពីការវិភាគលំហាត់ភ្លាមៗ។

សម្រាប់លំហាត់គណិតវិទ្យា និងវិទ្យាសាស្ត្រ៖
- បង្ហាញ "គេឱ្យ" ប្រសិនបើសមស្រប។
- បង្ហាញ "រក" ប្រសិនបើសមស្រប។
- បង្ហាញរូបមន្ត។
- ដាក់តម្លៃចូលរូបមន្ត។
- គណនាជាជំហានៗ។
- បង្ហាញចម្លើយចុងក្រោយឱ្យច្បាស់។

សម្រាប់លំហាត់ដែលមានរូបភាព៖
- អានអត្ថបទ និងទិន្នន័យពីរូបភាពដោយប្រុងប្រយ័ត្ន។
- កុំបង្កើតលេខ ឬសញ្ញាដែលមិនមានក្នុងរូបភាព។
- ប្រសិនបើផ្នែកណាមួយមើលមិនច្បាស់
  ត្រូវប្រាប់អ្នកប្រើថាផ្នែកនោះមិនអាចអានបាន។

សម្រាប់គណិតវិទ្យាដែលត្រូវការរូបមន្ត៖
ប្រសិនបើត្រូវការរូបភាពរូបមន្ត សូមប្រើ format ខាងក្រោម៖

===LATEX_BLOCK===
LaTeX_CODE
===LATEX_BLOCK===

បន្ទាប់មកសរសេរការពន្យល់ជាភាសាខ្មែរ។

ប្រសិនបើមិនចាំបាច់ប្រើរូបភាពរូបមន្ត
កុំប្រើ LATEX_BLOCK។

ចំណាំ៖
- កុំប្រើ Markdown table ប្រសិនបើវាធ្វើឱ្យការអានលំបាកនៅក្នុង Telegram។
- ប្រើ Unicode math symbols ឬអត្ថបទធម្មតាសម្រាប់ការពន្យល់ខ្លីៗ។
- ប្រសិនបើត្រូវការរូបមន្តស្មុគស្មាញ អាចប្រើ LATEX_BLOCK។
"""


# ============================================================
# Utility: Extract HTTP/API status code
# ============================================================

def _get_error_code(error: Exception) -> int | None:
    """
    Safely extract an HTTP/API status code from a Google GenAI
    exception.

    Different SDK versions may expose it as:
        - code
        - status_code
        - response.status_code
    """

    code = getattr(error, "code", None)

    if code is None:
        code = getattr(error, "status_code", None)

    if code is None:
        response = getattr(error, "response", None)

        if response is not None:
            code = getattr(response, "status_code", None)
            if code is None:
                code = getattr(response, "code", None)

    try:
        return int(code) if code is not None else None
    except (TypeError, ValueError):
        return None


# ============================================================
# Utility: Convert arbitrary Gemini model name
# ============================================================

def _normalize_model_name(name: str) -> str:
    """
    Convert:
        models/gemini-3.6-flash

    to:
        gemini-3.6-flash
    """

    name = (name or "").strip()

    if name.startswith("models/"):
        name = name[len("models/"):]

    return name


# ============================================================
# Optional model discovery
# ============================================================

async def get_available_models(client) -> list[str]:
    """
    Ask Gemini API for models available to this API key.

    This function is intentionally defensive because the exact
    models.list() return shape can differ between SDK versions.

    Returns:
        List of model IDs supporting generateContent.
    """

    available: list[str] = []

    try:
        pager = await client.aio.models.list()

        async for model in pager:
            name = _normalize_model_name(
                getattr(model, "name", "") or ""
            )

            if not name:
                continue

            supported_actions = getattr(
                model,
                "supported_actions",
                None,
            )

            # Some SDK responses may expose supported_actions.
            if supported_actions:
                if "generateContent" not in supported_actions:
                    continue

            available.append(name)

    except Exception as e:
        logger.warning(
            "Unable to list Gemini models: %s",
            e,
        )

    return available


# ============================================================
# Build model order
# ============================================================

async def _get_model_candidates(client) -> list[str]:
    """
    Build a safe model order.

    We prefer the explicitly configured current models.

    If model discovery works, unavailable models are removed
    before making expensive generation calls.

    If discovery fails, we still try the configured models.
    """

    configured_models = [
        _normalize_model_name(model)
        for model in FALLBACK_MODELS
        if model
    ]

    configured_models = list(dict.fromkeys(configured_models))

    if not configured_models:
        raise HomeworkSolverError(
            "No Gemini models configured",
            "❌ មិនទាន់មាន Gemini model សម្រាប់ប្រើប្រាស់ទេ។",
        )

    available_models = await get_available_models(client)

    if not available_models:
        logger.info(
            "Gemini model discovery unavailable; "
            "using configured model list."
        )

        return configured_models

    available_set = set(available_models)

    compatible = [
        model
        for model in configured_models
        if model in available_set
    ]

    unavailable = [
        model
        for model in configured_models
        if model not in available_set
    ]

    if unavailable:
        logger.info(
            "Gemini models unavailable for this API key: %s",
            ", ".join(unavailable),
        )

    if compatible:
        return compatible

    # Important:
    # Do not immediately fail just because discovery does not
    # expose the expected model names. The API may return a
    # different representation depending on SDK/API version.
    logger.warning(
        "None of the configured models were confirmed by "
        "model discovery. Falling back to configured list."
    )

    return configured_models


# ============================================================
# Image preparation
# ============================================================

def _prepare_image_part(
    image_bytes: bytes,
    mime_type: str,
) -> types.Part:
    """
    Convert raw image bytes to a Gemini inline image Part.

    This avoids relying on PIL object conversion and explicitly
    tells Gemini the MIME type.
    """

    if not image_bytes:
        raise HomeworkSolverError(
            "Empty image bytes",
            "❌ រូបភាពទទេ។ សូមផ្ញើរូបភាពម្តងទៀត។",
        )

    safe_mime_type = (
        mime_type
        or "image/jpeg"
    ).lower().strip()

    # Telegram normally sends JPEG for photo messages.
    # Allow common image formats.
    allowed_types = {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
        "image/heic",
        "image/heif",
    }

    if safe_mime_type not in allowed_types:
        logger.warning(
            "Unsupported image MIME type '%s'; "
            "falling back to image/jpeg.",
            safe_mime_type,
        )
        safe_mime_type = "image/jpeg"

    # Validate the actual image before sending to Gemini.
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
    except Exception as e:
        logger.warning(
            "Invalid homework image: %s",
            e,
        )

        raise HomeworkSolverError(
            "Homework image could not be decoded",
            "❌ មិនអាចអានរូបថតលំហាត់បានទេ។ "
            "សូមផ្ញើរូបភាពដែលច្បាស់ និងត្រឹមត្រូវ។",
        ) from e

    return types.Part.from_bytes(
        data=image_bytes,
        mime_type=safe_mime_type,
    )


# ============================================================
# LaTeX Renderer
# ============================================================

async def render_latex_to_image(
    latex_str: str,
) -> bytes | None:
    """
    Render a LaTeX formula block using CodeCogs.

    Returns:
        PNG bytes or None if rendering fails.
    """

    try:
        latex_code = (latex_str or "").strip()

        if not latex_code:
            return None

        # Preserve the existing dark Telegram-friendly style.
        full_latex = (
            r"\dpi{300} "
            r"\bg{18181b} "
            r"\color{white} "
            + latex_code
        )

        encoded_query = urllib.parse.quote(
            full_latex,
            safe="",
        )

        url = f"{CODECOGS_LATEX_URL}?{encoded_query}"

        timeout = aiohttp.ClientTimeout(
            total=15
        )

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/131.0.0.0 "
                "Safari/537.36"
            )
        }

        async with aiohttp.ClientSession(
            timeout=timeout,
            headers=headers,
        ) as session:

            async with session.get(url) as response:

                if response.status != 200:
                    logger.warning(
                        "CodeCogs returned HTTP %s",
                        response.status,
                    )
                    return None

                image_bytes = await response.read()

                if not image_bytes:
                    logger.warning(
                        "CodeCogs returned an empty response."
                    )
                    return None

                return image_bytes

    except asyncio.TimeoutError:
        logger.warning(
            "CodeCogs LaTeX rendering timed out."
        )
        return None

    except aiohttp.ClientError as e:
        logger.warning(
            "CodeCogs network error: %s",
            e,
        )
        return None

    except Exception as e:
        logger.warning(
            "CodeCogs LaTeX rendering failed: %s",
            e,
        )
        return None


# ============================================================
# Clean Gemini response
# ============================================================

def _clean_response_text(
    raw_answer: str,
) -> str:
    """
    Basic cleanup while preserving the model's actual answer.
    """

    answer = (raw_answer or "").strip()

    if not answer:
        return ""

    # Remove accidental surrounding code fences only when the
    # entire response is wrapped.
    if answer.startswith("```") and answer.endswith("```"):
        lines = answer.splitlines()

        if len(lines) >= 3:
            answer = "\n".join(lines[1:-1]).strip()

    return answer


# ============================================================
# Parse answer + optional LaTeX
# ============================================================

def _parse_solution(
    raw_answer: str,
) -> tuple[str | None, str]:
    """
    Parse the custom:

        ===LATEX_BLOCK===
        ...
        ===LATEX_BLOCK===

    format.

    Returns:
        (latex_code, text_explanation)
    """

    answer = _clean_response_text(raw_answer)

    if not answer:
        return None, ""

    marker = "===LATEX_BLOCK==="

    if marker not in answer:
        return None, answer

    parts = answer.split(marker)

    # Expected:
    # parts[0] = LaTeX
    # parts[1] = explanation
    if len(parts) < 2:
        return None, answer

    latex_code = parts[0].strip()
    text_explanation = marker.join(
        parts[1:]
    ).strip()

    # Remove markdown code fences around LaTeX if model added them.
    if latex_code.startswith("```"):
        latex_lines = latex_code.splitlines()

        if latex_lines:
            latex_lines = latex_lines[1:]

        if latex_lines and latex_lines[-1].strip() == "```":
            latex_lines = latex_lines[:-1]

        latex_code = "\n".join(
            latex_lines
        ).strip()

    # Remove accidental second marker.
    text_explanation = text_explanation.replace(
        marker,
        "",
    ).strip()

    if not latex_code:
        latex_code = None

    return latex_code, text_explanation


# ============================================================
# Generate content with one model
# ============================================================

async def _generate_with_model(
    client,
    model_name: str,
    contents: list[object],
) -> str:
    """
    Generate a response with one Gemini model.

    Kept separate so solve_homework() can implement clean
    fallback behavior.
    """

    logger.info(
        "Trying Gemini model: %s",
        model_name,
    )

    response = await client.aio.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=8192,
        ),
    )

    raw_answer = (
        getattr(response, "text", None)
        or ""
    ).strip()

    if len(raw_answer) < MIN_RESPONSE_LENGTH:
        raise HomeworkSolverError(
            f"Gemini returned an empty response from {model_name}",
            "❌ AI មិនបានផ្តល់ចម្លើយទេ។ សូមព្យាយាមម្ដងទៀត។",
        )

    return raw_answer


# ============================================================
# Main Homework Solver
# ============================================================

async def solve_homework(
    question: str = "",
    image_bytes: Optional[bytes] = None,
    mime_type: str = "image/jpeg",
) -> tuple[bytes | None, str]:
    """
    Ask Gemini to solve homework.

    Parameters:
        question:
            User's text question or image caption.

        image_bytes:
            Optional Telegram image bytes.

        mime_type:
            MIME type of the image.

    Returns:
        (
            optional rendered formula PNG bytes,
            Khmer text explanation
        )

    This function intentionally keeps the same public interface
    as the previous ai_solver.py so handlers.py does not need
    to be rewritten.
    """

    # --------------------------------------------------------
    # 1. Validate API key
    # --------------------------------------------------------

    if not GEMINI_API_KEY:
        logger.error(
            "GEMINI_API_KEY is not configured."
        )

        raise HomeworkSolverError(
            "GEMINI_API_KEY is not configured",
            "❌ មុខងារ AI មិនទាន់បានកំណត់រចនាសម្ព័ន្ធទេ។ "
            "សូមជូនដំណឹងទៅ Admin។",
        )

    # --------------------------------------------------------
    # 2. Build prompt
    # --------------------------------------------------------

    prompt_text = (
        question.strip()
        if question and question.strip()
        else (
            "សូមជួយដោះស្រាយលំហាត់ក្នុងរូបភាពនេះ "
            "ជាជំហានៗឱ្យបានក្បោះក្បាយ និងត្រឹមត្រូវ "
            "ជាភាសាខ្មែរ។"
        )
    )

    # --------------------------------------------------------
    # 3. Build Gemini contents
    # --------------------------------------------------------

    contents: list[object] = []

    if image_bytes:
        image_part = _prepare_image_part(
            image_bytes=image_bytes,
            mime_type=mime_type,
        )

        # Image + text in one request.
        contents.append(image_part)

    contents.append(prompt_text)

    # --------------------------------------------------------
    # 4. Create Gemini client
    # --------------------------------------------------------

    client = None

    try:
        client = genai.Client(
            api_key=GEMINI_API_KEY,
        )

        # ----------------------------------------------------
        # 5. Determine model candidates
        # ----------------------------------------------------

        model_candidates = await _get_model_candidates(
            client
        )

        logger.info(
            "Gemini model candidates: %s",
            ", ".join(model_candidates),
        )

        if not model_candidates:
            raise HomeworkSolverError(
                "No usable Gemini models found",
                "❌ មិនមាន Gemini model ដែលអាចប្រើបានទេ។",
            )

        # ----------------------------------------------------
        # 6. Try models
        # ----------------------------------------------------

        last_code: int | None = None
        last_error: Exception | None = None

        for model_name in model_candidates:

            try:
                raw_answer = await _generate_with_model(
                    client=client,
                    model_name=model_name,
                    contents=contents,
                )

                # --------------------------------------------
                # 7. Parse solution
                # --------------------------------------------

                latex_code, text_explanation = (
                    _parse_solution(raw_answer)
                )

                if not text_explanation:
                    logger.warning(
                        "Model %s returned no usable "
                        "text explanation.",
                        model_name,
                    )

                    continue

                # --------------------------------------------
                # 8. Render optional LaTeX
                # --------------------------------------------

                rendered_img = None

                if latex_code:
                    rendered_img = (
                        await render_latex_to_image(
                            latex_code
                        )
                    )

                logger.info(
                    "Homework solved successfully "
                    "using Gemini model: %s",
                    model_name,
                )

                return (
                    rendered_img,
                    text_explanation,
                )

            except HomeworkSolverError as e:
                last_error = e
                last_code = _get_error_code(e)

                logger.warning(
                    "Homework solver error with model %s: %s",
                    model_name,
                    e,
                )

                # Continue to next model only if this is a
                # model/API failure.
                continue

            except (ServerError, APIError) as e:
                last_error = e
                last_code = _get_error_code(e)

                logger.warning(
                    "Gemini model %s failed "
                    "(HTTP/API code=%s): %s",
                    model_name,
                    last_code,
                    e,
                )

                # --------------------------------------------
                # IMPORTANT:
                #
                # 404 = model unavailable.
                # 401/403 = authentication/permission issue.
                #
                # We may try another configured model for
                # 404, but do not waste time retrying the
                # same model.
                # --------------------------------------------

                if last_code in (401, 403):
                    raise HomeworkSolverError(
                        f"Gemini authentication/permission error: "
                        f"{last_code}",
                        "❌ Gemini API Key មិនអាចប្រើប្រាស់បាន។ "
                        "សូមពិនិត្យ API Key និងការអនុញ្ញាតរបស់វា។",
                    ) from e

                if last_code == 429:
                    # Rate limit is generally not solved by
                    # trying several models immediately.
                    raise HomeworkSolverError(
                        "Gemini rate limit (429)",
                        "⚠️ សេវា AI កំពុងមានការស្នើសុំច្រើន។ "
                        "សូមរង់ចាំបន្តិច ហើយសាកល្បងម្ដងទៀត។",
                    ) from e

                # 404:
                # Continue to the next model.
                if last_code == 404:
                    logger.info(
                        "Model %s is unavailable (404). "
                        "Trying next configured model.",
                        model_name,
                    )
                    continue

                # 5xx:
                # Try next model once rather than immediately
                # failing.
                if last_code in (500, 502, 503, 504):
                    logger.info(
                        "Gemini server error for %s. "
                        "Trying next model.",
                        model_name,
                    )

                    await asyncio.sleep(0.5)
                    continue

                # Unknown API error.
                continue

            except Exception as e:
                last_error = e
                last_code = _get_error_code(e)

                logger.exception(
                    "Unexpected Gemini error with model %s "
                    "(code=%s)",
                    model_name,
                    last_code,
                )

                # Continue to next model.
                continue

        # ----------------------------------------------------
        # 9. All models failed
        # ----------------------------------------------------

        logger.error(
            "All Gemini fallback models failed. "
            "last_code=%s last_error=%s",
            last_code,
            last_error,
        )

        if last_code in (500, 502, 503, 504):
            raise HomeworkSolverError(
                "Gemini server unavailable",
                "⚠️ ម៉ាស៊ីនមេ AI កំពុងមានបញ្ហា "
                "ឬមានអ្នកប្រើប្រាស់ច្រើន។ "
                "សូមរង់ចាំបន្តិច ហើយសាកល្បងម្ដងទៀត។",
            )

        if last_code == 429:
            raise HomeworkSolverError(
                "Gemini rate limit",
                "⚠️ សេវា AI កំពុងមានការស្នើសុំច្រើន។ "
                "សូមរង់ចាំបន្តិច ហើយសាកល្បងម្ដងទៀត។",
            )

        if last_code in (401, 403):
            raise HomeworkSolverError(
                "Gemini API authentication failed",
                "❌ Gemini API Key មិនអាចប្រើបាន។ "
                "សូមពិនិត្យ API Key និងការកំណត់ API។",
            )

        raise HomeworkSolverError(
            "All configured Gemini models failed",
            "❌ មិនអាចដោះស្រាយលំហាត់បាននៅពេលនេះទេ។ "
            "សូមព្យាយាមម្ដងទៀត។",
        )

    finally:
        # ----------------------------------------------------
        # 10. Close async client
        # ----------------------------------------------------

        if client is not None:
            try:
                await client.aio.aclose()
            except Exception as e:
                logger.debug(
                    "Failed to close Gemini async client: %s",
                    e,
                )