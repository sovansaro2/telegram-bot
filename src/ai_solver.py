import asyncio
import io
import logging
import re
import urllib.parse
from html import escape
from io import BytesIO
from typing import Optional

import aiohttp
from google import genai
from google.genai import types
from google.genai.errors import APIError, ServerError
from PIL import Image

from src.config import GEMINI_API_KEY

logger = logging.getLogger(__name__)

CODECOGS_LATEX_URL = "[https://latex.codecogs.com/png.image](https://latex.codecogs.com/png.image)"

FALLBACK_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.1-pro-preview",
    "gemini-flash-latest",
]


class HomeworkSolverError(RuntimeError):
    """Safe, user-facing failure raised when Gemini cannot solve homework."""

    def __init__(self, message: str, user_message: str | None = None):
        super().__init__(message)
        self.user_message = user_message or (
            "❌ សេវា AI មិនអាចដោះស្រាយលំហាត់បាននៅពេលនេះទេ។ "
            "សូមព្យាយាមម្តងទៀតនៅពេលក្រោយ។"
        )


SYSTEM_PROMPT = (
    "អ្នកជាគ្រូបង្រៀនខ្មែរដែលមានបទពិសោធន៍ សម្រាប់សិស្សវិទ្យាល័យកម្ពុជា ថ្នាក់ទី ១០ ដល់ ១២។ "
    "ឆ្លើយជាភាសាខ្មែរ ឲ្យច្បាស់លាស់ ត្រឹមត្រូវ និងសមស្របសម្រាប់សិស្សថ្នាក់ទី ១០ ដល់ ១២។ "
    "កុំសរសេរសួរសុខទុក្ខ ឬសេចក្តីផ្តើមណាមួយឡើយ។ ហាមប្រើពាក្យ សួស្ដី ឬ ជំរាបសួរ។ "
    "ចាប់ផ្តើមចម្លើយភ្លាមៗ ដោយបន្ទាត់ទីមួយត្រូវតែជា៖ 🎯 ចម្លើយ៖ [ចម្លើយចុងក្រោយ]។ "
    "បន្ទាប់មកដាក់បន្ទាត់៖ 📌 នេះជាប្រមាណវិធី៖ ហើយបង្ហាញការគណនា និងការពន្យល់ជាជំហានៗ។ "
    "រាល់ជំហានគណិតវិទ្យាដែលមានការគណនា ត្រូវដាក់ក្នុង Markdown code block ដោយប្រើ ```text នៅដើម និង ``` នៅចុង។ "
    "នៅចុងបំផុត ត្រូវសរសេរ ✅ ចម្លើយចុងក្រោយ៖ ហើយដាក់ [ចម្លើយចុងក្រោយ] នៅក្នុង Markdown code block ```text ផ្ទាល់ខ្លួនមួយ។ "
    "កុំដាក់អត្ថបទសួរសុខទុក្ខ ឬសេចក្តីផ្តើមមុនបន្ទាត់ 🎯 និងកុំដាក់អត្ថបទក្រោយ code block ចុងក្រោយ។ "
    "ហាមប្រើ LaTeX ក្នុងផ្នែកពន្យល់ (ផ្នែកទី២)។ ប្រើតែអក្សរធម្មតា និងតួអក្សរគណិតវិទ្យា Unicode (√, →, ∞, ×, ≠, ², ³) ប៉ុណ្ណោះ។ "
    "ត្រូវបែងចែកចម្លើយជាពីរផ្នែក ដោយប្រើបន្ទាត់ ===LATEX_BLOCK=== តែមួយគត់។ "
    "ផ្នែកទី១ មុនសញ្ញាបែងចែក ត្រូវមានតែប្លុករូបមន្តគណិតវិទ្យា LaTeX ពេញលេញសម្រាប់ Render ជារូបភាព។ "
    "ផ្នែកទី២ ក្រោយសញ្ញាបែងចែក ត្រូវមានការពន្យល់ជាភាសាខ្មែរ ជាជំហានៗ។"
)


def format_explanation_to_html(raw_text: str) -> str:
    """បម្លែងអត្ថបទ Markdown ទៅជា Telegram HTML ដែលមានប្រអប់ចុច Copy ស្អាត"""
    if not raw_text:
        return ""

    pattern = re.compile(r"```(?:[a-zA-Z0-9_-]+)?\n?(.*?)```", re.DOTALL)
    parts = []
    last_idx = 0

    for match in pattern.finditer(raw_text):
        start, end = match.span()
        non_code = raw_text[last_idx:start]
        if non_code:
            escaped_non_code = escape(non_code)
            escaped_non_code = re.sub(
                r"(🎯\s*ចម្លើយ៖?[^\n]*)", r"<b>\1</b>", escaped_non_code
            )
            escaped_non_code = re.sub(
                r"(📌\s*នេះជាប្រមាណវិធី៖?)", r"<b>\1</b>", escaped_non_code
            )
            escaped_non_code = re.sub(
                r"(✅\s*ចម្លើយចុងក្រោយ៖?)", r"<b>\1</b>", escaped_non_code
            )
            parts.append(escaped_non_code)

        code_content = match.group(1).strip()
        escaped_code = escape(code_content)
        parts.append(f"<pre><code>{escaped_code}</code></pre>")
        last_idx = end

    remaining = raw_text[last_idx:]
    if remaining:
        escaped_rem = escape(remaining)
        escaped_rem = re.sub(r"(🎯\s*ចម្លើយ៖?[^\n]*)", r"<b>\1</b>", escaped_rem)
        escaped_rem = re.sub(r"(📌\s*នេះជាប្រមាណវិធី៖?)", r"<b>\1</b>", escaped_rem)
        escaped_rem = re.sub(r"(✅\s*ចម្លើយចុងក្រោយ៖?)", r"<b>\1</b>", escaped_rem)
        parts.append(escaped_rem)

    return "".join(parts)


async def render_latex_to_image(latex_str: str) -> bytes | None:
    """Render a LaTeX formula block with CodeCogs into a Telegram-ready PNG."""
    try:
        latex_code = latex_str.strip()
        if not latex_code:
            return None

        full_latex = f"\\dpi{{300}} \\bg{{18181b}} \\color{{white}} {latex_code}"
        encoded_query = urllib.parse.quote(full_latex)
        url = f"{CODECOGS_LATEX_URL}?{encoded_query}"

        timeout = aiohttp.ClientTimeout(total=15)
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36"
            )
        }
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.get(url) as response:
                if response.status != 200:
                    logger.warning("CodeCogs returned HTTP %s", response.status)
                    return None
                image_bytes = await response.read()
                return image_bytes if image_bytes else None
    except Exception as e:
        logger.warning("CodeCogs LaTeX rendering failed: %s", e)
        return None


async def solve_homework(
    question: str = "",
    image_bytes: Optional[bytes] = None,
    mime_type: str = "image/jpeg",
) -> tuple[bytes | None, str]:
    """Ask Gemini and return an optional formula image with Khmer explanation."""
    if not GEMINI_API_KEY:
        raise HomeworkSolverError(
            "GEMINI_API_KEY is not configured",
            "❌ មុខងារ AI មិនទាន់បានកំណត់រចនាសម្ព័ន្ធទេ។ សូមជូនដំណឹងទៅ Admin។",
        )

    contents: list[object] = []

    if image_bytes:
        try:
            image = Image.open(BytesIO(image_bytes))
            contents.append(image)
        except Exception as e:
            logger.warning("Invalid homework image: %s", e)
            raise HomeworkSolverError(
                "Homework image could not be decoded",
                "❌ មិនអាចអានរូបថតលំហាត់បានទេ។ សូមផ្ញើរូបភាពដែលច្បាស់ និងត្រឹមត្រូវ។",
            ) from e

    prompt_text = (
        question.strip()
        if question.strip()
        else "សូមជួយដោះស្រាយលំហាត់ក្នុងរូបភាពនេះជាជំហានៗឱ្យបានក្បោះក្បាយ និងត្រឹមត្រូវជាភាសាខ្មែរ។"
    )
    contents.append(prompt_text)

    client = None
    last_status = None

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)

        for model_name in FALLBACK_MODELS:
            try:
                logger.info("Trying Gemini model: %s", model_name)
                response = await client.aio.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT
                    ),
                )
                raw_answer = (response.text or "").strip()
                if not raw_answer:
                    continue

                if "===LATEX_BLOCK===" in raw_answer:
                    latex_code, text_explanation = raw_answer.split(
                        "===LATEX_BLOCK===", 1
                    )
                    latex_code = latex_code.strip()
                    text_explanation = text_explanation.strip()
                    if latex_code.startswith("```"):
                        latex_code = latex_code.split("\n", 1)[-1]
                        latex_code = latex_code.rsplit("```", 1)[0].strip()
                    rendered_img = await render_latex_to_image(latex_code)
                    formatted_html = format_explanation_to_html(text_explanation)
                    return rendered_img, formatted_html

                formatted_html = format_explanation_to_html(raw_answer)
                return None, formatted_html

            except (ServerError, APIError, Exception) as e:
                code = (
                    getattr(e, "code", None)
                    or getattr(e, "status_code", None)
                    or 500
                )
                last_status = code
                logger.warning(
                    "Model %s failed with code %s (%s); switching to next model...",
                    model_name,
                    code,
                    e,
                )
                await asyncio.sleep(1)
                continue

        if last_status in (503, 500):
            raise HomeworkSolverError(
                "Gemini server overloaded",
                "⚠️ ម៉ាស៊ីនមេ AI កំពុងមានអ្នកប្រើប្រាស់ច្រើន (High Demand)។ "
                "សូមរង់ចាំប្រហែល ១ នាទី រួចសាកល្បងម្ដងទៀត!",
            )
        if last_status == 429:
            raise HomeworkSolverError(
                "Gemini rate limit",
                "⚠️ សេវា AI កំពុងមានការស្នើសុំច្រើន។ សូមរង់ចាំបន្តិច រួចសាកល្បងម្ដងទៀត។",
            )

        raise HomeworkSolverError(
            "All fallback models failed",
            "❌ មិនអាចដោះស្រាយលំហាត់បានទេនៅពេលនេះ។ សូមព្យាយាមម្តងទៀត។",
        )

    finally:
        if client is not None:
            try:
                await client.aio.aclose()
            except Exception:
                pass