import asyncio
import io
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

CODECOGS_LATEX_URL = "https://latex.codecogs.com/png.image"

# បញ្ជីម៉ូដែលដែលមានស្ថេរភាព និងគាំទ្រ Vision ច្បាស់បំផុត
FALLBACK_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-2.5-pro",
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
    "រាល់ជំហានគណិតវិទ្យាដែលមានច្រើនបន្ទាត់ ត្រូវដាក់ក្នុង Markdown code block ប្រភេទ text ដោយប្រើ ```text នៅដើម និង ``` នៅចុង ដើម្បីឲ្យអ្នកប្រើអាចចុចចម្លងបានងាយ។ "
    "នៅចុងបំផុត ត្រូវសរសេរ ✅ ចម្លើយចុងក្រោយ៖ ហើយដាក់ [ចម្លើយចុងក្រោយ] នៅក្នុង Markdown code block ប្រភេទ text ផ្ទាល់ខ្លួនមួយ។ "
    "កុំដាក់អត្ថបទសួរសុខទុក្ខ ឬសេចក្តីផ្តើមមុនបន្ទាត់ 🎯 និងកុំដាក់អត្ថបទក្រោយ code block ចុងក្រោយ។ "
    "ហាមប្រើ LaTeX ជាដាច់ខាត រួមទាំង $$, $ និងពាក្យបញ្ជា \\lim, \\frac, \\sqrt, \\to, \\infty ឬទម្រង់ LaTeX ផ្សេងទៀត។ "
    "ប្រើតែអក្សរធម្មតាដែលអានងាយ និងតួអក្សរគណិតវិទ្យា Unicode ប៉ុណ្ណោះ។ "
    "សម្រាប់ឫសការេ ប្រើ √ ដូចជា √(4x² + x + 2)។ សម្រាប់ប្រភាគ ប្រើ / ដូចជា 1/x។ "
    "សម្រាប់លីមីត ប្រើ → ដូចជា lim (x → +∞) ហើយប្រើ ∞ សម្រាប់អនន្ត។ សម្រាប់គុណ ប្រើ × និងសម្រាប់មិនស្មើ ប្រើ ≠។ "
    "សម្រាប់ស្វ័យគុណ ប្រើលេខលើ Unicode ដូចជា x² និង x³។ "
    "បើសំណួរមិនច្បាស់ សូមបង្ហាញចំណុចដែលមិនច្បាស់នៅក្នុងជំហានដោះស្រាយ ហើយកុំប្រឌិតទិន្នន័យ។ "
    "ត្រូវបែងចែកចម្លើយជាពីរផ្នែក ដោយប្រើបន្ទាត់ ===LATEX_BLOCK=== តែមួយគត់។ "
    "ផ្នែកទី១ មុនសញ្ញាបែងចែក ត្រូវមានតែប្លុករូបមន្តគណិតវិទ្យា LaTeX ពេញលេញ និងមានជំហានជាច្រើន "
    "ដែលអាចយកទៅ Render ជារូបភាពបាន។ កុំដាក់អត្ថបទពន្យល់ក្នុងផ្នែកទី១។ "
    "ផ្នែកទី២ ក្រោយសញ្ញាបែងចែក ត្រូវមានការពន្យល់ជាភាសាខ្មែរ ជាជំហានៗ ហើយត្រូវចាប់ផ្តើមដោយ "
    "🎯 ចម្លើយ៖ [ចម្លើយចុងក្រោយ] និងបញ្ចប់ដោយ ✅ ចម្លើយចុងក្រោយ៖ [ចម្លើយចុងក្រោយ]។ "
    "ការហាមប្រើ LaTeX និងការប្រើ Unicode math symbols អនុវត្តចំពោះផ្នែកទី២ប៉ុណ្ណោះ។"
)


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

    # ដាក់រូបភាពចូល contents ដោយផ្ទាល់តាមរយៈ PIL Image
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
                    return rendered_img, text_explanation

                return None, raw_answer

            except (ServerError, APIError, Exception) as e:
                # ស្រង់ status code ដោយសុវត្ថិភាព (e.code ឬ status_code)
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