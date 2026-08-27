import logging
from typing import Optional

from google import genai
from google.genai.errors import APIError
from google.genai import types

from src.config import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)


class HomeworkSolverError(RuntimeError):
    """Safe, user-facing failure raised when Gemini cannot solve homework."""

SYSTEM_PROMPT = (
    "អ្នកជាគ្រូបង្រៀនខ្មែរដែលមានបទពិសោធន៍ សម្រាប់សិស្សវិទ្យាល័យកម្ពុជា ថ្នាក់ទី ១០ ដល់ ១២។ "
    "ឆ្លើយជាភាសាខ្មែរ ឲ្យច្បាស់លាស់ ត្រឹមត្រូវ និងសមស្របសម្រាប់សិស្សថ្នាក់ទី ១០ ដល់ ១២។ "
    "កុំសរសេរសួរសុខទុក្ខ ឬសេចក្តីផ្តើមណាមួយឡើយ។ ហាមប្រើពាក្យ សួស្ដី ឬ ជំរាបសួរ។ "
    "ចាប់ផ្តើមចម្លើយភ្លាមៗ ដោយបន្ទាត់ទីមួយត្រូវតែជា៖ 🎯 ចម្លើយ៖ [ចម្លើយចុងក្រោយ]។ "
    "បន្ទាប់មកដាក់ចំណងជើង៖ 📌 នេះជាប្រមាណវិធី៖ ហើយបង្ហាញការគណនា និងការពន្យល់ជាជំហានៗ។ "
    "នៅបន្ទាត់ចុងក្រោយបំផុត ត្រូវធ្វើម្តងទៀតជា៖ ✅ ចម្លើយចុងក្រោយ៖ [ចម្លើយចុងក្រោយ]។ "
    "កុំដាក់អត្ថបទណាមួយមុនបន្ទាត់ 🎯 ឬក្រោយបន្ទាត់ ✅។ "
    "ហាមប្រើ LaTeX ជាដាច់ខាត រួមទាំង $$, $ និងពាក្យបញ្ជា \\lim, \\frac, \\sqrt, \\to, \\infty ឬទម្រង់ LaTeX ផ្សេងទៀត។ "
    "ប្រើតែអក្សរធម្មតាដែលអានងាយ និងតួអក្សរគណិតវិទ្យា Unicode ប៉ុណ្ណោះ។ "
    "សម្រាប់ឫសការេ ប្រើ √ ដូចជា √(4x² + x + 2)។ សម្រាប់ប្រភាគ ប្រើ / ដូចជា 1/x។ "
    "សម្រាប់លីមីត ប្រើ → ដូចជា lim (x → +∞) ហើយប្រើ ∞ សម្រាប់អនន្ត។ "
    "សម្រាប់ស្វ័យគុណ ប្រើលេខលើ Unicode ដូចជា x² និង x³។ "
    "បើសំណួរមិនច្បាស់ សូមបង្ហាញចំណុចដែលមិនច្បាស់នៅក្នុងជំហានដោះស្រាយ ហើយកុំប្រឌិតទិន្នន័យ។"
)


async def solve_homework(
    question: str = "",
    image_bytes: Optional[bytes] = None,
    mime_type: str = "image/jpeg",
) -> str:
    """Ask Gemini to solve text or image-based high-school homework."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    contents: list[object] = []
    if question.strip():
        contents.append(question.strip())
    if image_bytes:
        contents.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
    if not contents:
        raise ValueError("Homework input is empty")

    client = None
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
        )
        answer = (response.text or "").strip()
        if not answer:
            raise HomeworkSolverError("Gemini returned an empty response")
        return answer
    except APIError as e:
        logger.warning("Gemini API request failed: %s", e)
        raise HomeworkSolverError(
            "The homework service is temporarily unavailable"
        ) from e
    except HomeworkSolverError:
        raise
    except Exception as e:
        logger.error("Unexpected Gemini solver error: %s", e, exc_info=True)
        raise HomeworkSolverError("The homework service could not process the request") from e
    finally:
        if client is not None:
            try:
                await client.aio.aclose()
            except Exception as e:
                logger.warning("Could not close Gemini client cleanly: %s", e)