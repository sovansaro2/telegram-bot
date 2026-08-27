import logging
from typing import Optional

from google import genai
from google.genai.errors import APIError, ServerError
from google.genai import types

from src.config import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)


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
    "បើសំណួរមិនច្បាស់ សូមបង្ហាញចំណុចដែលមិនច្បាស់នៅក្នុងជំហានដោះស្រាយ ហើយកុំប្រឌិតទិន្នន័យ។"
)


async def solve_homework(
    question: str = "",
    image_bytes: Optional[bytes] = None,
    mime_type: str = "image/jpeg",
) -> str:
    """Ask Gemini to solve text or image-based high-school homework."""
    if not GEMINI_API_KEY:
        raise HomeworkSolverError(
            "GEMINI_API_KEY is not configured",
            "❌ មុខងារ AI មិនទាន់បានកំណត់រចនាសម្ព័ន្ធទេ។ សូមជូនដំណឹងទៅ Admin។",
        )

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
    except ServerError as e:
        status_code = getattr(e, "status_code", None)
        logger.warning("Gemini server error (%s): %s", status_code or 503, e)
        raise HomeworkSolverError(
            "Gemini server is temporarily overloaded",
            "⚠️ ម៉ាស៊ីនមេ AI កំពុងមានអ្នកប្រើប្រាស់ច្រើន (High Demand)។ "
            "សូមរង់ចាំប្រហែល ១ នាទី រួចសាកល្បងម្ដងទៀត!",
        ) from e
    except APIError as e:
        status_code = getattr(e, "status_code", None)
        logger.warning("Gemini API request failed (%s): %s", status_code, e)
        if status_code in (429, 503):
            raise HomeworkSolverError(
                "Gemini request was rate limited",
                "⚠️ សេវា AI កំពុងមានការស្នើសុំច្រើន។ សូមរង់ចាំបន្តិច រួចសាកល្បងម្ដងទៀត។",
            ) from e
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