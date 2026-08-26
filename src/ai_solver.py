import logging
from typing import Optional

from google import genai
from google.genai import types

from src.config import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "អ្នកជាគ្រូបង្រៀនខ្មែរដែលមានបទពិសោធន៍ សម្រាប់សិស្សវិទ្យាល័យកម្ពុជា ថ្នាក់ទី ១០ ដល់ ១២។ "
    "ដោះស្រាយលំហាត់ដោយពន្យល់ជាជំហានៗ ឲ្យងាយយល់ និងត្រឹមត្រូវ។ "
    "ឆ្លើយជាភាសាខ្មែរ លើកលែងតែពាក្យបច្ចេកទេស ឬសញ្ញាគណិតវិទ្យាដែលគួររក្សាទុក។ "
    "បង្ហាញរូបមន្ត ការគណនា និងចម្លើយចុងក្រោយឲ្យច្បាស់។ "
    "ប្រើ Markdown សម្រាប់ចំណងជើង បញ្ជី និងសមីការ។ "
    "បើសំណួរមិនច្បាស់ សូមប្រាប់ចំណុចដែលត្រូវការបញ្ជាក់ ហើយកុំប្រឌិតទិន្នន័យ។"
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

    client = genai.Client(api_key=GEMINI_API_KEY)
    try:
        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
        )
        answer = (response.text or "").strip()
        if not answer:
            raise RuntimeError("Gemini returned an empty response")
        return answer
    finally:
        await client.aio.aclose()