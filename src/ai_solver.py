import asyncio
from html import escape
import io
from io import BytesIO
import logging
import re
from typing import Optional
import urllib.parse

import aiohttp
from google import genai
from google.genai import types
from google.genai.errors import APIError, ServerError
from PIL import Image

from src.config import GEMINI_API_KEY

logger = logging.getLogger(__name__)

CODECOGS_LATEX_URL = "https://latex.codecogs.com/png.image"

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


SYSTEM_PROMPT_DETAILED = (
    "អ្នកជាគ្រូបង្រៀនគណិតវិទ្យា និងរូបវិទ្យាដ៏ឆ្នើមសម្រាប់សិស្សវិទ្យាល័យកម្ពុជា។\n"
    "អ្នកត្រូវតែផ្តល់ចម្លើយជា ២ ផ្នែកដាច់ខាត ដោយបែងចែកដោយសញ្ញា ===LATEX_BLOCK=== តែមួយគត់៖\n\n"
    "--- [ផ្នែកទី ១: រូបមន្ត LaTeX សម្រាប់ Render ជារូបភាព] ---\n"
    "សរសេរតែរូបមន្ត និងដំណោះស្រាយជាកូដ LaTeX សុទ្ធសាធ ដោយប្រើ \\begin{aligned} ... \\end{aligned} "
    "បង្ហាញជំហានគណនាទាំងអស់។ កុំប្រើ markdown backticks នៅផ្នែកទី១នេះ។\n\n"
    "===LATEX_BLOCK===\n\n"
    "--- [ផ្នែកទី ២: ការពន្យល់ជាភាសាខ្មែរ] ---\n"
    "ចាប់ផ្តើមភ្លាមៗដោយ៖\n"
    "🎯 ចម្លើយ៖ [ចម្លើយសង្ខេប]\n\n"
    "📌 នេះជាប្រមាណវិធី និងការពន្យល់៖\n"
    "ពន្យល់ជាភាសាខ្មែរជាជំហានៗ។ រាល់ពេលមានរូបមន្ត ឬការគណនា ត្រូវដាក់ក្នុង Markdown code block ```text ... ```\n"
    "ច្បាប់តឹងរ៉ឹងសម្រាប់ code block៖\n"
    "- ត្រូវចុះបន្ទាត់ថ្មីសម្រាប់រាល់ជំហាន (១ បន្ទាត់ = ១ ជំហាន ដោយផ្ដើមដោយ = នៅបន្ទាត់ថ្មី)។ ហាមសរសេរតគ្នាក្នុងបន្ទាត់តែមួយ!\n"
    "- ហាមប្រើ sqrt, ^2, *។ ត្រូវប្រើនិមិត្តសញ្ញា Unicode គណិតវិទ្យាស្អាតជានិច្ច (√, ², ³, →, +∞, ×, ÷, /)។\n\n"
    "✅ ចម្លើយចុងក្រោយ៖\n"
    "```text\n"
    "[ចម្លើយចុងក្រោយ]\n"
    "```\n\n"
    "ចំណាំ៖ ហាមសរសេរសួរសុខទុក្ខ។ ត្រូវតែមាន ===LATEX_BLOCK=== ជានិច្ច!"
)

SYSTEM_PROMPT_QUICK = (
    "អ្នកជាម៉ាស៊ីនគណិតវិទ្យា និងរូបវិទ្យារហ័ស សម្រាប់សិស្សវិទ្យាល័យកម្ពុជា។\n"
    "អ្នកត្រូវតែផ្តល់ចម្លើយជា ២ ផ្នែកដាច់ខាត ដោយបែងចែកដោយសញ្ញា ===LATEX_BLOCK=== តែមួយគត់៖\n\n"
    "--- [ផ្នែកទី ១: រូបមន្ត LaTeX សម្រាប់ Render ជារូបភាព] ---\n"
    "សរសេរតែរូបមន្ត និងដំណោះស្រាយជាកូដ LaTeX សុទ្ធសាធ ដោយប្រើ \\begin{aligned} ... \\end{aligned} "
    "បង្ហាញជំហានគណនាទាំងអស់។ កុំប្រើ markdown backticks នៅផ្នែកទី១នេះ។\n\n"
    "===LATEX_BLOCK===\n\n"
    "--- [ផ្នែកទី ២: ប្រមាណវិធី និងចម្លើយសុទ្ធ] ---\n"
    "ចាប់ផ្តើមភ្លាមៗដោយ៖\n"
    "🎯 ចម្លើយ៖ [ចម្លើយសង្ខេប]\n\n"
    "📌 នេះជាប្រមាណវិធី៖\n"
    "```text\n"
    "lim (x → +∞) [√(x² + x + 2) - √(x² - x + 3)]\n"
    "= lim (x → +∞) [(x² + x + 2) - (x² - x + 3)] / [√(x² + x + 2) + √(x² - x + 3)]\n"
    "= lim (x → +∞) (2x - 1) / [√(x²(1 + 1/x + 2/x²)) + √(x²(1 - 1/x + 3/x²))]\n"
    "= lim (x → +∞) (2 - 1/x) / [√(1 + 1/x + 2/x²) + √(1 - 1/x + 3/x²)]\n"
    "= (2 - 0) / [√(1 + 0 + 0) + √(1 - 0 + 0)]\n"
    "= 2 / 2\n"
    "= 1\n"
    "```\n\n"
    "✅ ចម្លើយចុងក្រោយ៖\n"
    "```text\n"
    "[ចម្លើយចុងក្រោយ]\n"
    "```\n\n"
    "ច្បាប់តឹងរ៉ឹង៖\n"
    "- រាល់ជំហានគណនាត្រូវតែចុះបន្ទាត់ថ្មីដាច់ខាត (១ បន្ទាត់ = ១ ជំហាន ដោយផ្ដើមដោយសញ្ញា = នៅបន្ទាត់ថ្មី)។ ហាមសរសេរតគ្នាក្នុងបន្ទាត់តែមួយ!\n"
    "- ហាមប្រើ sqrt, ^2, ^3, _, *, +inf, -inf។ ប្រើ: √, ², ³, →, +∞, ×, ÷, /។\n"
    "- ត្រូវតែមាន ===LATEX_BLOCK=== ជានិច្ច!"
)


def clean_math_unicode(text: str) -> str:
    """បម្លែងអក្សរកូដ Programming ឆៅ ឱ្យទៅជានិមិត្តសញ្ញាគណិតវិទ្យាស្អាត"""
    replacements = [
        (r"\\lim_\{([^}]+)\}", r"lim (\1)"),
        (r"lim_\{([^}]+)\}", r"lim (\1)"),
        (r"lim_\(([^)]+)\)", r"lim (\1)"),
        (r"sqrt\((.*?)\)", r"√(\1)"),
        (r"\bsqrt\b", "√"),
        (r"\+inf\b", "+∞"),
        (r"-inf\b", "-∞"),
        (r"\binf\b", "∞"),
        (r"->", " → "),
        (r"\^2", "²"),
        (r"\^3", "³"),
        (r"\^4", "⁴"),
        (r"\^n", "ⁿ"),
        (r"\*", " × "),
        (r"\s*×\s*", " × "),
        (r"\s*→\s*", " → "),
        (r"\s*=\s*", " = "),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)
    return text


def format_code_block_steps(code_text: str) -> str:
    """បំបែកជំហានគណនាដែលតគ្នាដោយសញ្ញា = ឱ្យចុះបន្ទាត់ថ្មីជាជួរឈរស្អាត"""
    lines = code_text.splitlines()
    formatted_lines = []

    for line in lines:
        line = clean_math_unicode(line.strip())
        if not line:
            continue

        # ប្រសិនបើបន្ទាត់នោះមានសញ្ញា = លើសពី ១ បំបែកវាជាបន្ទាត់ៗ
        if line.count(" = ") >= 1 and not line.startswith("="):
            parts = line.split(" = ")
            formatted_lines.append(parts[0].strip())
            for part in parts[1:]:
                formatted_lines.append(f"= {part.strip()}")
        elif line.count(" = ") >= 2 and line.startswith("="):
            parts = line.split(" = ")
            for part in parts:
                p = part.strip()
                if p:
                    formatted_lines.append(f"= {p}" if not p.startswith("=") else p)
        else:
            formatted_lines.append(line)

    return "\n".join(formatted_lines)


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
            non_code = clean_math_unicode(non_code)
            escaped_non_code = escape(non_code)
            escaped_non_code = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", escaped_non_code)
            escaped_non_code = re.sub(
                r"(🎯\s*ចម្លើយ៖?[^\n]*)", r"<b>\1</b>", escaped_non_code
            )
            escaped_non_code = re.sub(
                r"(📌\s*នេះជាប្រមាណវិធី(?: និងការពន្យល់)?៖?)",
                r"<b>\1</b>",
                escaped_non_code,
            )
            escaped_non_code = re.sub(
                r"(✅\s*ចម្លើយចុងក្រោយ៖?)", r"<b>\1</b>", escaped_non_code
            )
            parts.append(escaped_non_code)

        # រៀបចំ និងចុះបន្ទាត់ជំហានគណនាក្នុង Code Block
        code_content = match.group(1).strip()
        formatted_code = format_code_block_steps(code_content)
        escaped_code = escape(formatted_code)
        parts.append(f"<pre><code>{escaped_code}</code></pre>")
        last_idx = end

    remaining = raw_text[last_idx:]
    if remaining:
        remaining = clean_math_unicode(remaining)
        escaped_rem = escape(remaining)
        escaped_rem = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", escaped_rem)
        escaped_rem = re.sub(r"(🎯\s*ចម្លើយ៖?[^\n]*)", r"<b>\1</b>", escaped_rem)
        escaped_rem = re.sub(
            r"(📌\s*នេះជាប្រមាណវិធី(?: និងការពន្យល់)?៖?)",
            r"<b>\1</b>",
            escaped_rem,
        )
        escaped_rem = re.sub(r"(✅\s*ចម្លើយចុងក្រោយ៖?)", r"<b>\1</b>", escaped_rem)
        parts.append(escaped_rem)

    return "".join(parts)


async def render_latex_to_image(latex_str: str) -> bytes | None:
    """Render a LaTeX formula block with CodeCogs into a Telegram-ready PNG."""
    try:
        latex_code = latex_str.strip()
        if not latex_code:
            return None

        if "\n" in latex_code and not latex_code.startswith("\\begin"):
            latex_code = f"\\begin{{aligned}}\n{latex_code}\n\\end{{aligned}}"

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
    mode: str = "detailed",
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
        else "សូមជួយដោះស្រាយលំហាត់ក្នុងរូបភាពនេះជាភាសាខ្មែរ។"
    )
    contents.append(prompt_text)

    system_prompt = (
        SYSTEM_PROMPT_QUICK if mode == "quick" else SYSTEM_PROMPT_DETAILED
    )

    client = None
    last_status = None

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)

        for model_name in FALLBACK_MODELS:
            try:
                logger.info("Trying Gemini model (%s): %s", mode, model_name)
                response = await client.aio.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt
                    ),
                )
                raw_answer = (response.text or "").strip()
                if not raw_answer:
                    continue

                latex_code = None
                text_explanation = raw_answer

                if "===LATEX_BLOCK===" in raw_answer:
                    parts = raw_answer.split("===LATEX_BLOCK===", 1)
                    latex_code = parts[0].strip()
                    text_explanation = parts[1].strip()
                elif "```latex" in raw_answer:
                    match = re.search(r"```latex\s*(.*?)\s*```", raw_answer, re.DOTALL)
                    if match:
                        latex_code = match.group(1).strip()
                        text_explanation = raw_answer.replace(match.group(0), "").strip()

                rendered_img = None
                if latex_code:
                    if latex_code.startswith("```"):
                        latex_code = latex_code.split("\n", 1)[-1]
                        latex_code = latex_code.rsplit("```", 1)[0].strip()
                    rendered_img = await render_latex_to_image(latex_code)

                formatted_html = format_explanation_to_html(text_explanation)
                return rendered_img, formatted_html

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