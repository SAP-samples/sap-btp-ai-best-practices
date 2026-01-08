import asyncio
import base64
import logging
from typing import Any, AsyncIterator, Dict, List, Tuple, Optional

import fitz  # PyMuPDF
from gen_ai_hub.orchestration.models.config import OrchestrationConfig
from gen_ai_hub.orchestration.models.llm import LLM
from gen_ai_hub.orchestration.models.message import (
    SystemMessage,
    UserMessage,
    ImageItem,
)
from gen_ai_hub.orchestration.models.template import Template
from gen_ai_hub.orchestration.service import OrchestrationService

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-pro"

DEFAULT_TRANSCRIPTION_PROMPT = (
    "You are a transcription assistant.\n\n"
    "Your task:\n"
    "Extract all content from the provided scanned PDF and output it in Markdown, without summarizing or skipping anything that is legible.\n\n"
    "General rules:\n"
    "Do not summarize or omit content.\n"
    "Preserve the original order of content.\n"
    "Separate pages with a clear marker:\n"
    "--- PAGE X ---\n"
    "If something is unreadable, write [UNREADABLE] instead of guessing.\n\n"
    "Text:\n"
    "Transcribe all visible text exactly as it appears (including headers, footers, page numbers, footnotes, captions).\n"
    "Keep paragraph breaks where they clearly exist.\n"
    "If the document has obvious headings, convert them to Markdown headings (#, ##, etc.) while keeping the original wording.\n\n"
    "Lists:\n"
    "Convert bullet lists to Markdown lists using - or *.\n"
    "Convert numbered lists to 1., 2., 3. etc.\n"
    "Preserve nesting/indentation as best as possible.\n\n"
    "Tables:\n"
    "Convert every table into a proper Markdown table.\n"
    "Include all rows and columns, even if they look repetitive.\n"
    "If a cell is empty, leave it blank.\n\n"
    "Images, diagrams, stamps, signatures:\n"
    "If an image contains legible text (e.g., a scanned form field, label, or figure caption), transcribe that text.\n"
    "If the image does not contain text or the text is unreadable, add a short placeholder like:\n"
    '![IMAGE: description if obvious, otherwise "no readable text"]\n'
    "For signatures, write [SIGNATURE] and any nearby printed name if visible.\n"
    "For stamps/seals with text, transcribe the text as best as possible; if partially unreadable, use [UNREADABLE] for missing parts.\n\n"
    "Numbers, codes, IDs:\n"
    "Pay extra attention to numbers, dates, IDs, codes, and references.\n"
    "Do not “clean” or reformat them; copy what you see.\n\n"
    "Quality / uncertainties:\n"
    "If you are unsure about a word due to scan quality, mark it like this: appli[cant?].\n"
    "Do not invent content that is not visible in the scan.\n\n"
    "Output:\n"
    "Return only the Markdown transcription of the document, from the first page to the last, following the rules above.\n"
    "Do not add explanations, comments, or instructions outside the document content itself.\n"
)


def _build_batch_user_parts(pages: List[Tuple[int, str]]) -> List[Any]:
    """Create a list of multimodal parts accepted by UserMessage: str or ImageItem."""
    if not pages:
        return []
    start = pages[0][0]
    end = pages[-1][0]
    parts: List[Any] = []
    parts.append(f"(This batch contains pages {start} to {end} in order.)")
    for page_num, data_url in pages:
        parts.append(f"--- PAGE {page_num} ---")
        parts.append(ImageItem(url=data_url))
    return parts


class DocumentTranscriber:
    def __init__(
        self,
        orchestration_service: Optional[OrchestrationService] = None,
        model: str = DEFAULT_MODEL,
        prompt: str = DEFAULT_TRANSCRIPTION_PROMPT,
        dpi: int = 300,
        image_format: str = "png",  # "png" or "jpg"
        jpeg_quality: int = 75,
        grayscale: bool = False,
        auto_grayscale_megapixels: Optional[float] = None,
        max_concurrency: int = 200,
        max_retries: int = 5,
        base_retry_delay: float = 0.6,
    ) -> None:
        self._orchestration = orchestration_service or OrchestrationService()
        self._model = model
        self._prompt = prompt
        self._dpi = dpi
        self._image_format = image_format.lower()
        self._jpeg_quality = int(jpeg_quality)
        self._grayscale = bool(grayscale)
        self._auto_gray_mp = auto_grayscale_megapixels
        self._max_concurrency = max_concurrency
        self._max_retries = max_retries
        self._base_retry_delay = base_retry_delay

    def _build_config(self) -> OrchestrationConfig:
        return OrchestrationConfig(
            template=Template(
                messages=[
                    SystemMessage(self._prompt),
                ]
            ),
            llm=LLM(name=self._model),
        )

    async def _acall_orchestration_markdown(self, parts: List[Any]) -> Tuple[str, Any]:
        config = self._build_config()
        result = await self._orchestration.arun(
            config=config,
            history=[UserMessage(parts)],
        )
        content = result.orchestration_result.choices[0].message.content
        usage = result.orchestration_result.usage
        return content, usage

    def render_pdf(self, pdf_bytes: bytes) -> List[Tuple[int, str]]:
        """Render the PDF pages to data URLs using configured format and options."""
        try:
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                results: List[Tuple[int, str]] = []
                for i in range(doc.page_count):
                    page = doc.load_page(i)
                    zoom = self._dpi / 72.0
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    try:
                        data_bytes, mime = self._encode_pixmap(pix)
                    finally:
                        del pix
                        del page
                    b64 = base64.b64encode(data_bytes).decode("ascii")
                    data_url = f"data:{mime};base64,{b64}"
                    results.append((i + 1, data_url))
                return results
        except Exception:
            raise ValueError("Invalid PDF file.")

    def _encode_pixmap(self, pix: fitz.Pixmap) -> Tuple[bytes, str]:
        """Convert pixmap to bytes with optional grayscale and desired format."""
        # Decide grayscale
        do_gray = self._grayscale
        if self._auto_gray_mp is not None:
            pixels = pix.width * pix.height
            if pixels >= int(self._auto_gray_mp * 1_000_000):
                do_gray = True

        if do_gray and pix.n != 1:  # not already grayscale
            try:
                gray = fitz.Pixmap(fitz.csGRAY, pix)
                try:
                    pix = gray
                finally:
                    # gray now referenced by pix; original 'pix' will be GC'ed
                    pass
            except Exception:
                # Fallback: keep original pix if grayscale conversion fails
                pass

        if self._image_format in ("jpg", "jpeg"):
            data = pix.tobytes(
                output="jpeg", jpg_quality=max(1, min(self._jpeg_quality, 95))
            )
            return data, "image/jpeg"
        else:
            data = pix.tobytes("png")
            return data, "image/png"

    async def transcribe_pages(self, pages: List[Tuple[int, str]]) -> Tuple[List[str], Dict[str, int]]:
        if not pages:
            return [], {}

        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def process_one(idx: int, page: Tuple[int, str]) -> Tuple[int, str, Any]:
            async with semaphore:
                delay = self._base_retry_delay
                page_no = page[0]
                for attempt in range(self._max_retries + 1):
                    try:
                        user_parts = _build_batch_user_parts([page])
                        logger.info(
                            "Processing page %s/%s (attempt %s)",
                            idx + 1,
                            len(pages),
                            attempt + 1,
                        )
                        text, usage = await self._acall_orchestration_markdown(user_parts)
                        return idx, text.strip(), usage
                    except Exception as e:
                        if attempt < self._max_retries:
                            await asyncio.sleep(delay)
                            delay *= 2
                        else:
                            logger.error("Page %s failed after retries: %s", page_no, e)
                            placeholder = f"--- PAGE {page_no} ---\n[UNREADABLE]"
                            return idx, placeholder, None

        tasks = [process_one(i, p) for i, p in enumerate(pages)]
        results = await asyncio.gather(*tasks)

        outputs: List[str] = []
        total_usage = {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}
        
        for _, text, usage in results:
            outputs.append(text)
            if usage:
                # Attempt to sum up usage if it has standard attributes
                try:
                    if hasattr(usage, "completion_tokens"):
                         total_usage["completion_tokens"] += usage.completion_tokens or 0
                    if hasattr(usage, "prompt_tokens"):
                         total_usage["prompt_tokens"] += usage.prompt_tokens or 0
                    if hasattr(usage, "total_tokens"):
                         total_usage["total_tokens"] += usage.total_tokens or 0
                except Exception:
                    pass

        return outputs, total_usage

    async def transcribe_pdf_to_markdown(self, pdf_bytes: bytes) -> Dict[str, Any]:
        pages = self.render_pdf(pdf_bytes)
        if not pages:
            raise ValueError("No pages found in PDF.")
        outputs, token_usage = await self.transcribe_pages(pages)
        combined = "\n\n".join(outputs).strip()
        return {
            "markdown": combined,
            "page_count": len(pages),
            "model": self._model,
            "batches": len(pages),
            "token_usage": token_usage,
        }

    async def stream_transcription_events(
        self, pdf_bytes: bytes
    ) -> AsyncIterator[Dict[str, Any]]:
        try:
            yield {"type": "status", "message": "Preparing document..."}
            try:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            except Exception:
                yield {"type": "error", "message": "Invalid PDF file."}
                return

            try:
                yield {
                    "type": "status",
                    "message": "Document opened. Rendering pages...",
                }
                total_pages = doc.page_count
                yield {"type": "start", "page_count": total_pages}

                pages: List[Tuple[int, str]] = []
                dpi = self._dpi
                for i in range(total_pages):
                    page = doc.load_page(i)
                    zoom = dpi / 72.0
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    try:
                        data_bytes, mime = self._encode_pixmap(pix)
                    finally:
                        del pix
                        del page
                    b64 = base64.b64encode(data_bytes).decode("ascii")
                    data_url = f"data:{mime};base64,{b64}"
                    pages.append((i + 1, data_url))
                    yield {
                        "type": "render_progress",
                        "current": i + 1,
                        "total": total_pages,
                    }
            finally:
                try:
                    doc.close()
                except Exception:
                    pass

            if not pages:
                yield {"type": "error", "message": "No pages found in PDF."}
                return

            semaphore = asyncio.Semaphore(self._max_concurrency)

            async def process_one(
                idx: int, page: Tuple[int, str]
            ) -> Tuple[int, str, int, bool]:
                async with semaphore:
                    delay = self._base_retry_delay
                    page_no = page[0]
                    for attempt in range(self._max_retries + 1):
                        try:
                            user_parts = _build_batch_user_parts([page])
                            text, _ = await self._acall_orchestration_markdown(user_parts)
                            return idx, text.strip(), attempt, True
                        except Exception:
                            if attempt < self._max_retries:
                                await asyncio.sleep(delay)
                                delay *= 2
                            else:
                                placeholder = f"--- PAGE {page_no} ---\n[UNREADABLE]"
                                return idx, placeholder, attempt, False

            tasks: List[asyncio.Task] = []
            start_times: Dict[int, float] = {}
            loop = asyncio.get_event_loop()
            for i, p in enumerate(pages):
                start_times[i] = loop.time()
                yield {"type": "process_start", "index": i, "page": p[0]}
                tasks.append(asyncio.create_task(process_one(i, p)))

            results: Dict[int, str] = {}
            completed = 0
            for fut in asyncio.as_completed(tasks):
                try:
                    idx, text, attempts, ok = await fut
                    results[idx] = text
                    completed += 1
                    duration_ms = int(
                        (loop.time() - start_times.get(idx, loop.time())) * 1000
                    )
                    yield {
                        "type": "process_done",
                        "index": idx,
                        "page": pages[idx][0],
                        "duration_ms": duration_ms,
                        "attempts": attempts + 1,
                        "ok": ok,
                    }
                    yield {
                        "type": "process_progress",
                        "current": completed,
                        "total": len(pages),
                    }
                except Exception as e:
                    yield {"type": "error", "message": str(e)}

            combined = "\n\n".join(results[i] for i in range(len(pages))).strip()
            yield {
                "type": "result",
                "markdown": combined,
                "page_count": len(pages),
                "batches": len(pages),
            }
        except Exception as e:
            yield {"type": "error", "message": str(e)}
