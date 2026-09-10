from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING

from app.core.config import Settings
from app.services.retrieval import RetrievedChunk

if TYPE_CHECKING:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

LOGGER = logging.getLogger(__name__)
CITATION_PATTERN = re.compile(r"\[S(\d+)\]")


class GenerationService:
    context_top_k = 6
    context_max_words = 150

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = None
        self.types = None
        self.tokenizer: AutoTokenizer | None = None
        self.generator: AutoModelForSeq2SeqLM | None = None
        self.device = "cpu"
        mode = settings.generation_mode.lower().strip()

        if mode == "gemini":
            if not settings.gemini_api_key.strip():
                raise ValueError("GEMINI_API_KEY is required when GENERATION_MODE=gemini")
            from google import genai
            from google.genai import types
            self.client = genai.Client(api_key=settings.gemini_api_key)
            self.types = types
            LOGGER.info("Generator ready: %s via Gemini API", settings.gemini_model)
        elif mode == "local":
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(settings.local_generator_model)
            self.generator = AutoModelForSeq2SeqLM.from_pretrained(settings.local_generator_model).to(self.device)
            self.generator.eval()
            LOGGER.info("Generator ready: %s on %s", settings.local_generator_model, self.device)
        else:
            raise ValueError("GENERATION_MODE must be 'gemini' or 'local'")

    @staticmethod
    def _trim_to_words(text: str, max_words: int) -> str:
        words = text.split()
        return " ".join(words[:max_words]) + (" …" if len(words) > max_words else "")

    def make_context(self, ranked: list[RetrievedChunk]) -> str:
        blocks = []
        for number, result in enumerate(ranked[: self.context_top_k], start=1):
            chunk = result.chunk
            blocks.append(
                f"[S{number}] Title: {chunk.get('title', 'Untitled')} "
                f"({chunk.get('year') or 'n.d.'}, p.{chunk.get('page', '?')})\n"
                f"{self._trim_to_words(result.text, self.context_max_words)}"
            )
        return "\n\n".join(blocks)

    @staticmethod
    def build_prompt(question: str, context: str) -> str:
        return f"""You are a rigorous academic research assistant.

Answer the question using ONLY the numbered evidence snippets from the research corpus.

Rules:
1. Never use outside knowledge. Never invent facts, numbers, methods, or citations.
2. After every factual claim, add one or more source tags exactly like [S1] or [S2].
3. Synthesize across sources; do not dump the sources one by one.
4. If the evidence is insufficient, reply exactly: "The corpus does not provide enough evidence."

Question: {question}

Evidence:
{context}

Answer:
"""

    def validate_citations(self, answer: str, available_sources: int | None = None) -> tuple[bool, list[int], str]:
        limit = self.context_top_k if available_sources is None else min(available_sources, self.context_top_k)
        cited = sorted({int(value) for value in CITATION_PATTERN.findall(answer)})
        invalid = [value for value in cited if value < 1 or value > limit]
        if not cited:
            return False, [], "no valid [S#] citation tags found"
        if invalid:
            return False, cited, f"invalid citation tags: {invalid}"
        return True, cited, f"grounded - cited={cited}"

    def generate_answer(self, question: str, ranked: list[RetrievedChunk]) -> tuple[str, str]:
        if not ranked:
            return "The corpus does not provide enough evidence.", "no candidates"
        prompt = self.build_prompt(question, self.make_context(ranked))

        if self.client is not None:
            answer = self._gemini_answer(prompt)
        else:
            import torch
            if self.tokenizer is None or self.generator is None:
                raise RuntimeError("Local generator is not initialized")
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                output = self.generator.generate(
                    **inputs, max_new_tokens=180, do_sample=False, num_beams=4, repetition_penalty=1.08
                )
            answer = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()

        if not answer:
            raise RuntimeError("The generator returned an empty response")
        valid, cited, diagnostic = self.validate_citations(answer, len(ranked))
        if valid:
            top = max((r.rerank_score for r in ranked[: self.context_top_k]), default=0.0)
            if top < 0:
                diagnostic += f" - weak evidence signal (max rerank={top:.2f})"
        return answer, diagnostic

    def _gemini_answer(self, prompt: str, max_retries: int = 3) -> str:
        if self.client is None or self.types is None:
            raise RuntimeError("Gemini client is not initialized")
        backoff = 5
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.settings.gemini_model,
                    contents=prompt,
                    config=self.types.GenerateContentConfig(
                        system_instruction="You are a precise academic RAG assistant. Follow the [S#] citation format exactly.",
                        temperature=0.0,
                        max_output_tokens=600,
                    ),
                )
                answer = (getattr(response, "text", None) or "").strip()
                if answer:
                    return answer
                raise RuntimeError("Gemini returned an empty response")
            except Exception:
                if attempt == max_retries:
                    raise
                LOGGER.warning("Gemini generation failed on attempt %s/%s; retrying", attempt, max_retries)
                time.sleep(backoff)
                backoff *= 2
        raise RuntimeError("Gemini generation failed")

    def source_objects(self, ranked: list[RetrievedChunk]) -> list[dict]:
        sources = []
        for number, result in enumerate(ranked[: self.context_top_k], start=1):
            chunk = result.chunk
            url = chunk.get("source_url") or chunk.get("pdf_url") or "No URL available"
            sources.append({
                "source": f"S{number}",
                "paper_id": str(chunk.get("paper_id") or ""),
                "title": chunk.get("title", "Untitled"),
                "authors": chunk.get("authors") or [],
                "year": chunk.get("year"),
                "page": chunk.get("page"),
                "section": chunk.get("section"),
                "url": url,
                "pdf_url": chunk.get("pdf_url") or "",
            })
        return sources

    def source_strings(self, ranked: list[RetrievedChunk]) -> list[str]:
        return [
            f"{item['source']} — {item['title']} ({item.get('year') or 'n.d.'}, p. {item.get('page', '?')}) — {item.get('url') or 'No URL available'}"
            for item in self.source_objects(ranked)
        ]
