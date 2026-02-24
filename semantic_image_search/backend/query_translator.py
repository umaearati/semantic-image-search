from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from semantic_image_search.backend.config import Config
from semantic_image_search.backend.logger import GLOBAL_LOGGER as log
from semantic_image_search.backend.exception.custom_exception import SemanticImageSearchException


class QueryTranslator:
    """
    LLM-based Query Rewriter for CLIP-style image caption search.
    Optimised with:
    - Input validation
    - Length control
    - Early cache check
    - Conversational detection
    - Conditional LLM invocation
    """

    def __init__(self):
        try:
            log.info("Initializing QueryTranslator...", model=Config.OPENAI_MODEL)

            self.llm = ChatOpenAI(
                model=Config.OPENAI_MODEL,
                temperature=0,
                timeout=20,
            )

            # 🔹 In-memory cache
            self._query_cache = {}

            # 🔹 Maximum character limit for cost control
            self.MAX_QUERY_LENGTH = 200

            self.prompt_template = PromptTemplate(
                input_variables=["input_query"],
                template="""
You are an expert at rewriting queries for the CLIP image–text model.

Goal:
Rewrite the user query into a short, concrete, descriptive image caption.
The rewritten query must maximize CLIP retrieval accuracy.

Guidelines:
- Keep the original meaning.
- Use 3–12 word caption style.
- Remove chat words (show me, give me, please, etc.)
- Keep colors, objects, actions.
- Translate to English if needed.
- Do NOT add new details.

User Query: {input_query}

Respond with only the rewritten caption.
                """.strip(),
            )

            log.info("QueryTranslator initialized successfully")

        except Exception as e:
            log.error("Failed to initialize QueryTranslator", error=str(e))
            raise SemanticImageSearchException(
                "Failed to initialize QueryTranslator", e
            )

    def translate(self, user_query: str) -> str:
        """Translate chat-style input into CLIP-style caption."""

        # 1️⃣ Input validation
        if not isinstance(user_query, str) or not user_query.strip():
            log.error("Invalid input query for translation", query=user_query)
            raise ValueError("Query must be a non-empty string")

        log.info("Translating query", input_query=user_query)

        # 2️⃣ Normalisation
        normalized_query = user_query.strip().lower()

        # 3️⃣ Length control (cost protection)
        if len(normalized_query) > self.MAX_QUERY_LENGTH:
            log.info(
                "Query truncated due to length limit",
                original_length=len(normalized_query),
                max_length=self.MAX_QUERY_LENGTH,
            )
            normalized_query = normalized_query[:self.MAX_QUERY_LENGTH]

        # 4️⃣ 🔥 Early cache check (fastest path)
        if normalized_query in self._query_cache:
            log.info(
                "Cache hit - returning cached translation",
                query=normalized_query,
            )
            return self._query_cache[normalized_query]

        # 5️⃣ Conversational detection
        CHAT_PATTERNS = [
            "show me",
            "please",
            "give me",
            "can you",
            "i want",
            "find me",
            "could you",
        ]

        is_conversational = any(
            pattern in normalized_query for pattern in CHAT_PATTERNS
        )

        if not is_conversational:
            log.info("Caption-style query detected - skipping rewrite")
            return normalized_query

        log.info("Conversational query detected - rewriting required")

        # 6️⃣ LLM invocation
        try:
            prompt = self.prompt_template.format(
                input_query=normalized_query
            )

            log.info("Sending translation prompt to LLM")

            final_caption = self.llm.invoke(prompt).content.strip()

            # 7️⃣ Store result in cache
            self._query_cache[normalized_query] = final_caption

            log.info(
                "Translation completed",
                original=user_query,
                translated=final_caption,
            )

            return final_caption

        except Exception as e:
            log.error("LLM translation failed", query=user_query, error=str(e))
            raise SemanticImageSearchException(
                "LLM translation failed", e
            )


# ---- Lazy Singleton ----
_translator_instance = None


def translate_query(user_query: str) -> str:
    global _translator_instance
    if _translator_instance is None:
        _translator_instance = QueryTranslator()
    return _translator_instance.translate(user_query)












































































# It indicates the attribute is intended for internal use within the class and not part of the public interface. It improves encapsulation and code clarity.