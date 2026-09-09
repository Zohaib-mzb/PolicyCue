from google import genai

from backend.app.core.config import get_settings


settings = get_settings()
client = genai.Client(api_key=settings.gemini_api_key)


def generate_answer(
    question: str,
    context: list[dict],
) -> str:
    if not context:
        return "I could not find relevant information in the provided documents."

    context_text = "\n\n".join(
        f"[Source {i + 1}]\n{item['text']}"
        for i, item in enumerate(context)
    )

    prompt = f"""
You are PolicyLens, a policy analysis assistant.

Answer the user's question using ONLY the provided policy content.

Rules:
- Do not use outside knowledge.
- Do not invent facts.
- Separate what the policy explicitly says from your interpretation.
- If the content is insufficient, clearly say so.
- If the policy is ambiguous, explain the ambiguity.
- Give a practical answer.
- Keep the answer concise.

User question:
{question}

Policy content:
{context_text}
"""

    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=prompt,
    )

    return response.text