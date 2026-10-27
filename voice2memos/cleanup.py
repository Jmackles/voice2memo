"""Cleanup: provider base class, Ollama/OpenAI/Anthropic/Fake, rewrite + entity + tag calls.

Each provider exposes one ``complete`` method. The Fake provider answers from a
fixture by matching on which prompt it was handed, and has modes that raise or
sleep past the timeout for error-state tests. ``provider = fake`` in the .cfg
runs the whole app end to end with no models loaded.

Ollama ``format: json`` guarantees valid JSON, not an array: prompts ask for an
object like ``{"items": [...]}`` and the caller reads ``items``.
"""
