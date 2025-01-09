import json
from pathlib import Path
from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def extract_quotes(transcriptions_dir: Path) -> list:
    """Extract motivational quotes from transcriptions using LlamaIndex."""
    # Configure LlamaIndex settings
    Settings.llm = Ollama(model="gemma2")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # Load and index documents
    documents = SimpleDirectoryReader(input_dir=transcriptions_dir).load_data()
    index = VectorStoreIndex.from_documents(documents)

    # Query for motivational quotes
    detailed_prompt = """
    Your task is to identify motivational or inspirational excerpts from the provided transcript. These excerpts must:

    1. Be **at least 2-5 sentences long** to provide enough context for a meaningful video clip.
    2. **Preserve the exact wording** from the transcript without any modifications or omissions.
    3. Focus on **overcoming challenges, achieving goals, or self-improvement**.
    4. Evoke **positive emotions** such as hope, determination, or resilience.
    5. Provide **unique insights or actionable advice** that can benefit the audience.

    When analyzing the text:
    - Avoid generic statements, conversational filler, or overly brief responses.
    - Select excerpts that stand out as powerful and memorable.

    Output the excerpts in **verbatim** as a list of strings, with no additional commentary or explanation. Each string must correspond exactly to a section of the transcript.
    """

    query_engine = index.as_query_engine()
    response = query_engine.query(detailed_prompt)
    quotes = json.loads(response.response.strip().replace("```", "").strip())

    # Save extracted quotes
    with open("motivational_lines.json", "w") as output_file:
        json.dump(quotes, output_file)

    return quotes
