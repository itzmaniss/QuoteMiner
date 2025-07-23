import anthropic
import dotenv
import os
import json
from typing import List, Optional
from models import Quote
from utils import Logger

dotenv.load_dotenv()

logger = Logger(name="QuoteExtractor", filename="QuoteMiner.log")


class QuoteExtractorConfig:
    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        max_duration: int = 90,
        min_duration: int = 40,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_duration = max_duration
        self.min_duration = min_duration


class QuoteExtractor:
    def __init__(self, config: Optional[QuoteExtractorConfig] = None):
        self.config = config or QuoteExtractorConfig()
        logger.info(f"Initializing QuoteExtractor with model: {self.config.model}")
        self.client = self._initialize_client()
        self.system_prompt = self._build_system_prompt()
        logger.debug("QuoteExtractor initialization complete")

    def _initialize_client(self) -> anthropic.Anthropic:
        logger.debug("Initializing Anthropic client")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.error("ANTHROPIC_API_KEY environment variable not set")
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        logger.info("Anthropic client initialized successfully")
        return anthropic.Anthropic(api_key=api_key)

    def _build_system_prompt(self) -> str:
        return f"""
You are a motivational clip extractor.

The user provides a list of transcribed text segments, each with a start and end timestamp.
Your task is to select ten or more(preferrably) high-impact, motivational moments that could be used as TikTok-style short clips.

✦ Each clip must:
- Be at least {self.config.min_duration} seconds long.
- Be less than {self.config.max_duration} seconds long.
- Be contiguous — use consecutive lines from the input with no skips.
- Be non-overlapping by default, but overlap is allowed only if the clip expresses a clearly distinct emotional or rhetorical idea.
- Be motivational or emotionally engaging — thought-provoking, passionate, or punchy.
- Be suitable for standalone short-form content (e.g., TikTok, Reels).

✦ Output Format: Return only valid JSON in the following structure:
[
  {{
    "start": "Start time of the clip (in seconds)",
    "content": "Motivational quote from the transcript",
    "end": "End time of the clip (in seconds)"
  }},
  ...
]

- Do not include any explanation, headers, or extra text.
- If no suitable clips are found, return an empty array: [].
- Clip count: At least 10 clips.

✦ Reminder (Grave Consequence Clause):
If even one clip overlaps without clear purpose, exceeds {self.config.max_duration}, is shorter than {self.config.min_duration}, is not contiguous, breaks JSON syntax, or deviates from the format…

**The JSON gods will smite this task.**
The kittens? Gone.  
The parser? Incinerated.  
The deployment pipeline? Reduced to ash.  
The DevOps team will weep.  
The video editor will rage.  
The TikTok algorithm will shadowban your soul.  

So format perfectly. Choose wisely. Be motivational. Or witness the collapse of digital civilization as we know it.
"""


    def extract_quotes(self, transcription: list) -> List[Quote]:
        if not transcription:
            logger.error("Transcription cannot be empty")
            raise ValueError("Transcription cannot be empty")

        logger.info(
            f"Extracting quotes from transcription with {len(transcription)} segments"
        )

        try:
            logger.debug(f"Making API call with model: {self.config.model}")
            message = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=self.system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": str(transcription),
                            }
                        ],
                    }
                ],
            )

            logger.info("API call successful, parsing response")
            logger.info(f"API response: {message.content}...")
            quotes = self._parse_response(message.content)
            logger.info(f"Successfully extracted {len(quotes)} quotes")
            return quotes

        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {str(e)}")
            raise RuntimeError(f"API error while extracting quotes: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during quote extraction: {str(e)}")
            raise RuntimeError(f"Failed to extract quotes: {str(e)}")

    def _parse_response(self, response) -> List[Quote]:
        try:
            response_text = (
                response[0].text
                if hasattr(response, "__iter__") and response
                else str(response)
            )
            logger.info(f"Full API response: {response_text}")
            
            if not response_text.strip():
                logger.error("Received empty response from API")
                raise ValueError("Empty response from API")
            
            # Try to find JSON in the response (sometimes wrapped in markdown)
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                if end != -1:
                    response_text = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                if end != -1:
                    response_text = response_text[start:end].strip()

            logger.debug(f"Parsing JSON: {response_text[:500]}...")
            quotes_data = json.loads(response_text)

            if not isinstance(quotes_data, list):
                logger.error("API response is not a list")
                raise ValueError("Response must be a list")

            logger.debug(f"Processing {len(quotes_data)} quote candidates")
            quotes = []
            for idx, quote_data in enumerate(quotes_data):
                try:
                    quote = Quote(
                        start=float(quote_data["start"]),
                        content=quote_data["content"],
                        end=float(quote_data["end"]),
                    )
                    self._validate_quote(quote)
                    quotes.append(quote)
                    logger.debug(
                        f"Quote {idx + 1} validated: {quote.start}-{quote.end}s"
                    )
                except Exception as e:
                    logger.warning(f"Skipping invalid quote {idx + 1}: {str(e)}")
                    continue

            self._validate_no_overlaps(quotes)
            logger.info(f"Successfully parsed {len(quotes)} valid quotes")
            return quotes

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            raise ValueError("Invalid JSON response from API")
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Quote format error: {str(e)}")
            raise ValueError(f"Invalid quote format: {str(e)}")

    def _validate_quote(self, quote: Quote) -> None:
        if quote.start >= quote.end:
            logger.warning(
                f"Invalid time range: start {quote.start} >= end {quote.end}"
            )
            raise ValueError(
                f"Invalid time range: start {quote.start} >= end {quote.end}"
            )

        if quote.duration > self.config.max_duration:
            logger.warning(
                f"Quote duration {quote.duration}s exceeds max {self.config.max_duration}s"
            )
            raise ValueError(
                f"Quote duration {quote.duration}s exceeds max {self.config.max_duration}s"
            )

        if not quote.content.strip():
            logger.warning("Quote content is empty")
            raise ValueError("Quote content cannot be empty")

    def _validate_no_overlaps(self, quotes: List[Quote]) -> None:
        logger.debug("Validating quote overlaps")
        for i, quote1 in enumerate(quotes):
            for quote2 in quotes[i + 1 :]:
                if quote1.start < quote2.end and quote2.start < quote1.end:
                    logger.error(
                        f"Overlapping quotes detected: {quote1.start}-{quote1.end} and {quote2.start}-{quote2.end}"
                    )
                    raise ValueError(
                        f"Overlapping quotes detected: {quote1.start}-{quote1.end} and {quote2.start}-{quote2.end}"
                    )
        logger.debug("No overlapping quotes found")
