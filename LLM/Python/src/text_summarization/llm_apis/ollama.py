import logging
import ollama

from text_summarization.llm_apis.base_client import BaseClient

logger = logging.getLogger(__name__)


class OllamaClient(BaseClient):
    def __init__(self, host='http://comma5.duckdns.org:11434'):
        self.host = host
        self.client = ollama.Client(host=self.host)

    def summarize(self, text: str, model_name: str, prompt: str, fallback_summary: str = None) -> str:
        """
        Ollama API summarization using official ollama Python library.

        Args:
            text: Input text to summarize
            model_name: Ollama model name (e.g., "granite3.1-dense:8b")
            prompt: Custom prompt template with placeholder for text
            fallback_summary: Fallback summary to return on error

        Returns:
            Generated summary or fallback_summary on error
        """
        try:
            formatted_prompt = prompt.format(text=text[:3000])

            logger.info(f"Making Ollama request with model {model_name}")
            response = self.client.generate(
                model=model_name,
                prompt=formatted_prompt,
                options={
                    'temperature': 0.3,
                    'top_k': 40,
                    'top_p': 0.9,
                    # 'num_predict': 200,  # TODO: update dynamically from min_words and max_words (depends on model ..)
                    'stop': ['\n\n', '---', 'References:', 'Bibliography:']
                }
            )

            if response and 'response' in response:
                summary = response['response'].strip()
                summary = summary.replace("Summary:", "").strip()
                summary = summary.split('\n')[0]

                if len(summary) < 20:
                    logger.warning(f"Ollama returned very short summary: {summary}")
                    if fallback_summary:
                        return fallback_summary
                    raise ValueError("Summary too short and no fallback provided")

                logger.info(f"Successfully generated summary with Ollama {model_name}")
                return summary
            else:
                logger.error(f"Unexpected response format from Ollama: {response}")
                if fallback_summary:
                    return fallback_summary
                raise ValueError("Invalid response format from Ollama")

        except ollama.ResponseError as e:
            logger.error(f"Ollama API error: {e}")
            if fallback_summary:
                return fallback_summary

        except ConnectionError as e:
            logger.error(f"Failed to connect to Ollama server: {e}")
            if fallback_summary:
                return fallback_summary

        except Exception as e:
            logger.error(f"Unexpected error in Ollama summarization: {e}")
            if fallback_summary:
                return fallback_summary
