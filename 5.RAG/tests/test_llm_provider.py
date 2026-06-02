"""Unit tests for the cloud LLM provider module."""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Ensure the src directory is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llm_provider import (
    PROVIDER_PRESETS,
    LLMConfig,
    detect_provider,
    get_llm_config,
    chat_completion,
    chat_completion_stream,
    _strip_think_tags,
)


class TestProviderPresets(unittest.TestCase):
    """Verify built-in provider presets."""

    def test_atlas_cloud_preset_exists(self):
        self.assertIn("atlas_cloud", PROVIDER_PRESETS)

    def test_minimax_preset_exists(self):
        self.assertIn("minimax", PROVIDER_PRESETS)

    def test_openai_preset_exists(self):
        self.assertIn("openai", PROVIDER_PRESETS)

    def test_minimax_preset_fields(self):
        p = PROVIDER_PRESETS["minimax"]
        self.assertEqual(p["base_url"], "https://api.minimax.io/v1")
        self.assertEqual(p["default_model"], "MiniMax-M3")
        self.assertEqual(p["env_key"], "MINIMAX_API_KEY")

    def test_atlas_cloud_preset_fields(self):
        p = PROVIDER_PRESETS["atlas_cloud"]
        self.assertEqual(p["base_url"], "https://api.atlascloud.ai/v1")
        self.assertEqual(p["default_model"], "deepseek-ai/DeepSeek-V3-0324")
        self.assertEqual(p["env_key"], "ATLAS_CLOUD_API_KEY")

    def test_openai_preset_fields(self):
        p = PROVIDER_PRESETS["openai"]
        self.assertEqual(p["env_key"], "OPENAI_API_KEY")


class TestDetectProvider(unittest.TestCase):
    """Test auto-detection from environment variables."""

    @patch.dict(os.environ, {"ATLAS_CLOUD_API_KEY": "atlas-test"}, clear=False)
    def test_detects_atlas_cloud(self):
        self.assertEqual(detect_provider(), "atlas_cloud")

    @patch.dict(
        os.environ,
        {"MINIMAX_API_KEY": "test-key", "ATLAS_CLOUD_API_KEY": "", "ATLAS_CLOUD_MODEL": ""},
        clear=False,
    )
    def test_detects_minimax(self):
        self.assertEqual(detect_provider(), "minimax")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False)
    def test_detects_openai(self):
        # Remove MINIMAX key to ensure OpenAI is detected
        env = os.environ.copy()
        env.pop("MINIMAX_API_KEY", None)
        env.pop("ATLAS_CLOUD_API_KEY", None)
        env.pop("ATLAS_CLOUD_MODEL", None)
        with patch.dict(os.environ, env, clear=True):
            os.environ["OPENAI_API_KEY"] = "sk-test"
            self.assertEqual(detect_provider(), "openai")

    @patch.dict(os.environ, {}, clear=True)
    def test_returns_none_when_no_keys(self):
        self.assertIsNone(detect_provider())

    @patch.dict(
        os.environ,
        {
            "ATLAS_CLOUD_API_KEY": "atlas-key",
            "MINIMAX_API_KEY": "mm-key",
            "OPENAI_API_KEY": "sk-key",
        },
        clear=False,
    )
    def test_atlas_has_priority(self):
        self.assertEqual(detect_provider(), "atlas_cloud")


class TestGetLLMConfig(unittest.TestCase):
    """Test LLMConfig construction and validation."""

    @patch.dict(os.environ, {"ATLAS_CLOUD_API_KEY": "atlas-test"}, clear=False)
    def test_atlas_cloud_config(self):
        cfg = get_llm_config(provider="atlas_cloud")
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg.provider, "atlas_cloud")
        self.assertEqual(cfg.model, "deepseek-ai/DeepSeek-V3-0324")
        self.assertEqual(cfg.base_url, "https://api.atlascloud.ai/v1")

    @patch.dict(
        os.environ,
        {
            "ATLAS_CLOUD_API_KEY": "atlas-test",
            "ATLAS_CLOUD_MODEL": "deepseek-ai/DeepSeek-V3.1",
        },
        clear=False,
    )
    def test_atlas_cloud_model_from_env(self):
        cfg = get_llm_config(provider="atlas_cloud")
        self.assertEqual(cfg.model, "deepseek-ai/DeepSeek-V3.1")

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "mm-test"}, clear=False)
    def test_minimax_config(self):
        cfg = get_llm_config(provider="minimax")
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg.provider, "minimax")
        self.assertEqual(cfg.model, "MiniMax-M3")
        self.assertEqual(cfg.base_url, "https://api.minimax.io/v1")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False)
    def test_openai_config(self):
        cfg = get_llm_config(provider="openai")
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg.provider, "openai")
        self.assertEqual(cfg.model, "gpt-4o-mini")

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "mm-key"}, clear=False)
    def test_custom_model_override(self):
        cfg = get_llm_config(provider="minimax", model="MiniMax-M2.7-highspeed")
        self.assertEqual(cfg.model, "MiniMax-M2.7-highspeed")

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "mm-key"}, clear=False)
    def test_minimax_temperature_clamping_zero(self):
        cfg = get_llm_config(provider="minimax", temperature=0.0)
        self.assertGreater(cfg.temperature, 0.0)

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "mm-key"}, clear=False)
    def test_minimax_temperature_clamping_high(self):
        cfg = get_llm_config(provider="minimax", temperature=2.0)
        self.assertLessEqual(cfg.temperature, 1.0)

    @patch.dict(os.environ, {}, clear=True)
    def test_returns_none_without_api_key(self):
        cfg = get_llm_config(provider="minimax")
        self.assertIsNone(cfg)

    def test_returns_none_for_unknown_provider(self):
        cfg = get_llm_config(provider="nonexistent")
        self.assertIsNone(cfg)

    @patch.dict(
        os.environ,
        {"MINIMAX_API_KEY": "mm-key", "ATLAS_CLOUD_API_KEY": "", "ATLAS_CLOUD_MODEL": ""},
        clear=False,
    )
    def test_auto_detect_config(self):
        cfg = get_llm_config()
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg.provider, "minimax")


class TestStripThinkTags(unittest.TestCase):
    """Test removal of <think> blocks."""

    def test_strips_think_tags(self):
        text = "<think>reasoning</think>Final answer."
        self.assertEqual(_strip_think_tags(text), "Final answer.")

    def test_strips_multiline_think(self):
        text = "<think>\nstep 1\nstep 2\n</think>\nAnswer here."
        self.assertEqual(_strip_think_tags(text), "Answer here.")

    def test_no_think_tags(self):
        text = "Plain answer."
        self.assertEqual(_strip_think_tags(text), "Plain answer.")

    def test_empty_string(self):
        self.assertEqual(_strip_think_tags(""), "")


class TestChatCompletion(unittest.TestCase):
    """Test chat_completion with mocked OpenAI client."""

    @patch.dict(os.environ, {"ATLAS_CLOUD_API_KEY": "atlas-test"}, clear=False)
    def test_chat_completion_stream_returns_text(self):
        cfg = get_llm_config(provider="atlas_cloud")
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock(delta=MagicMock(content="Hello "))]
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock(delta=MagicMock(content="Atlas"))]
        chunk3 = MagicMock()
        chunk3.choices = [MagicMock(delta=MagicMock(content=None))]

        with patch("openai.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = [chunk1, chunk2, chunk3]
            MockOpenAI.return_value = mock_client

            result = chat_completion_stream(cfg, [{"role": "user", "content": "hi"}])
            self.assertEqual(result, "Hello Atlas")

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "mm-test"}, clear=False)
    def test_chat_completion_returns_text(self):
        cfg = get_llm_config(provider="minimax")
        mock_msg = MagicMock()
        mock_msg.content = "Test response"
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("openai.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            MockOpenAI.return_value = mock_client

            result = chat_completion(cfg, [{"role": "user", "content": "hi"}])
            self.assertEqual(result, "Test response")

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "mm-test"}, clear=False)
    def test_chat_completion_strips_think_tags(self):
        cfg = get_llm_config(provider="minimax")
        mock_msg = MagicMock()
        mock_msg.content = "<think>internal</think>Clean answer"
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("openai.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            MockOpenAI.return_value = mock_client

            result = chat_completion(cfg, [{"role": "user", "content": "hi"}])
            self.assertEqual(result, "Clean answer")

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "mm-test"}, clear=False)
    def test_chat_completion_passes_config(self):
        cfg = get_llm_config(provider="minimax", temperature=0.5, max_tokens=512)
        mock_msg = MagicMock()
        mock_msg.content = "ok"
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("openai.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            MockOpenAI.return_value = mock_client

            chat_completion(cfg, [{"role": "user", "content": "test"}])

            MockOpenAI.assert_called_once_with(
                api_key=cfg.api_key, base_url=cfg.base_url
            )
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            self.assertEqual(call_kwargs["model"], "MiniMax-M3")
            self.assertEqual(call_kwargs["temperature"], 0.5)
            self.assertEqual(call_kwargs["max_tokens"], 512)

    @patch.dict(os.environ, {"ATLAS_CLOUD_API_KEY": "atlas-test"}, clear=False)
    def test_chat_completion_stream_passes_stream_flag(self):
        cfg = get_llm_config(provider="atlas_cloud", temperature=0.3, max_tokens=256)
        event = MagicMock()
        event.choices = [MagicMock(delta=MagicMock(content="ok"))]

        with patch("openai.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = [event]
            MockOpenAI.return_value = mock_client

            chat_completion_stream(cfg, [{"role": "user", "content": "test"}])

            MockOpenAI.assert_called_once_with(
                api_key=cfg.api_key, base_url=cfg.base_url
            )
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            self.assertEqual(call_kwargs["model"], "deepseek-ai/DeepSeek-V3-0324")
            self.assertEqual(call_kwargs["temperature"], 0.3)
            self.assertEqual(call_kwargs["max_tokens"], 256)
            self.assertTrue(call_kwargs["stream"])


class TestCloudAnswerGeneration(unittest.TestCase):
    """Test cloud answer/summarize wrappers in generate_answer module."""

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "mm-test"}, clear=False)
    def test_cloud_answer_with_context(self):
        from generate_answer import cloud_answer_with_context

        cfg = get_llm_config(provider="minimax")

        with patch("openai.OpenAI") as MockOpenAI:
            mock_msg = MagicMock()
            mock_msg.content = "The vulnerability is critical."
            mock_choice = MagicMock()
            mock_choice.message = mock_msg
            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            MockOpenAI.return_value = mock_client

            result = cloud_answer_with_context(
                "Context about CVE-2024-1234.", "What is the severity?", cfg
            )
            self.assertEqual(result, "The vulnerability is critical.")

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "mm-test"}, clear=False)
    def test_cloud_answer_empty_context(self):
        from generate_answer import cloud_answer_with_context

        cfg = get_llm_config(provider="minimax")
        result = cloud_answer_with_context("", "question?", cfg)
        self.assertIn("No relevant context", result)

    def test_cloud_answer_falls_back_to_bart(self):
        """When llm_config is None, falls back to local BART."""
        from generate_answer import cloud_answer_with_context

        with patch("generate_answer.answer_with_context") as mock_bart:
            mock_bart.return_value = "BART answer"
            result = cloud_answer_with_context("ctx", "q?", None)
            mock_bart.assert_called_once_with("ctx", "q?")
            self.assertEqual(result, "BART answer")

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "mm-test"}, clear=False)
    def test_cloud_summarize_document(self):
        from generate_answer import cloud_summarize_document

        cfg = get_llm_config(provider="minimax")

        with patch("openai.OpenAI") as MockOpenAI:
            mock_msg = MagicMock()
            mock_msg.content = "Summary of the report."
            mock_choice = MagicMock()
            mock_choice.message = mock_msg
            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            MockOpenAI.return_value = mock_client

            result = cloud_summarize_document(
                ["chunk1", "chunk2"], max_summary_length=200, llm_config=cfg
            )
            self.assertEqual(result, "Summary of the report.")

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "mm-test"}, clear=False)
    def test_cloud_summarize_empty_chunks(self):
        from generate_answer import cloud_summarize_document

        cfg = get_llm_config(provider="minimax")
        result = cloud_summarize_document([], llm_config=cfg)
        self.assertIn("No content available", result)

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "mm-test"}, clear=False)
    def test_cloud_summarize_by_sections(self):
        from generate_answer import cloud_summarize_by_sections

        cfg = get_llm_config(provider="minimax")

        with patch("openai.OpenAI") as MockOpenAI:
            mock_msg = MagicMock()
            mock_msg.content = "Section summary."
            mock_choice = MagicMock()
            mock_choice.message = mock_msg
            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            MockOpenAI.return_value = mock_client

            results = cloud_summarize_by_sections(
                ["a", "b", "c", "d"], section_size=2, llm_config=cfg
            )
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0], "Section summary.")


if __name__ == "__main__":
    unittest.main()
