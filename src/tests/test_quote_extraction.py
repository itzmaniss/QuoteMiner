import pytest
import json
import os
import sys
from unittest.mock import Mock, patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import QuoteExtractor


class TestJSONExtraction:
    def setup_method(self):
        """Setup test instance."""
        self.extractor = QuoteExtractor()
    
    def test_extract_json_with_preamble_text(self):
        """Test extraction when AI model adds explanatory text before JSON."""
        sample_response = '''Here are 3 non-overlapping, high-impact, motivational clips from the transcript:

[
  {
    "start": 53.28,
    "content": "we all know that we are supposed to pay attention to our emotions, but at the same time, we are often told that we shouldn't take all of our emotions seriously, nor should we react to all of our emotions with behaviors. And indeed, that is true. What's been lacking, however, and what Dr. Mark Brackett finally delivers to us, is a roadmap to think about our emotions in a very structured way, and thereby to engage with our emotions, sometimes shift our emotions, and certainly to understand the emotional expressions of others.",
    "end": 82.72
  },
  {
    "start": 116.16,
    "content": "to improve your relationship to yourself and to others.",
    "end": 120.8
  },
  {
    "start": 333.68,
    "content": "Also presenting at Udomonium are other excellent scientists and clinicians who have appeared on the Huberman Lab podcast, including Dr. Sarah Gottfried, Dr. Zachary Knight, and Dr. Robin Carthart Harris, along with nearly 70 other experts.",
    "end": 347.04
  }
]'''
        
        extracted_json = self.extractor._extract_json_from_response(sample_response)
        
        # Should successfully extract valid JSON
        parsed = json.loads(extracted_json)
        assert isinstance(parsed, list)
        assert len(parsed) == 3
        
        # Verify first item structure
        first_item = parsed[0]
        assert "start" in first_item
        assert "content" in first_item  
        assert "end" in first_item
        assert first_item["start"] == 53.28
        assert first_item["end"] == 82.72
    
    def test_extract_json_from_markdown_block(self):
        """Test extraction from markdown code block."""
        sample_response = '''Here's your JSON:

```json
[
  {
    "start": 10.0,
    "content": "test quote",
    "end": 20.0
  }
]
```

Hope this helps!'''
        
        extracted_json = self.extractor._extract_json_from_response(sample_response)
        parsed = json.loads(extracted_json)
        
        assert len(parsed) == 1
        assert parsed[0]["start"] == 10.0
        assert parsed[0]["content"] == "test quote"
    
    def test_extract_json_from_generic_code_block(self):
        """Test extraction from generic code block without language tag."""
        sample_response = '''```
[
  {
    "start": 15.5,
    "content": "another test quote",
    "end": 25.5
  }
]
```'''
        
        extracted_json = self.extractor._extract_json_from_response(sample_response)
        parsed = json.loads(extracted_json)
        
        assert len(parsed) == 1
        assert parsed[0]["start"] == 15.5
    
    @patch('builtins.open', create=True)
    @patch('os.makedirs')
    def test_save_failed_response(self, mock_makedirs, mock_open):
        """Test that failed responses are saved to debug file."""
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        failed_response = "This is not valid JSON at all!"
        
        self.extractor._save_failed_response(failed_response)
        
        # Verify directory creation
        mock_makedirs.assert_called_once_with("debug_responses", exist_ok=True)
        
        # Verify file was written to
        mock_open.assert_called_once()
        mock_file.write.assert_called()
        
        # Check that the failed response was written
        written_content = ''.join(call.args[0] for call in mock_file.write.call_args_list)
        assert "FAILED JSON EXTRACTION" in written_content
        assert failed_response in written_content
    
    def test_bracket_extraction_fallback(self):
        """Test extraction using bracket matching as fallback."""
        sample_response = '''Some random text before
        [{"start": 5.0, "content": "test", "end": 10.0}]
        Some random text after'''
        
        extracted_json = self.extractor._extract_json_from_response(sample_response)
        parsed = json.loads(extracted_json)
        
        assert len(parsed) == 1
        assert parsed[0]["start"] == 5.0
    
    def test_clean_json_only(self):
        """Test extraction when model provides only clean JSON without extra text."""
        clean_json_response = '''[
  {
    "start": 30.5,
    "content": "This is a perfect motivational quote that follows all the rules.",
    "end": 45.2
  },
  {
    "start": 60.0,
    "content": "Another clean quote with proper formatting.",
    "end": 75.8
  }
]'''
        
        extracted_json = self.extractor._extract_json_from_response(clean_json_response)
        parsed = json.loads(extracted_json)
        
        assert len(parsed) == 2
        assert parsed[0]["start"] == 30.5
        assert parsed[0]["end"] == 45.2
        assert parsed[1]["start"] == 60.0
        assert "perfect motivational quote" in parsed[0]["content"]


if __name__ == "__main__":
    pytest.main([__file__])