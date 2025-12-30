import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Add the parent directory to sys.path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tools import get_weather
from src.rag import process_pdf, query_qdrant

class TestAgentCore(unittest.TestCase):

    # --- TEST 1: API Handling (Weather) ---
    @patch('src.tools.requests.get')
    def test_get_weather_success(self, mock_get):
        # Mock the API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "weather": [{"description": "clear sky"}],
            "main": {"temp": 25.0}
        }
        mock_get.return_value = mock_response

        # Call the function
        result = get_weather.invoke({"city": "London"})
        
        # Assertions (Fixed syntax here)
        self.assertIn("clear sky", result)
        self.assertIn("25.0", str(result))

    # --- TEST 2: Retrieval Logic (RAG) ---
    def test_rag_processing(self):
        # Test empty query behavior
        response = query_qdrant("xyz_impossible_string")
        self.assertIsInstance(response, str)

    # --- TEST 3: Tool Logic ---
    def test_weather_tool_structure(self):
        # Check if tool has correct name and args
        self.assertEqual(get_weather.name, "get_weather")
        self.assertTrue("city" in get_weather.args)

if __name__ == '__main__':
    unittest.main()