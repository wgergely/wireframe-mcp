# Development Rules

## Code Style
- Code style: use Google style documentation
- Use type hinting
- Use Python 3.12 compatibility for type hinting
- Use ruff format and check for code cleanliness.

## Testing
Use pytest only. Never import unittest. Use pytest.ini to define a minimal set of markers that include unit and integration. pytest plugins for async and MCP and FastAPI like testing environment. Make sure every test case is pytest native and has proper pytest markers. 
Never import unittest or unittest mock/patch. 
