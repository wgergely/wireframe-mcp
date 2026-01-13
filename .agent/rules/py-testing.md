---
trigger: always_on
---

- Use `pytest`
- Do not import `unittest`. Do not use unittest fixtures, mocks, patches
- Use pytest.ini to define markers (include unit and integration)
- Use conftest.py to define 
- Ensure pytest plugins are defined and utilised
- Ensure every test case is pytest native and has supported pytest markers
- DO NOT use arbitary test file names. WRONG: test_new_providers.py GOOD: test.py