---
trigger: always_on
---

# Python Module Structure

- Use micro API-based project structure: `{mod}/lib.py`, `{mod}/test.py`
- Declare `__all__` in `lib.py` to define public exports
- Module `__init__.py` must export internals for api consumers
- Major feature modules must use `{feature}/{api1}`, `{feature}/{api2}`, etc. like structure
