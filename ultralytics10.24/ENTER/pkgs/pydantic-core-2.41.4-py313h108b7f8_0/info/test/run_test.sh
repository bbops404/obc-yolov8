

set -ex



pip check
python -c "from pydantic_core import PydanticUndefinedType"
pytest -v --ignore=tests/test_docstrings.py --ignore=tests/test_hypothesis.py --ignore=tests/validators/test_allow_partial.py --ignore=tests/validators/test_frozenset.py --ignore=tests/validators/test_list.py --ignore=tests/validators/test_set.py
exit 0
