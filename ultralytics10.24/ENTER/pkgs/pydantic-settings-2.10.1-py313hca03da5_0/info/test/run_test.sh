

set -ex



pip check
pytest -v tests --ignore=tests/test_docs.py
exit 0
