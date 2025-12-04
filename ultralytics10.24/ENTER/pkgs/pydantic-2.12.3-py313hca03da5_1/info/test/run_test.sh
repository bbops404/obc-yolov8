

set -ex



pip check
python -c "from importlib.metadata import version; assert(version('pydantic')=='2.12.3')"
pytest -ra -vv --tb=short tests  --ignore=tests/test_docs.py  --deselect=tests/test_types_typeddict.py::test_readonly_qualifier_warning
exit 0
