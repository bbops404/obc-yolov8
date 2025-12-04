

set -ex



pip check
python -c "from importlib.metadata import version; assert(version('more-itertools')=='10.8.0')"
pytest -v tests
exit 0
