

set -ex



pip check
python -c "from importlib.metadata import version; assert(version('idna')=='3.11')"
pytest -v tests
exit 0
