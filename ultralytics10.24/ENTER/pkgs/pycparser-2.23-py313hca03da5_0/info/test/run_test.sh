

set -ex



pip check
python -c "from importlib.metadata import version; assert(version('pycparser')=='2.23')"
pytest --rootdir=. -vv tests
exit 0
