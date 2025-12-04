

set -ex



pip check
python -c "from importlib.metadata import version; assert(version('semver')=='3.0.4')"
pytest -v tests
pysemver -h
exit 0
