

set -ex



pip check
python -c "from importlib.metadata import version; assert(version('readchar')=='4.2.1')"
exit 0
