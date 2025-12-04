

set -ex



pip check
typer --help
pytest -v tests  --deselect=tests/test_tutorial/test_multiple_values/test_arguments_with_multiple_values/test_tutorial001.py::test_main --deselect=tests/test_tutorial/test_parameter_types/test_number/test_tutorial001.py::test_invalid_score --deselect=tests/test_tutorial/test_parameter_types/test_number/test_tutorial001_an.py::test_invalid_score
exit 0
