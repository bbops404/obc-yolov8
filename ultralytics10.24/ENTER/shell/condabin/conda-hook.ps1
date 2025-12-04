$Env:CONDA_EXE = "/Users/user/Documents/obc-yolov8/ultralytics10.24/ENTER/bin/conda"
$Env:_CONDA_EXE = "/Users/user/Documents/obc-yolov8/ultralytics10.24/ENTER/bin/conda"
$Env:_CE_M = $null
$Env:_CE_CONDA = $null
$Env:CONDA_PYTHON_EXE = "/Users/user/Documents/obc-yolov8/ultralytics10.24/ENTER/bin/python"
$Env:_CONDA_ROOT = "/Users/user/Documents/obc-yolov8/ultralytics10.24/ENTER"
$CondaModuleArgs = @{ChangePs1 = $True}

Import-Module "$Env:_CONDA_ROOT\shell\condabin\Conda.psm1" -ArgumentList $CondaModuleArgs

Remove-Variable CondaModuleArgs