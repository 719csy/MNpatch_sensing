$ErrorActionPreference = "Stop"

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Config = Join-Path $Here "configs/fig4_ood_config.yaml"
$Script = Join-Path $Here "src/process_fig4_ood.py"

python $Script --config $Config
