<#
.SYNOPSIS
  Cria o ambiente virtual do MinerU em .venv-mineru, na raiz do projeto.

.DESCRIPTION
  O MinerU exige Python 3.10-3.12 e o benchmark roda em 3.13+. Os dois nao
  cabem no mesmo interpretador, dai um venv separado, cujo caminho e declarado
  em `config/settings.yaml` (mineru.venv: .venv-mineru) e nao descoberto - para
  que uma execucao feita hoje seja reproduzivel numa maquina diferente amanha.

  Sao ~3,4 GB entre dependencias e pesos, por isso o venv fica fora do git.

  Requer Python 3.12 disponivel. O script tenta, nesta ordem: `uv` (mais
  rapido, e o que criou o ambiente original), depois `py -3.12`.

.EXAMPLE
  .\ferramentas\mineru\setup_mineru.ps1
  .\ferramentas\mineru\setup_mineru.ps1 -Cpu      # sem GPU NVIDIA
#>
param(
  # Indice de wheels do PyTorch. cu132 = CUDA 13.2, que e o que a maquina
  # original usa. Troque conforme o driver, ou passe -Cpu.
  [string]$Cuda = "cu132",
  [switch]$Cpu
)

$ErrorActionPreference = "Stop"
$raiz = (Resolve-Path "$PSScriptRoot\..\..").Path
$venv = Join-Path $raiz ".venv-mineru"

if (Test-Path $venv) {
  Write-Host "$venv ja existe. Apague-o para recriar do zero." -ForegroundColor Yellow
  exit 0
}

if (Get-Command uv -ErrorAction SilentlyContinue) {
  uv venv --python 3.12 $venv
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
  py -3.12 -m venv $venv
} else {
  throw "Nem `uv` nem `py` encontrados. Instale o Python 3.12 ou o uv."
}

$py = Join-Path $venv "Scripts\python.exe"
& $py -m pip install --upgrade pip

$indice = if ($Cpu) { "https://download.pytorch.org/whl/cpu" }
          else      { "https://download.pytorch.org/whl/$Cuda" }

# O indice extra serve as wheels com sufixo local (+cu132 / +cpu), que o pip
# prefere por terem local version identifier maior. Sem ele vem a build
# generica, que no Windows arrasta dependencias nvidia-* de varios GB mesmo
# quando nao ha GPU.
& $py -m pip install --extra-index-url $indice -r (Join-Path $PSScriptRoot "requirements-mineru.txt")

& $py -m pip show mineru | Select-String "^Name|^Version"
Write-Host "`nPronto: $venv" -ForegroundColor Green
Write-Host "Os pesos sao baixados na primeira execucao, do ModelScope" -ForegroundColor Green
Write-Host "(MINERU_MODEL_SOURCE=modelscope, definido por config/settings.yaml)." -ForegroundColor Green
