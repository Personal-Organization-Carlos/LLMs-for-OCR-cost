# MinerU: a linha de base local

O [MinerU](https://github.com/opendatalab/MinerU) não é um LLM: é um analisador
de documentos — detecção de layout, OCR, reconstrução de tabelas — que roda na
máquina, sem chave e sem custo por token. Entra no benchmark como quatro
"modelos", que são os três backends locais do MinerU 3.4 mais uma variação de
esforço.

## Instalação

```powershell
.\ferramentas\mineru\setup_mineru.ps1          # GPU NVIDIA (CUDA 13.2)
.\ferramentas\mineru\setup_mineru.ps1 -Cpu     # sem GPU
.\ferramentas\mineru\setup_mineru.ps1 -Cuda cu126
```

Cria `.venv-mineru/` na raiz do projeto — ~3,4 GB, fora do git. É o caminho que
`config/settings.yaml` declara em `mineru.venv`, e é por isso que ele é
relativo: o mesmo `settings.yaml` funciona em qualquer máquina.

Os pesos vêm do **ModelScope**, não do Hugging Face, na primeira execução. O
cache do HF usa symlinks, e criá-los no Windows exige um privilégio que nem
toda máquina concede (WinError 1314) — não é preferência, é a única fonte que
funciona ali.

## Execução

```powershell
python -m bench run --models opendatalab/mineru-pipeline opendatalab/mineru-vlm `
                             opendatalab/mineru-hybrid opendatalab/mineru-hybrid-high
python -m bench score
```

Os artefatos brutos de cada chamada — o Markdown, o `content_list.json`, o
`middle.json`, o `layout.pdf` e o `span.pdf` — ficam em
`results/<run_id>/mineru/<modelo>/<doc>__r<n>/`, junto do `mineru.log`
completo.

## Por que a versão está fixada

Os pesos e o formato de saída mudam entre versões, e a subpasta em que o
Markdown é gerado já mudou uma vez. Com a versão solta, dois resultados
coletados em semanas diferentes deixariam de ser comparáveis sem nenhum aviso.
