# Benchmark de transcrição de documentos históricos

Mede quanto uma página do Diário Oficial (1890–1950) sobrevive quando é
transcrita por um modelo multimodal. Cada sistema recebe a **imagem** da página
e um prompt; o que ele devolve é comparado com a **referência** — a transcrição
conferida à mão, em HTML — por cinco medidas.

São avaliados modelos de API (OpenRouter e a API nativa do Google) e o MinerU,
um analisador de documentos que roda na própria máquina. O eixo da comparação é
**aberto × fechado** e **qualidade × custo**: quanto se paga, em dinheiro e em
tempo, por cada ponto de precisão.

---

## As cinco medidas

| medida | o que responde | onde está |
|---|---|---|
| **WER** | quantas *palavras* da página o sistema errou | `bench/metrics.py` |
| **CER** | quantos *caracteres* da página o sistema errou | `bench/metrics.py` |
| **Custo de token** | tokens de entrada, de saída e US$ por chamada | vem do provedor, gravado em `bench/clients.py` |
| **`html_sem_texto_extra`** | obedeceu ao "retorne apenas o código HTML sem texto extra"? | `bench/metrics.py` |
| **`num_f1`** | recuperou os números — réis, datas, processos? | `bench/metrics.py` |

**WER e CER** são a mesma conta em unidades diferentes: distância de Levenshtein
dividida pelo tamanho da referência, isto é `(S + D + I) / N_ref`. Podem passar
de 1,0 — uma alucinação longa acrescenta inserções sem aumentar o denominador.
Antes de comparar, os dois lados passam pela mesma normalização
(`bench/normalize.py`): a marcação vira texto, a tipografia é canonizada, tudo
vai a minúsculas, a pontuação sai das bordas dos tokens e as palavras quebradas
por hífen no fim da linha são reunidas. Sem isso o WER mediria coincidência de
estilo, não fidelidade de transcrição.

**A casa de hospedagem é fixada.** Dentro do OpenRouter o mesmo modelo é
servido por várias casas, e elas não são equivalentes: o Kimi K3 tem 18, com
quantização fp4, mxfp4, fp8 ou bf16 conforme quem atende. Sem fixar, a casa muda
a cada chamada e a variação entra na média como se fosse propriedade do modelo.
Cada modelo declara a sua em `endpoint_tag` (sempre a casa **de origem** —
Alibaba para os Qwen, Anthropic para os Claude, OpenAI para o GPT, Moonshot AI
para o Kimi), `bench check` confere que o nome existe, e
`permitir_fallback_de_hospedagem: false` faz a chamada **falhar** em vez de ser
desviada em silêncio para outra casa. O `provider_upstream` gravado em cada
chamada é a prova de quem serviu.

**Custo de token** é medido, não estimado. Pelo OpenRouter vem o valor
realmente cobrado (`cost_source` = `usage` ou `generation_api`); pela API do
Google, que devolve tokens mas não valor, vem da tabela declarada em
`models.yaml` (`cost_source` = `tabela_precos`). Ao comparar custo entre
provedores, essa diferença de procedência precisa ser dita. O MinerU roda local:
custo zero, e os campos de token ficam **nulos de propósito** — não há
tokenizador comum entre ele e os LLMs, e inventar uma contagem o faria entrar
nas médias de custo por token como se fosse comparável.

**`html_sem_texto_extra`** é a obediência ao formato pedido, medida à parte da
qualidade: depois de tirar uma cerca de código que envolva a resposta inteira
(```` ```html … ``` ````, que é formatação e não texto), o que sobra tem de ser
HTML e tem de começar em `<` e terminar em `>`. Um "Claro! Aqui está a
transcrição:" na frente, ou um "Espero ter ajudado" no fim, quebra uma das duas
pontas. O WER quase não sente essas frases; quem for usar a saída, sim.

A coluna sai **vazia** nas chamadas sob o prompt sentinela `p0` — o MinerU nunca
recebeu a instrução "retorne apenas o código HTML", e marcá-lo como `False`
seria registrar desobediência a uma ordem que não foi dada. Vazio é o que ela
significa ali: não se aplica. Essas linhas também ficam de fora da média no
`resumo.csv` e do gráfico de obediência.

**`num_f1`** compara os números por multiconjunto: interessa se os valores da
página foram recuperados, não onde foram parar. Um modelo pode ter WER baixo e
ser inútil para pesquisa histórica se errar sistematicamente os réis.

---

## Instalação

### 1. O benchmark

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. As chaves de API

Lidas do ambiente, nunca de arquivo versionado:

```powershell
$env:OPENROUTER_API_KEY = 'sk-or-...'   # a maioria dos modelos
$env:GEMINI_API_KEY     = '...'         # só para os `provider: gemini`
```

Só é preciso a chave dos provedores que você for usar: os clientes são
construídos sob demanda, e uma execução só com MinerU não pede nenhuma.

### 3. O MinerU (opcional)

Roda num ambiente virtual separado, porque exige Python 3.10–3.12 enquanto o
benchmark roda em 3.13+:

```powershell
py install 3.12                              # se ainda não houver um 3.12
.\ferramentas\mineru\setup_mineru.ps1        # GPU NVIDIA (CUDA 13.2)
.\ferramentas\mineru\setup_mineru.ps1 -Cpu   # sem GPU
```

São ~3,2 GB de dependências, mais os pesos baixados na primeira transcrição.
Sem esse passo, os quatro modelos `opendatalab/*` falham com uma mensagem
dizendo isso; os de API continuam funcionando normalmente.

> **O venv não é relocável.** Ele é criado com o caminho absoluto embutido nos
> executáveis de `Scripts/`. Se a pasta do projeto for movida depois, o
> `mineru.exe` para de achar o próprio Python. A correção é regenerar os
> executáveis sem tocar nas dependências:
>
> ```powershell
> .\.venv-mineru\Scripts\python.exe -m ensurepip --upgrade
> .\.venv-mineru\Scripts\python.exe -m pip install --force-reinstall --no-deps mineru==3.4.4
> ```

### 4. Confira antes de gastar

```powershell
python -m bench check
```

Valida os três YAML, lista os documentos e os modelos habilitados, diz quais
chaves faltam, mostra onde está o MinerU e confere contra o catálogo do
OpenRouter que cada `endpoint_tag` declarado existe — com o preço da faixa que
ele fixa.

---

## Uso

```powershell
python -m bench check                  # dataset, modelos, prompts, chaves
python -m bench run --dry-run          # quais chamadas seriam feitas
python -m bench run                    # faz as chamadas (gasta dinheiro)
python -m bench score                  # calcula as métricas da última execução
python -m bench graficos               # desenha as figuras (todas as execuções)
python -m bench all                    # run + score + gráficos
```

Recortes úteis:

```powershell
python -m bench run --models qwen/qwen3-vl-235b-a22b-instruct   # um modelo só
python -m bench run --docs 1891-08-01_p606 1900-04-01_p735      # documentos escolhidos
python -m bench run --all-docs                                  # todos os 16
python -m bench run --resume 20260902-1030                      # retoma o que faltou
python -m bench score --run-id 20260902-1030                    # repontua uma execução antiga
python -m bench graficos --run-ids 20260902-1030                # figuras de uma execução só
```

Por padrão só rodam os documentos listados em `dataset.active`
(`config/settings.yaml`). `run --resume` refaz apenas o que faltou ou falhou:
nenhuma chamada bem-sucedida é paga duas vezes.

---

## O que fica em disco

```
results/<run_id>/
  chamadas.jsonl        o índice: uma linha por chamada (tokens, custo, tempo, erro)
  chamadas/<modelo>/<doc>__<prompt>__r<n>/
      resposta.html     O TEXTO DEVOLVIDO (.md ou .txt se o formato for outro)
      bruto.json        a resposta do provedor VERBATIM, uma entrada por tentativa
      normalizado.txt   esse texto depois da normalização — o que o WER/CER comparam
      referencia.txt    a referência depois da MESMA normalização
  mineru/<modelo>/      artefatos do MinerU: layout.pdf, content_list.json, log
  metricas.csv          uma linha por chamada, com as cinco medidas
  resumo.csv            uma linha por (modelo, prompt), com as médias
                        (`n_validas` conta chamadas; `n_paginas`, páginas distintas)

results/graficos/       as sete figuras, sobre TODAS as execuções
```

**Tudo que pertence a uma chamada mora na pasta dela.** Conferir uma
transcrição é abrir uma pasta, não cruzar três árvores paralelas. O
`chamadas.jsonl` é o índice — responde "quanto custou, quanto demorou, o que
falhou" sem abrir pasta nenhuma; a pasta responde "o que exatamente esse modelo
devolveu".

`chamadas.jsonl` e as pastas de chamada são o dado primário — custaram dinheiro.
`metricas.csv`, `resumo.csv` e as figuras são derivados: `score` e `graficos` os regeneram a qualquer
momento, inclusive depois de você mudar a definição de uma métrica. É por isso
que a coleta é separada do cálculo.

No `resumo.csv`, **confiabilidade e qualidade não se misturam**: `taxa_sucesso`
e `taxa_truncamento` são calculadas sobre *todas* as chamadas; WER, CER, num_f1
e custo, só sobre as **válidas** (respondeu e não foi truncada). Misturar as
duas coisas premiaria quem falha, porque quem falha muito acaba avaliado só nos
casos fáceis em que conseguiu responder.

> **Truncamento.** As páginas são longas (3 colunas + tabelas). Uma resposta
> cortada no teto de tokens produz um WER catastrófico que *parece*
> incompetência do modelo. Ela é marcada com `truncated=true` e fica fora das
> médias de qualidade. O teto está em `generation.max_tokens`.

### Conferir um número na mão

A coluna `pasta` do `metricas.csv` leva direto à chamada. Lá dentro:

| arquivo | o que é |
|---|---|
| `resposta.html` | só o texto que o modelo devolveu (`.md` para o MinerU, `.txt` se vier texto corrido — a extensão segue o formato detectado) |
| `bruto.json` | a resposta do provedor como ela chegou, **uma entrada por tentativa** — se a chamada só passou na terceira, os dois erros anteriores estão lá, com corpo e status |
| `normalizado.txt` | esse texto depois da normalização |
| `referencia.txt` | a referência depois da **mesma** normalização |

Para auditar um WER, um diff entre os dois últimos:

```powershell
$p = "results/<run_id>/chamadas/<modelo>/<doc>__p1__r0"
git diff --no-index "$p/referencia.txt" "$p/normalizado.txt"
```

A diferença que a métrica contou está ali, na forma exata em que ela a contou —
sem marcação, em minúsculas, sem hifenização de fim de linha. Os tokens do WER
são esse arquivo separado nos espaços, com a pontuação tirada das bordas; o CER
usa o mesmo texto com as quebras de linha viradas espaço.

A referência é repetida em cada pasta de propósito: são ~8 KB, e é o que faz o
diff ser um comando só, sem navegar.

Uma chamada que falhou tem só o `bruto.json` — que é justamente onde está o
motivo. Ele responde às perguntas que aparecem meses depois: quantos tokens
nativos o provedor contou, qual foi o corpo do erro 429, qual casa serviu o
modelo aberto. Guardá-lo é assimétrico: o disco é barato e a chamada não.

---

## Gráficos

`python -m bench graficos` escreve sete PNG em `results/graficos/`,
agregando **todas as execuções** — não só a última:

| figura | o que mostra |
|---|---|
| `cer_wer_por_modelo.png` | CER e WER lado a lado, ordenado pelo **CER** |
| `num_f1_por_modelo.png` | fidelidade dos números |
| `custo_beneficio.png` | CER x custo por página — onde cada sistema cai |
| `custo_n_paginas.png` | a conta em dinheiro do acervo, em quatro papéis |
| `tempo_do_acervo.png` | a conta em tempo do acervo, modelo a modelo |
| `html_sem_texto_extra.png` | % das respostas que vieram só com o HTML |
| `erro_por_dificuldade.png` | quanto a complexidade da página custa em erro |

O mesmo comando reescreve a **tabela de resultados** dentro de
`results/relatorio.html`, a partir dos mesmos dados que alimentam as figuras.
A tabela mora entre dois comentários marcadores e tudo que está fora deles é
texto do autor, que nunca é tocado. Sistemas que ainda não cobriram todas as
páginas do conjunto recebem a etiqueta `parcial` e ficam fora da disputa pelo
negrito de melhor valor, porque uma média sobre uma página não é comparável a
uma média sobre dezesseis. Se os marcadores não existirem no arquivo, o
comando avisa e não altera nada.

A agregação é **por chamada**, não por média de médias: as linhas de todos os
`metricas.csv` entram na mesma conta, então uma execução de 16 páginas pesa
dezesseis vezes mais que uma de 1. `--run-ids A B` restringe a execuções
específicas.

**Cada célula do desenho entra uma vez.** Uma página remedida — porque a
execução foi retomada, ou simplesmente refeita — aparece em mais de um
`metricas.csv`, e empilhá-los cruamente a faria pesar duas vezes na média do
modelo. Vale a medição **mais recente entre as válidas**: a remedição existe
justamente para consertar o que falhou, então a ordem crua deixaria uma falha
posterior apagar o sucesso que ela veio corrigir. O comando diz quantas
remedições descartou.

**Um sistema pode sair da análise, com o motivo escrito.** O campo `excluido`
no `models.yaml` guarda a justificativa; quem o tem não é chamado, não aparece
em nenhuma figura nem na tabela, e as medições que já produziu continuam no
disco — sair da análise não é deixar de ter existido. É diferente de
`enabled: false`, que só diz "não chame". O motivo é impresso pelo `bench check`
e pelo `bench graficos`, para a exclusão nunca ser silenciosa. Hoje há um:
o `google/gemini-3.7-flash` pela API nativa do Google, que rendeu 9 transcrições
válidas em 58 tentativas (15,5%) — o mesmo modelo segue no estudo pela via
OpenRouter, com as 16 páginas.

**Todo `model_id` medido é conferido contra o `models.yaml`.** Os CSV são um
histórico e guardam sistemas renomeados ou retirados do registro; sem a
conferência, um `model_id` órfão entra nas figuras como se fosse mais um sistema
do estudo e ninguém tem onde ler o que ele é. Nada é removido — o comando avisa,
nomeando o órfão.

**Todo rótulo carrega o `n`** — o número de **páginas distintas** por trás da
média. Marcar só quem tem poucas deixaria implícito que as demais são
comparáveis entre si; uma média de 16 páginas não é a mesma afirmação que uma
de 1. É página, e não chamada, porque uma página remedida não é uma página a
mais.

**Superlativo só entre quem cobriu o conjunto.** A mesma regra que dá a etiqueta
`parcial` na tabela decide quem disputa "melhor qualidade", "mais barato" e
"melhor custo-benefício". Os parciais continuam nas figuras — o dado existe —,
mas aparecem como ponto vazado no `custo_beneficio.png` e não recebem papel no
`custo_n_paginas.png`, que é a figura que multiplica o custo medido por milhões
de páginas. Se ninguém tiver cobertura completa, a disputa acontece entre
parciais e o subtítulo da figura diz isso.

O `tempo_do_acervo.png` faz a mesma projeção com a **latência** medida: o eixo
mostra o prazo em série, uma página de cada vez, que é exatamente como o tempo
foi medido — e o rótulo traduz isso em dias de calendário supondo 50 chamadas
simultâneas. O número serial fica no eixo porque é o observado; a concorrência é
premissa declarada, não medida. A espera do `rpm_limit` não entra na conta: ela
é decisão de configuração sua, não lentidão do modelo.

O `erro_por_dificuldade.png` é a única figura que agrega por **página** e não
por modelo. As imagens do `Dataset/` declaram o grau no fim do nome
(`1891_0599_Facil.jpg`), e `Document.dificuldade` lê dali: **fácil** é página
basicamente só de texto, **médio** tem algo a mais como uma tabela simples e
**difícil** tem estrutura composta. Cada transcrição de cada sistema conta como
uma observação da página, o que só é comparável se a cobertura for
equilibrada — condição que a figura **mede e declara no subtítulo**, em vez de
afirmar. A barra é
a média e o traço claro sobre ela é a mediana, e a distância entre as duas diz
se o grau é difícil para todos ou se poucos sistemas desabam nele. Uma imagem
sem grau no nome fica de fora e a figura declara quantas foram.

O `custo_n_paginas.png` projeta o custo medido por página para o tamanho de
acervo declarado em `PAGINAS_DO_PROJETO`, em `bench/graficos.py`, em quatro
papéis:
**melhor qualidade** (menor CER), **mais barato** (menor US$/página), **melhor
custo-benefício**, definido como `custo ÷ (1 − CER)` — o custo por página
efetivamente aproveitável —, e **melhor gratuito** (menor CER entre os que não
cobram). Um modelo que custa metade mas erra o dobro não é mais barato. Os dois
últimos existem porque "de graça" e "o mais barato que se paga" são pisos
diferentes, e a comparação entre eles é o que decide se vale abrir a carteira.
Quando o mesmo modelo ocupa dois papéis, os papéis aparecem juntos no rótulo —
e isso é o achado: não há trade-off a discutir.

Três decisões que não são gosto:

* **Nunca dois eixos y.** WER e CER cabem na mesma figura porque são a mesma
  unidade; custo e qualidade viram dispersão, não um segundo eixo.
* **O sistema local tem cor própria, em todas as figuras.** Cinza, porque
  rodar nesta máquina não é uma terceira categoria de licença — os MinerU são
  `tipo: aberto` — e sim uma condição que muda o que os eixos significam: custo
  por token não se aplica e o tempo mede esta GPU, não o modelo.
* **As cores são verificadas, não escolhidas a olho.** Azul `#2a78d6` e laranja
  `#eb6834` passam nos testes de daltonismo (ΔE 24,7 em protanopia) e de
  contraste contra o fundo. E a cor nunca é a única pista: toda barra tem o
  valor escrito na ponta e a legenda está sempre presente quando há duas séries.
* **Barras acima de 1,5 de erro são cortadas, com o valor real ao lado.** Um
  único caso extremo — um modelo que despeja um bloco base64 no meio do texto
  chega a CER 13 — esmagaria todas as outras barras contra o zero. A barra
  cortada ganha um marcador `▶` e o número verdadeiro: nada fica escondido.

Se quiser plotar por conta própria, `metricas.csv` tem uma linha por chamada e
serve de tabela para qualquer outra ferramenta.

---

## Os arquivos

```
config/models.yaml     quais sistemas, por qual provedor, com que preço e cota
config/prompts.yaml    p0 (sentinela, sem prompt) e p1 (o prompt do estudo)
config/settings.yaml   dataset ativo, API, MinerU, geração, execução
Dataset/<doc_id>/      uma imagem + o HTML de referência por documento

bench/config.py        lê e valida os três YAML
bench/dataset.py       descobre os documentos
bench/clients.py       OpenRouter, Gemini e MinerU — mesmo método, mesmo retorno
bench/runner.py        monta as chamadas, executa e grava
bench/normalize.py     HTML/Markdown -> texto -> tokens comparáveis
bench/metrics.py       WER, CER, num_f1, html_sem_texto_extra
bench/score.py         aplica as métricas e escreve os dois CSV
bench/graficos.py      as sete figuras
bench/relatorio.py     a tabela de resultados do relatorio.html
bench/cli.py           check | run | score | graficos | all
```

Duas regras que explicam o resto do código:

* **A coleta é separada do cálculo.** `run` só chama e grava; `score` mede
  depois, offline. A definição de uma métrica muda ao longo do trabalho; os
  dados pagos não podem depender dela.
* **Concorrência só entre modelos.** Dentro de um mesmo modelo as chamadas são
  serializadas, e os quatro MinerU dividem uma fila só — dois processos
  simultâneos disputariam a mesma GPU e mediriam contenção, não desempenho.

E uma regra que explica o `p0`: o MinerU não recebe instrução em linguagem
natural. Ele roda **uma vez**, sob o prompt sentinela `p0`; os modelos
instruídos nunca rodam sob ele. Sem isso, cada sentinela acrescentado ao
registro dispararia uma rodada paga a mais em todos os modelos de API.

---

## Onde mexer para fazer X

| quero… | mexo em |
|---|---|
| acrescentar um modelo | `config/models.yaml` (só isso) |
| fixar/mudar o endpoint (casa + faixa) | `endpoint_tag` em `config/models.yaml` — confira com `bench check` |
| mudar o prompt | `config/prompts.yaml` |
| ativar mais documentos | `dataset.active` em `config/settings.yaml`, ou `--all-docs` |
| mudar o teto de tokens | `generation.max_tokens` em `config/settings.yaml` |
| mudar como o texto é normalizado | `bench/normalize.py` (é fixo, não tem chave em YAML) |
| acrescentar uma métrica | `bench/metrics.py` + a lista `COLUNAS` em `bench/score.py` |
| entender um número estranho | `results/<run_id>/chamadas/…` — a pasta daquela chamada, com os quatro arquivos |

---

## Escopo — o que este benchmark não mede

Delimitar isto importa tanto quanto o que ele mede:

* **Só um prompt instruído.** O registro tem `p0` (sentinela, para sistemas que
  não recebem instrução) e `p1`. Não há eixo de especificidade de prompt: a
  comparação é entre sistemas, com a instrução fixa.
* **Nada sobre modernização ortográfica.** Quando um modelo escreve `sábado`
  onde a página diz `sabbado`, isso entra no WER como um erro qualquer, junto
  com as leituras erradas. Separar as duas coisas exigiria uma métrica própria,
  alinhada com a lista de grafias de época de cada documento.
* **Nada sobre estrutura de tabela.** A árvore do HTML não é comparada; o texto
  das células entra na contagem de palavras em ordem de leitura, como o resto.
* **Nada sobre ordem de leitura.** Um sistema que lê as três colunas do jornal
  na ordem errada é punido pelo WER com a mesma severidade de quem erra as
  palavras — e os dois casos ficam indistinguíveis no número final.
* **Uma única chamada por página, por padrão.** `execution.repetitions`
  aumenta isso, mas o padrão não mede variância dentro do mesmo dia.
