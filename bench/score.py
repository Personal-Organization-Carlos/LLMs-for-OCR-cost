"""Cálculo das métricas sobre uma execução já coletada.

Roda inteiramente offline, a partir do que o `run` gravou. Pode ser refeito
quantas vezes for preciso — inclusive depois de mudar a definição de uma
métrica — sem gastar API.

Produz duas tabelas na raiz da execução:

    metricas.csv   uma linha por chamada, com tudo
    resumo.csv     uma linha por (modelo, prompt), com as médias

e, dentro da pasta de cada chamada, os dois lados da comparação já
normalizados — `normalizado.txt` e `referencia.txt` —, que é o que permite
conferir um número à mão.

A separação entre CONFIABILIDADE e QUALIDADE é mantida no resumo: taxa de
sucesso e taxa de truncamento se calculam sobre TODAS as chamadas; WER, CER,
num_f1 e custo, só sobre as válidas (`ok` e não truncadas). Misturar as duas
coisas premiaria quem falha, porque quem falha muito acaba avaliado só nos
casos fáceis em que conseguiu responder.
"""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean

from .config import Settings, load_models, load_settings, motivos_de_exclusao
from .dataset import discover_documents
from .metrics import calcular
from .normalize import prepare
from .runner import CHAMADAS, NORMALIZADO, REFERENCIA, ler_chamadas

METRICAS = "metricas.csv"
RESUMO = "resumo.csv"

# Colunas do metricas.csv, nesta ordem.
COLUNAS = [
    "run_id", "empresa", "tipo", "model_id", "model_label", "api_provider",
    "provider_upstream", "doc_id", "prompt_id", "repetition",
    "ok", "error_kind", "truncated", "finish_reason", "attempts",
    # --- as cinco medidas do estudo ------------------------------------
    "wer", "cer", "num_f1", "html_sem_texto_extra",
    "prompt_tokens", "completion_tokens", "total_tokens", "cost_usd", "cost_source",
    # --- contexto para conferir os números acima -----------------------
    "latency_s", "saida_formato", "ref_n_tokens", "hyp_n_tokens",
    "num_n_ref", "num_n_hyp", "num_acertos", "timestamp_utc",
    # --- a pasta com as evidências daquela chamada ---------------------
    "pasta", "resposta",
]


def _ler_chamadas(run_dir: Path) -> list[dict]:
    if not (run_dir / CHAMADAS).exists():
        raise FileNotFoundError(f"{run_dir / CHAMADAS} não existe. Rode `run` antes de `score`.")
    return ler_chamadas(run_dir)


def _tabela_de_precos() -> dict[str, tuple[float | None, float | None]]:
    """Preço declarado por modelo, lido do models.yaml a cada pontuação."""
    return {m.id: (m.preco_entrada_usd_mtok, m.preco_saida_usd_mtok) for m in load_models()}


def _recalcular_custo_estimado(linha: dict, precos: dict) -> None:
    """Refaz o `cost_usd` das chamadas cujo custo é ESTIMADO, não medido.

    Pelo OpenRouter o valor vem cobrado e é dado primário — não se toca. Pela
    API nativa do Gemini não existe valor cobrado: o número é preço-de-tabela x
    tokens, e portanto derivado da configuração. Como derivado, tem que ser
    recalculado aqui, senão uma correção de preço no `models.yaml` só valeria
    para execuções futuras e o CSV misturaria duas tabelas de preço — foi o que
    aconteceu ao trocar os Gemini da faixa flex para a padrão.

    Os tokens continuam sendo os medidos na coleta; o que muda é só o preço
    aplicado a eles.
    """
    if linha.get("cost_source") != "tabela_precos":
        return
    modelo = precos.get(linha.get("model_id"))
    if modelo is None:
        return
    entrada, saida = modelo
    if entrada is None and saida is None:
        return
    linha["cost_usd"] = (
        (linha.get("prompt_tokens") or 0) * (entrada or 0.0)
        + (linha.get("completion_tokens") or 0) * (saida or 0.0)
    ) / 1e6


def score_run(run_dir: Path, settings: Settings | None = None) -> Path:
    settings = settings or load_settings()
    run_dir = Path(run_dir)
    documentos = {d.doc_id: d for d in discover_documents(settings)}
    chamadas = _ler_chamadas(run_dir)
    precos = _tabela_de_precos()

    # Só os documentos que esta execução usou, e cada um preparado UMA vez: é a
    # parte cara da conta e não depende do modelo.
    referencias = {
        doc_id: prepare(documentos[doc_id].reference_html)
        for doc_id in sorted({c["doc_id"] for c in chamadas} & set(documentos))
    }

    linhas: list[dict] = []
    for chamada in chamadas:
        if chamada["doc_id"] not in referencias:
            continue
        linha = {"run_id": run_dir.name, **{c: chamada.get(c) for c in COLUNAS if c in chamada}}
        _recalcular_custo_estimado(linha, precos)

        if chamada.get("ok") and chamada.get("resposta"):
            pasta = run_dir / chamada["pasta"]
            bruto = (pasta / chamada["resposta"]).read_text(encoding="utf-8", errors="replace")
            # As métricas rodam sobre a resposta como o modelo a devolveu: o
            # prompt pede o documento e nada mais, então texto em volta, se
            # houver, é resposta fora do pedido e conta como tal.
            hipotese = prepare(bruto)
            referencia = referencias[chamada["doc_id"]]
            linha.update(
                calcular(
                    referencia,
                    hipotese,
                    bruto,
                    # Sob o prompt sentinela ninguém pediu HTML: a coluna de
                    # obediência ao formato não se aplica e sai vazia.
                    pede_html=not chamada.get("prompt_agnostic"),
                )
            )
            # Os dois lados da comparação, na forma exata em que ela foi feita,
            # ao lado da resposta que os gerou. É o que permite conferir um WER
            # na mão: um diff entre estes dois arquivos mostra precisamente o
            # que a métrica contou — sem marcação, em minúsculas, sem
            # hifenização de fim de linha. A referência é repetida em cada
            # pasta de propósito: são ~8 KB, e é o que faz o diff ser um
            # comando só, sem navegar.
            (pasta / NORMALIZADO).write_text(hipotese.normalized, encoding="utf-8")
            (pasta / REFERENCIA).write_text(referencia.normalized, encoding="utf-8")
        linhas.append(linha)

    caminho = run_dir / METRICAS
    _escrever_csv(caminho, COLUNAS, linhas)

    # Os dois CSV respondem a perguntas diferentes, e a exclusão só vale para um
    # deles. O `metricas.csv` é a tabela de AUDITORIA: uma linha por chamada,
    # todas, inclusive as de quem saiu da análise — foram pagas e continuam
    # sendo evidência. O `resumo.csv` APRESENTA médias, e a média de um sistema
    # que a análise dispensou não é resultado: era o que acontecia, com o
    # `resumo.csv` publicando um CER que nenhuma figura mostrava.
    #
    # A regra mora em `config.motivos_de_exclusao`, a mesma que as figuras e a
    # tabela do relatório aplicam — três números que se apresentam como o mesmo
    # não podem responder a perguntas diferentes.
    excluidos = motivos_de_exclusao()
    do_resumo = [l for l in linhas if l.get("model_id") not in excluidos]
    for model_id, motivo in sorted(excluidos.items()):
        n = sum(1 for l in linhas if l.get("model_id") == model_id)
        if n:
            print(f"[score] {model_id} fora da análise ({n} chamada(s) ficam no "
                  f"{METRICAS} mas não entram no {RESUMO}): {motivo}")
    _escrever_csv(run_dir / RESUMO, *resumir(do_resumo))

    print(f"[score] {len(linhas)} chamada(s) avaliada(s) -> {caminho}")
    print(f"[score] resumo por modelo e prompt          -> {run_dir / RESUMO}")
    print(f"[score] normalizado.txt e referencia.txt gravados na pasta de cada chamada")
    return caminho


# --------------------------------------------------------------------------
# Resumo
# --------------------------------------------------------------------------

CHAVES_DO_RESUMO = ["empresa", "tipo", "model_id", "model_label", "prompt_id"]

# Médias calculadas só sobre as chamadas válidas.
MEDIAS = ["wer", "cer", "num_f1", "cost_usd", "completion_tokens", "prompt_tokens", "latency_s"]


def valida(linha: dict) -> bool:
    """Respondeu e nao foi cortada — a unica condicao em que a qualidade e medivel.

    Publica e usada tambem pelos graficos: a regra que separa CONFIABILIDADE de
    QUALIDADE precisa ser a mesma no resumo, nas figuras e na tabela do
    relatorio, senao tres numeros que se apresentam como o mesmo passam a
    responder a perguntas diferentes.
    """
    return bool(linha.get("ok")) and not linha.get("truncated")


def _media(valores: list) -> float | None:
    numeros = [v for v in valores if isinstance(v, (int, float))]
    return mean(numeros) if numeros else None


def resumir(
    linhas: list[dict], chaves: list[str] | None = None
) -> tuple[list[str], list[dict]]:
    """Agrupa as chamadas e resume cada grupo.

    `chaves` é parametrizável porque a mesma conta serve a duas perguntas: o
    `resumo.csv` agrupa por (modelo, prompt) dentro de uma execução, e os
    gráficos agrupam por modelo sobre TODAS as execuções. A regra que não muda
    é a separação entre confiabilidade e qualidade.
    """
    chaves = chaves or CHAVES_DO_RESUMO
    grupos: dict[tuple, list[dict]] = {}
    for linha in linhas:
        grupos.setdefault(tuple(linha.get(c) for c in chaves), []).append(linha)

    colunas = chaves + [
        "n_chamadas", "n_sucesso", "taxa_sucesso", "n_truncadas", "taxa_truncamento",
        # `n_validas` conta CHAMADAS; `n_paginas`, PAGINAS distintas. Os dois so
        # coincidem quando cada pagina foi medida uma vez. Com `repetitions > 1`,
        # com mais de um prompt, ou com uma pagina remedida noutra execucao, eles
        # divergem — e e `n_paginas` que responde "isto cobre o conjunto?", que e
        # a pergunta que decide quem disputa um superlativo.
        "n_validas", "n_paginas", "html_sem_texto_extra_taxa",
    ] + [f"{m}_media" for m in MEDIAS]

    resumo = []
    for chave, grupo in sorted(grupos.items(), key=lambda item: [str(v) for v in item[0]]):
        n = len(grupo)
        sucesso = [g for g in grupo if g.get("ok")]
        truncadas = [g for g in grupo if g.get("truncated")]
        # Qualidade só sobre o que dá para medir: respondeu e não foi cortada.
        # Sai de `grupo`, e não de `sucesso`: `valida` já testa `ok`, e filtrar
        # duas vezes sugeria que as duas condições eram independentes.
        validas = [g for g in grupo if valida(g)]
        linha = dict(zip(chaves, chave))
        linha.update(
            {
                "n_chamadas": n,
                "n_sucesso": len(sucesso),
                "taxa_sucesso": len(sucesso) / n,
                "n_truncadas": len(truncadas),
                "taxa_truncamento": len(truncadas) / n,
                "n_validas": len(validas),
                "n_paginas": len({g.get("doc_id") for g in validas if g.get("doc_id")}),
                # Só entram as chamadas em que a instrução foi de fato dada; as
                # do prompt sentinela têm a coluna vazia e ficam de fora, senão
                # o MinerU apareceria com 0% de obediência a uma ordem que
                # nunca recebeu.
                "html_sem_texto_extra_taxa": _media(
                    [
                        bool(g["html_sem_texto_extra"])
                        for g in validas
                        if g.get("html_sem_texto_extra") is not None
                    ]
                ),
            }
        )
        linha.update({f"{m}_media": _media([g.get(m) for g in validas]) for m in MEDIAS})
        resumo.append(linha)
    return colunas, resumo


def _escrever_csv(caminho: Path, colunas: list[str], linhas: list[dict]) -> None:
    # utf-8-sig para que o Excel abra os acentos corretamente.
    with open(caminho, "w", encoding="utf-8-sig", newline="") as fh:
        escritor = csv.DictWriter(fh, fieldnames=colunas, extrasaction="ignore")
        escritor.writeheader()
        escritor.writerows(linhas)
