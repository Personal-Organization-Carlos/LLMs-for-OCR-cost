"""A tabela de resultados do relatório, gerada a partir das medições.

A escrita é feita entre dois marcadores dentro do `relatorio.html`. Tudo que
está fora deles é texto do autor e nunca é tocado, então a seção pode ser
reescrita quantas vezes for preciso sem risco para o resto do documento.
"""

from __future__ import annotations

import html
from pathlib import Path

from .config import Settings, load_settings
from .dataset import discover_documents
from .formato import pt

ABRE = "<!-- tabela-de-resultados: gerada por bench.relatorio, não editar à mão -->"
FECHA = "<!-- fim da tabela-de-resultados -->"

# Os números do cabeçalho e do rodapé saem da MESMA leitura que alimenta a
# tabela e as figuras. Digitados à mão eles envelheciam em silêncio: uma
# execução a mais, ou duas apagadas, e o relatório passava a anunciar um número
# de chamadas e um custo que nenhum arquivo sustentava — e nada avisava.
ABRE_META = "<!-- numeros-do-cabecalho: gerados por bench.relatorio, não editar à mão -->"
FECHA_META = "<!-- fim dos numeros-do-cabecalho -->"
ABRE_RODAPE = "<!-- numeros-do-rodape: gerados por bench.relatorio, não editar à mão -->"
FECHA_RODAPE = "<!-- fim dos numeros-do-rodape -->"

RELATORIO = "relatorio.html"

# Nome de cada via na linguagem do relatório, que fala de caminho de consulta e
# não de campo de configuração.
VIAS = {
    "openrouter": "OpenRouter",
    "gemini": "API do Google",
    "mineru": "execução local",
}


def _celula(valor, casas: int, melhor: bool) -> str:  # noqa: ANN001
    """Célula numérica, marcada em negrito quando é o melhor valor da coluna."""
    recuo = " " * 6
    if valor is None:
        return f'{recuo}<td class="num vazio">n/a</td>'
    classe = "num melhor" if melhor else "num"
    return f'{recuo}<td class="{classe}">{pt(valor, casas)}</td>'


def _melhor_valor(linhas: list[dict], campo: str, maior_e_melhor: bool):  # noqa: ANN001, ANN201
    """O melhor valor de uma coluna, ou None quando ninguém tem o dado.

    Só entram na disputa os sistemas com cobertura completa. Uma média sobre
    uma página não disputa com uma média sobre dezesseis, e premiar a primeira
    em negrito seria afirmar no visual o que o dado não sustenta.
    """
    valores = [l[campo] for l in linhas if l[campo] is not None and l["completo"]]
    if not valores:
        return None
    return max(valores) if maior_e_melhor else min(valores)


def montar_tabela(dados: list[dict], total_de_documentos: int) -> str:
    """O bloco HTML da tabela, ordenado pelo erro por caractere.

    A cobertura ja vem marcada por `por_modelo`, que e o unico lugar onde a
    regra mora: paginas distintas medidas contra o tamanho do conjunto. Antes
    ela era recalculada aqui comparando n_validas — uma contagem de CHAMADAS —
    com um numero de documentos, o que dava "completo" a quem tivesse medido
    metade do conjunto duas vezes.
    """
    linhas = sorted((d for d in dados if d["cer_media"] is not None),
                    key=lambda d: d["cer_media"])

    pior_cer = max(l["cer_media"] for l in linhas) if linhas else 1.0
    melhores = {
        "cer_media": _melhor_valor(linhas, "cer_media", False),
        "wer_media": _melhor_valor(linhas, "wer_media", False),
        "num_f1_media": _melhor_valor(linhas, "num_f1_media", True),
    }

    corpo = []
    for l in linhas:
        # A largura da barra é relativa ao pior erro da tabela, para a coluna
        # ser lida de relance sem que ninguém precise comparar decimais.
        largura = round(l["cer_media"] / pior_cer * 100) if pior_cer else 0
        destaque = " melhor" if l["cer_media"] == melhores["cer_media"] else ""
        parcial = "" if l["completo"] else (
            f'<span class="parcial" title="cobertura incompleta, '
            f'{l["n_paginas"]} de {total_de_documentos} páginas">parcial</span>')
        obediencia = l.get("html_sem_texto_extra_taxa")
        custo, tempo = l.get("cost_usd_media"), l.get("latency_s_media")
        corpo.append(f"""    <tr>
      <th scope="row">{html.escape(l["model_label"])}{parcial}<span class="sub">{html.escape(l["empresa"])}</span></th>
      <td><span class="selo {l["tipo"]}">{l["tipo"]}</span></td>
      <td class="via">{VIAS.get(l["api_provider"], l["api_provider"])}</td>
      <td class="num n">{l["n_paginas"]}<span class="de">/{total_de_documentos}</span></td>
      <td class="num barra-celula"><span class="barra" style="width:{largura}%"></span><span class="valor{destaque}">{pt(l["cer_media"], 3)}</span></td>
{_celula(l["wer_media"], 3, l["wer_media"] == melhores["wer_media"])}
{_celula(l["num_f1_media"], 3, l["num_f1_media"] == melhores["num_f1_media"])}
      <td class="num">{"n/a" if obediencia is None else f"{round(obediencia * 100)}%"}</td>
      <td class="num">{"grátis" if not custo else pt(custo, 4)}</td>
      <td class="num">{"n/a" if tempo is None else round(tempo)}</td>
    </tr>""")

    return f"""{ABRE}
<div class="rolagem">
<table class="resultados">
  <thead><tr>
    <th scope="col">Sistema</th><th scope="col">Tipo</th><th scope="col">Via</th>
    <th scope="col" class="num">Páginas</th><th scope="col" class="num">CER</th>
    <th scope="col" class="num">WER</th><th scope="col" class="num">num_f1</th>
    <th scope="col" class="num">HTML limpo</th>
    <th scope="col" class="num">US$ por página</th>
    <th scope="col" class="num">Segundos</th>
  </tr></thead>
  <tbody>
{chr(10).join(corpo)}
  </tbody>
</table>
</div>
{FECHA}"""


def _periodo(execucoes: list[Path]) -> str:
    """"07 a 09/09/2026", ou "08/09/2026" quando a coleta durou um dia só."""
    dias = sorted({p.name[:8] for p in execucoes})
    def completo(d: str) -> str:
        return f"{d[6:8]}/{d[4:6]}/{d[:4]}"
    if len(dias) == 1:
        return completo(dias[0])
    # Dentro do mesmo mês só o dia inicial precisa aparecer: "07 a 09/09/2026".
    if dias[0][:6] == dias[-1][:6]:
        return f"{dias[0][6:8]} a {completo(dias[-1])}"
    return f"{completo(dias[0])} a {completo(dias[-1])}"


def montar_meta(dados: list[dict], brutas: list[dict], execucoes: list[Path],
                total_de_documentos: int) -> str:
    """Os números do cabeçalho, contados sobre as chamadas CRUAS.

    `Chamadas` e `Custo total` contam tudo o que foi feito e pago, inclusive as
    remedições e os sistemas fora da análise — a pergunta ali é de orçamento.
    `Sistemas avaliados` conta as linhas da tabela, que é a outra pergunta: o
    que sustenta os resultados. Os dois números não batem entre si de
    propósito.
    """
    custo = sum(l["cost_usd"] for l in brutas
                if isinstance(l.get("cost_usd"), (int, float)))
    return f"""{ABRE_META}
    <span>Documentos <b>{total_de_documentos}</b></span>
    <span>Sistemas avaliados <b>{len(dados)}</b></span>
    <span>Chamadas <b>{pt(len(brutas), 0)}</b></span>
    <span>Custo total <b>US$ {pt(custo, 2)}</b></span>
    <span>Coleta em <b>{_periodo(execucoes)}</b></span>
{FECHA_META}"""


def montar_rodape(brutas: list[dict], execucoes: list[Path],
                  total_de_documentos: int) -> str:
    """A frase de procedência do rodapé, dos mesmos números do cabeçalho."""
    return (f"{ABRE_RODAPE}{pt(len(brutas), 0)} chamadas em {len(execucoes)} "
            f"execuções, sobre {total_de_documentos} páginas{FECHA_RODAPE}")


def _substituir(texto: str, abre: str, fecha: str, novo: str, nome: str) -> str | None:
    """Troca o miolo entre dois marcadores. Sem eles, avisa e não altera nada."""
    if abre not in texto or fecha not in texto:
        print(f"[relatorio] marcadores de {nome} ausentes em {RELATORIO}; esse "
              f"bloco ficou como está.")
        return None
    return texto[: texto.index(abre)] + novo + texto[texto.index(fecha) + len(fecha):]


def atualizar_relatorio(settings: Settings | None = None,
                        run_ids: list[str] | None = None) -> Path | None:
    """Reescreve os blocos gerados do `relatorio.html`, se ele existir.

    São três, cada um entre o seu par de marcadores: a tabela de resultados, os
    números do cabeçalho e a frase de procedência do rodapé. Todo o resto é
    texto do autor e nunca é tocado.
    """
    from .graficos import carregar_brutas, carregar_chamadas, por_modelo

    settings = settings or load_settings()
    caminho = settings.results_dir / RELATORIO
    if not caminho.exists():
        print(f"[relatorio] {RELATORIO} não existe, nada a atualizar.")
        return None

    texto = caminho.read_text(encoding="utf-8")
    if ABRE not in texto or FECHA not in texto:
        print(f"[relatorio] marcadores ausentes em {RELATORIO}. Insira o par "
              f"de comentários em volta da tabela para que ela seja gerada.")
        return None

    total = len(discover_documents(settings))
    brutas, execucoes = carregar_brutas(settings, run_ids)
    dados = por_modelo(carregar_chamadas(settings, run_ids), total)

    texto = _substituir(texto, ABRE, FECHA, montar_tabela(dados, total), "tabela")
    escritos = ["tabela"]
    for abre, fecha, novo, nome in (
        (ABRE_META, FECHA_META, montar_meta(dados, brutas, execucoes, total), "cabeçalho"),
        (ABRE_RODAPE, FECHA_RODAPE, montar_rodape(brutas, execucoes, total), "rodapé"),
    ):
        atualizado = _substituir(texto, abre, fecha, novo, nome)
        if atualizado is not None:
            texto = atualizado
            escritos.append(nome)

    caminho.write_text(texto, encoding="utf-8")
    print(f"[relatorio] {', '.join(escritos)} regravado(s) -> {caminho}")
    return caminho
