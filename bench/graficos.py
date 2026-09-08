"""Seis figuras sobre TODAS as chamadas já coletadas.

    cer_wer_por_modelo.png      erro por caractere e por palavra, ordenado pelo CER
    num_f1_por_modelo.png       fidelidade dos números
    custo_beneficio.png         CER x custo por página — onde cada sistema cai
    custo_n_paginas.png         a conta em dinheiro do acervo, em três cenários
    tempo_do_acervo.png         a conta em tempo do acervo, modelo a modelo
    html_sem_texto_extra.png    obediência ao "retorne apenas o código HTML"

Os gráficos agregam **todas as execuções**, não a última: cada linha de
`metricas.csv` é uma chamada, e as chamadas de um mesmo modelo entram na mesma
média venham do dia que vierem. Agregar por chamada, e não por média de médias,
é o que mantém o peso correto quando uma execução tem 16 páginas e outra tem 1.

Decisões de desenho que não são gosto:

* **Barras horizontais, ordenadas pelo valor.** A comparação é de magnitude
  entre nomes compridos; ordenar é o que faz a figura ser lida sem procurar.
* **Uma escala por figura.** Nunca dois eixos y — CER e WER cabem juntos porque
  são a mesma unidade (taxa de erro); custo e qualidade viram dispersão, não um
  segundo eixo.
* **Duas cores só, e verificadas.** Azul `#2a78d6` e laranja `#eb6834` passam
  nos testes de daltonismo (ΔE 24,7 em protanopia) e de contraste contra o
  fundo. A cor nunca é a única pista: toda barra tem o valor escrito na ponta e
  a legenda está sempre presente quando há duas séries.
* **O texto nunca veste a cor da série** — rótulos e valores ficam em tinta
  neutra; quem carrega a identidade é a barra ao lado.
* **`n=` no rótulo de quem tem menos evidência.** Um modelo medido em 1 página
  não pode ser lido como um medido em 16, e a figura precisa dizer isso sem
  nota de rodapé.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sem janela: só arquivos
import matplotlib.pyplot as plt  # noqa: E402

from .config import Settings, load_settings  # noqa: E402
from .score import METRICAS, _resumir  # noqa: E402


AZUL, LARANJA = "#2a78d6", "#eb6834"      # aberto / fechado
VERDE, VIOLETA = "#008300", "#4a3aa7"     # CER / WER
CINZA = "#6f6d66"                         # execucao local, que nao e categoria
TINTA, TINTA_FRACA, FUNDO = "#0b0b0b", "#52514e", "#fcfcfb"

# Cor por tipo de modelo, o mesmo mapa em todas as figuras que o usam.
COR_POR_TIPO = {"aberto": AZUL, "fechado": LARANJA}

DPI = 150
ALTURA_POR_BARRA = 0.45  # polegadas
GRAFICOS_DIR = "graficos"

# Piso de faixas verticais. Sem ele, uma comparação com um modelo só esticaria
# a sua única barra por toda a altura do gráfico — a espessura da barra passaria
# a depender de quantos modelos rodaram, que é informação que ela não carrega.
MINIMO_DE_FAIXAS = 3

# Teto do eixo de erro. WER e CER passam de 1,0 quando o sistema alucina — e um
# único caso extremo (um modelo que despeja um bloco base64 no texto chega a 13)
# esmagaria todas as outras barras contra o zero. A barra é cortada no teto, com
# um marcador e o valor VERDADEIRO ao lado: nada é escondido.
TETO_DE_ERRO = 1.5

# O horizonte do projeto: a conta que interessa não é a da página, é a do
# acervo. O valor abaixo é o tamanho do acervo a projetar e muda conforme o
# recorte; por isso as figuras leem daqui em vez de trazer o número no nome.
PAGINAS_DO_PROJETO = 3_600_000


# --------------------------------------------------------------------------
# Dados: todas as chamadas de todas as execuções
# --------------------------------------------------------------------------

_NUMERICAS = {
    "wer", "cer", "num_f1", "cost_usd", "latency_s", "completion_tokens",
    "prompt_tokens", "total_tokens", "ref_n_tokens", "hyp_n_tokens",
}
_BOOLEANAS = {"ok", "truncated", "html_sem_texto_extra"}


def _converter(linha: dict) -> dict:
    """CSV devolve tudo como texto; as métricas precisam de número e booleano."""
    convertida = dict(linha)
    for campo in _NUMERICAS:
        valor = linha.get(campo)
        try:
            convertida[campo] = float(valor) if valor not in ("", None) else None
        except ValueError:
            convertida[campo] = None
    for campo in _BOOLEANAS:
        valor = linha.get(campo)
        convertida[campo] = {"True": True, "False": False}.get(valor)
    return convertida


def carregar_chamadas(settings: Settings, run_ids: list[str] | None = None) -> list[dict]:
    """Junta o `metricas.csv` de todas as execuções (ou das indicadas)."""
    raiz = settings.results_dir
    execucoes = sorted(p for p in raiz.iterdir() if p.is_dir() and (p / METRICAS).exists()) \
        if raiz.is_dir() else []
    if run_ids:
        procurados = set(run_ids)
        execucoes = [p for p in execucoes if p.name in procurados]
        faltando = procurados - {p.name for p in execucoes}
        if faltando:
            raise FileNotFoundError(f"Execução(ões) sem {METRICAS}: {sorted(faltando)}")
    if not execucoes:
        raise FileNotFoundError(
            f"Nenhuma execução avaliada em {raiz}. Rode `run` e depois `score`."
        )

    linhas: list[dict] = []
    for execucao in execucoes:
        with open(execucao / METRICAS, encoding="utf-8-sig") as fh:
            linhas.extend(_converter(l) for l in csv.DictReader(fh))
    print(f"[graficos] {len(linhas)} chamada(s) de {len(execucoes)} execução(ões): "
          f"{', '.join(p.name for p in execucoes)}")
    return linhas


def _por_modelo(linhas: list[dict]) -> list[dict]:
    """Uma linha por modelo, agregando as chamadas de todas as execuções."""
    _, resumo = _resumir(
        linhas, ["model_id", "model_label", "tipo", "empresa", "api_provider"]
    )
    for r in resumo:
        # O `n` aparece em TODOS os rótulos, não só nos poucos: marcar apenas
        # quem tem menos deixaria implícito que os demais são comparáveis entre
        # si, e a média de 16 páginas não é a mesma coisa que a de 1.
        r["rotulo"] = f"{r['model_label']} (n={r['n_validas']})"
    return [r for r in resumo if r["n_validas"]]


# --------------------------------------------------------------------------
# Andaimes das figuras
# --------------------------------------------------------------------------

def _figura(n_barras: int, largura: float = 9.0):  # noqa: ANN201
    altura = 1.6 + max(n_barras, MINIMO_DE_FAIXAS) * ALTURA_POR_BARRA
    fig, ax = plt.subplots(figsize=(largura, altura), dpi=DPI)
    fig.patch.set_facecolor(FUNDO)
    ax.set_facecolor(FUNDO)
    ax.grid(axis="x", color="#e3e2df", linewidth=0.8)  # grade recessiva
    ax.set_axisbelow(True)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color("#d5d4d0")
    ax.tick_params(colors=TINTA_FRACA, labelsize=9, length=0)
    return fig, ax


def _faixas(ax, n_barras: int) -> None:  # noqa: ANN001
    """Fixa a altura de cada faixa, para a barra sair sempre com a mesma espessura.

    Quando há menos barras que o piso, a sobra é dividida entre topo e base,
    para que poucas barras fiquem centradas em vez de encostadas embaixo.
    """
    sobra = max(0, MINIMO_DE_FAIXAS - n_barras) / 2
    ax.set_ylim(-0.6 - sobra, n_barras - 0.4 + sobra)


def _dispersao(largura: float = 9.0, altura: float = 6.0):  # noqa: ANN201
    fig, ax = plt.subplots(figsize=(largura, altura), dpi=DPI)
    fig.patch.set_facecolor(FUNDO)
    ax.set_facecolor(FUNDO)
    ax.grid(color="#e3e2df", linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("#d5d4d0")
    ax.tick_params(colors=TINTA_FRACA, labelsize=9, length=0)
    return fig, ax


def _e_local(linha: dict) -> bool:
    """O sistema roda nesta máquina, e não numa API."""
    return linha.get("api_provider") == "mineru"


def _legenda_por_tipo(ax, linhas: list[dict], com_local: bool = False,  # noqa: ANN001
                     loc: str = "upper right", rotulo_local: str = "local") -> None:
    """Legenda das cores presentes, sem inventar entrada que não aparece.

    `loc` existe porque o canto vazio depende da ordenação: quando as barras
    menores ficam em cima, sobra o canto superior; quando ficam embaixo, sobra
    o inferior.
    """
    from matplotlib.patches import Patch

    itens = []
    for tipo, cor in COR_POR_TIPO.items():
        if any(d["tipo"] == tipo and not (com_local and _e_local(d)) for d in linhas):
            itens.append(Patch(facecolor=cor, label=tipo))
    if com_local and any(_e_local(d) for d in linhas):
        itens.append(Patch(facecolor=CINZA, label=rotulo_local))
    if itens:
        ax.legend(handles=itens, frameon=False, fontsize=9,
                  labelcolor=TINTA_FRACA, loc=loc)


def _titulo(ax, texto: str, subtitulo: str) -> None:  # noqa: ANN001
    ax.set_title(texto, color=TINTA, fontsize=13, fontweight="bold", loc="left", pad=18)
    ax.annotate(subtitulo, xy=(0, 1), xytext=(0, 8), xycoords="axes fraction",
                textcoords="offset points", color=TINTA_FRACA, fontsize=9)


def _rotulo_na_ponta(ax, x: float, y: float, texto: str) -> None:  # noqa: ANN001
    ax.annotate(texto, xy=(x, y), xytext=(5, 0), textcoords="offset points",
                va="center", color=TINTA_FRACA, fontsize=8.5)


def _salvar(fig, caminho: Path) -> Path:  # noqa: ANN001
    """Grava o PNG.

    Se a escrita falhar com `OSError`, o arquivo está aberto num visualizador —
    feche-o e rode de novo. O erro aparece na saída do comando, nomeando a
    figura que não foi regravada.
    """
    fig.tight_layout()
    fig.savefig(caminho, facecolor=FUNDO, bbox_inches="tight")
    plt.close(fig)
    return caminho


def _pt(valor: float, casas: int = 3) -> str:
    """Número em português: vírgula decimal, ponto de milhar."""
    return f"{valor:,.{casas}f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def _usd(valor: float, casas: int = 2) -> str:
    r"""Valor em dólar, com o cifrão escapado.

    Entre dois `$` o matplotlib entra em modo fórmula: "US$ 383 · US$ 0,0038"
    saía em itálico matemático, como se fosse uma equação. A barra invertida
    desliga a interpretação.
    """
    return rf"US\$ {_pt(valor, casas)}"


# --------------------------------------------------------------------------
# 1. CER e WER, ordenados pelo CER
# --------------------------------------------------------------------------

def _grafico_cer_wer(dados: list[dict], destino: Path) -> Path | None:
    linhas = [d for d in dados if d["cer_media"] is not None]
    if not linhas:
        return None
    # Ordenado pelo CER: é a medida mais estável das duas (um erro de uma letra
    # custa uma letra, não a palavra inteira), e por isso a que ordena melhor.
    linhas.sort(key=lambda d: d["cer_media"], reverse=True)
    fig, ax = _figura(len(linhas))
    posicoes = range(len(linhas))

    maior = max(max(d["cer_media"], d["wer_media"] or 0) for d in linhas)
    limite = min(maior * 1.15, TETO_DE_ERRO)
    cortou = maior > limite

    for deslocamento, campo, cor, nome in (
        (0.20, "cer_media", VERDE, "CER (caractere)"),
        (-0.20, "wer_media", VIOLETA, "WER (palavra)"),
    ):
        valores = [d[campo] or 0.0 for d in linhas]
        ax.barh([p + deslocamento for p in posicoes], [min(v, limite) for v in valores],
                height=0.36, color=cor, label=nome)
        for p, valor in zip(posicoes, valores):
            if valor > limite:
                ax.plot(limite, p + deslocamento, marker=">", markersize=7,
                        color=cor, clip_on=False)
                ax.annotate(_pt(valor), xy=(limite, p + deslocamento), xytext=(12, 0),
                            textcoords="offset points", va="center",
                            color=TINTA_FRACA, fontsize=8.5, annotation_clip=False)
            else:
                _rotulo_na_ponta(ax, valor, p + deslocamento, _pt(valor))

    ax.set_yticks(list(posicoes), [d["rotulo"] for d in linhas])
    _faixas(ax, len(linhas))
    ax.set_xlabel("taxa de erro (0 = perfeito)", color=TINTA_FRACA, fontsize=9)
    ax.set_xlim(0, limite)
    ax.legend(frameon=False, fontsize=9, labelcolor=TINTA_FRACA, loc="upper right")
    _titulo(ax, "Erro de transcrição por modelo",
            "ordenado pelo CER · menor é melhor"
            + (f" · acima de {_pt(limite, 1)} a barra é cortada" if cortou else ""))
    return _salvar(fig, destino / "cer_wer_por_modelo.png")


# --------------------------------------------------------------------------
# 2. Números
# --------------------------------------------------------------------------

def _grafico_num_f1(dados: list[dict], destino: Path) -> Path | None:
    linhas = [d for d in dados if d["num_f1_media"] is not None]
    if not linhas:
        return None
    linhas.sort(key=lambda d: d["num_f1_media"])
    fig, ax = _figura(len(linhas))
    valores = [d["num_f1_media"] for d in linhas]
    ax.barh(range(len(linhas)), valores, height=0.35,
            color=[COR_POR_TIPO.get(d["tipo"], CINZA) for d in linhas])
    for i, valor in enumerate(valores):
        _rotulo_na_ponta(ax, valor, i, _pt(valor))
    ax.set_yticks(range(len(linhas)), [d["rotulo"] for d in linhas])
    _faixas(ax, len(linhas))
    ax.set_xlabel("F1 dos valores numéricos (1 = todos recuperados)",
                  color=TINTA_FRACA, fontsize=9)
    ax.set_xlim(0, 1.08)
    _legenda_por_tipo(ax, linhas, loc="lower right")
    _titulo(ax, "Fidelidade dos números",
            "réis, datas, números de processo · maior é melhor")
    return _salvar(fig, destino / "num_f1_por_modelo.png")


# --------------------------------------------------------------------------
# 3. Custo-benefício: CER x custo
# --------------------------------------------------------------------------

def custo_por_pagina_correta(linha: dict) -> float | None:
    """Custo dividido pela fração de caracteres que saiu certa.

    É o critério de custo-benefício usado aqui: um modelo que custa metade mas
    erra o dobro não é mais barato, é mais caro por página aproveitável.
    Formalmente, `custo / (1 - CER)`. Um CER acima de 1, ou seja, alucinação
    maior que a própria página, não tem página aproveitável nenhuma e fica de
    fora.

    Só se aplica a quem cobra. Para um sistema local o resultado seria sempre
    zero, o que o tornaria vencedor automático de um critério que existe para
    comparar preços entre si — e "de graça" já tem papel próprio.
    """
    custo, cer = linha.get("cost_usd_media"), linha.get("cer_media")
    if not custo or cer is None or cer >= 1:
        return None
    return custo / (1 - cer)


def _ordenar_por(chave):  # noqa: ANN001, ANN201
    """Chave de ordenação que joga o indefinido para o fim, sem confundir com zero.

    `valor or infinito` seria tentador e está errado, porque zero é falso em
    Python: um custo de zero viraria infinito e o sistema gratuito sairia da
    comparação sem que ninguém tivesse decidido isso.
    """
    def comparar(linha):
        valor = chave(linha)
        return float("inf") if valor is None else valor

    return comparar


def _grafico_custo_beneficio(dados: list[dict], destino: Path) -> Path | None:
    linhas = [d for d in dados
              if d["cer_media"] is not None and d["cost_usd_media"] is not None]
    if not linhas:
        return None
    fig, ax = _dispersao()

    for tipo, cor in (("aberto", AZUL), ("fechado", LARANJA)):
        grupo = [d for d in linhas if d["tipo"] == tipo]
        if grupo:
            ax.scatter([d["cost_usd_media"] for d in grupo], [d["cer_media"] for d in grupo],
                       s=90, color=cor, edgecolors=FUNDO, linewidths=2, label=tipo, zorder=3)
    # Cada ponto é um sistema: sem o nome ao lado, a figura não se lê.
    for d in linhas:
        ax.annotate(d["rotulo"], xy=(d["cost_usd_media"], d["cer_media"]), xytext=(8, 3),
                    textcoords="offset points", color=TINTA_FRACA, fontsize=8)

    maior_custo = max(d["cost_usd_media"] for d in linhas) or 0.001
    ax.set_xlim(-maior_custo * 0.06 - 0.0005, maior_custo * 1.30)
    ax.set_xlabel(r"custo por página (US\$)", color=TINTA_FRACA, fontsize=9)
    ax.set_ylabel("CER", color=TINTA_FRACA, fontsize=9)
    ax.legend(frameon=False, loc="best", fontsize=9, labelcolor=TINTA_FRACA)

    pagos = [d for d in linhas if d["cost_usd_media"]]
    subtitulo = "canto inferior esquerdo é o melhor negócio"
    if pagos:
        melhor = min(pagos, key=_ordenar_por(custo_por_pagina_correta))
        subtitulo += f" · melhor custo por página correta entre os pagos, {melhor['model_label']}"
    _titulo(ax, "Custo-benefício: erro x preço", subtitulo)
    return _salvar(fig, destino / "custo_beneficio.png")


# --------------------------------------------------------------------------
# 4. A conta de n mil páginas
# --------------------------------------------------------------------------

def escolher_cenarios(dados: list[dict]) -> list[tuple[str, dict]]:
    """Os modelos que respondem à pergunta de orçamento.

    São cinco papéis, e um mesmo modelo pode ocupar mais de um. Quando ocupa,
    isso é o achado: significa que ali não há trade-off a discutir.

    Os dois últimos existem porque "de graça" e "o mais barato que se paga" são
    dois pisos diferentes, e a comparação entre eles é o que decide se vale
    abrir a carteira.
    """
    elegiveis = [d for d in dados
                 if d["cer_media"] is not None and d["cost_usd_media"] is not None]
    if not elegiveis:
        return []
    gratuitos = [d for d in elegiveis if not d["cost_usd_media"]]
    pagos = [d for d in elegiveis if d["cost_usd_media"]]

    # "Mais barato" e "melhor custo-benefício" olham só para quem cobra. Um
    # sistema local venceria os dois com zero, o que responderia à pergunta
    # errada: quem já decidiu não pagar não está escolhendo entre preços.
    papeis = [("melhor qualidade", min(elegiveis, key=lambda d: d["cer_media"]))]
    if pagos:
        papeis.append(("melhor custo-benefício",
                       min(pagos, key=_ordenar_por(custo_por_pagina_correta))))
        papeis.append(("mais barato", min(pagos, key=lambda d: d["cost_usd_media"])))
    if gratuitos:
        papeis.append(("melhor gratuito", min(gratuitos, key=lambda d: d["cer_media"])))
    juntos: dict[str, tuple[list[str], dict]] = {}
    for papel, modelo in papeis:
        rotulos, _ = juntos.setdefault(modelo["model_id"], ([], modelo))
        rotulos.append(papel)
    return [(" + ".join(rotulos), modelo) for rotulos, modelo in juntos.values()]


def _grafico_projecao(dados: list[dict], destino: Path) -> Path | None:
    cenarios = escolher_cenarios(dados)
    if not cenarios:
        return None
    cenarios.sort(key=lambda c: c[1]["cost_usd_media"] * PAGINAS_DO_PROJETO, reverse=True)

    fig, ax = _figura(len(cenarios), largura=9.5)
    valores = [c[1]["cost_usd_media"] * PAGINAS_DO_PROJETO for c in cenarios]
    ax.barh(range(len(cenarios)), valores, height=0.35,
            color=[CINZA if _e_local(m) else COR_POR_TIPO.get(m["tipo"], CINZA)
                   for _, m in cenarios])
    for i, (valor, (papel, modelo)) in enumerate(zip(valores, cenarios)):
        # O preço sozinho não decide nada: o que ele compra vem junto.
        _rotulo_na_ponta(
            ax, valor, i,
            f"{_usd(valor, 0)}   ·   CER {_pt(modelo['cer_media'])}"
            f"   ·   {_usd(modelo['cost_usd_media'], 4)}/página",
        )
    # O rótulo leva o `n`: "melhor qualidade" medido em 1 página não é a mesma
    # afirmação que "melhor qualidade" medido em 16.
    ax.set_yticks(range(len(cenarios)),
                  [f"{modelo['rotulo']}\n{papel}" for papel, modelo in cenarios])
    _faixas(ax, len(cenarios))
    ax.set_xlabel(f"custo total de {_pt(PAGINAS_DO_PROJETO, 0)} páginas (US$)",
                  color=TINTA_FRACA, fontsize=9)
    ax.set_xlim(0, max(valores) * 1.75)
    # O eixo usa o mesmo formato dos rótulos: com o horizonte em milhões de
    # páginas os valores passam de cinco dígitos, e "45963" ao lado de
    # "US$ 45.963" faria o leitor conferir duas vezes se é o mesmo número.
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: _pt(v, 0))
    )
    _legenda_por_tipo(ax, [m for _, m in cenarios], com_local=True,
                      rotulo_local="local (sem custo por token)")
    _titulo(ax, f"Quanto custaria transcrever {_pt(PAGINAS_DO_PROJETO, 0)} páginas",
            "projeção linear do custo medido por página · o erro medido vai junto")
    return _salvar(fig, destino / "custo_n_paginas.png")


# --------------------------------------------------------------------------
# 5. A conta do TEMPO
# --------------------------------------------------------------------------

# Quantas chamadas se manteria no ar ao mesmo tempo num mutirão de transcrição.
# Serve só para traduzir o tempo serial em prazo de calendário; é premissa
# declarada, não medida — e é por isso que o número serial vai no eixo e o
# paralelo fica no rótulo.
CHAMADAS_SIMULTANEAS = 50

SEGUNDOS_POR_DIA = 86_400
SEGUNDOS_POR_ANO = 365 * SEGUNDOS_POR_DIA


def _grafico_tempo(dados: list[dict], destino: Path) -> Path | None:
    linhas = [d for d in dados if d["latency_s_media"] is not None]
    if not linhas:
        return None
    linhas.sort(key=lambda d: d["latency_s_media"], reverse=True)

    fig, ax = _figura(len(linhas), largura=9.5)
    # Serial: uma página de cada vez, que é exatamente como a latência foi
    # medida. Multiplicar pelo acervo sem supor concorrência nenhuma mantém o
    # eixo colado no que foi observado.
    anos = [d["latency_s_media"] * PAGINAS_DO_PROJETO / SEGUNDOS_POR_ANO for d in linhas]
    # Os locais saem em cinza porque o tempo deles não é comparável ao dos
    # demais: mede esta GPU, não o modelo. Trocar de máquina muda a barra.
    ax.barh(range(len(linhas)), anos, height=0.35,
            color=[CINZA if _e_local(d) else COR_POR_TIPO.get(d["tipo"], CINZA)
                   for d in linhas])
    for i, (d, valor) in enumerate(zip(linhas, anos)):
        dias = d["latency_s_media"] * PAGINAS_DO_PROJETO / CHAMADAS_SIMULTANEAS / SEGUNDOS_POR_DIA
        _rotulo_na_ponta(
            ax, valor, i,
            f"{_pt(valor, 1)} anos em série   ·   {_pt(d['latency_s_media'], 0)} s/página"
            f"   ·   {_pt(dias, 0)} dias com {CHAMADAS_SIMULTANEAS} simultâneas",
        )
    ax.set_yticks(range(len(linhas)), [d["rotulo"] for d in linhas])
    _faixas(ax, len(linhas))
    ax.set_xlabel(f"anos para transcrever {_pt(PAGINAS_DO_PROJETO, 0)} páginas, "
                  "uma de cada vez", color=TINTA_FRACA, fontsize=9)
    ax.set_xlim(0, max(anos) * 2.6)
    _legenda_por_tipo(ax, linhas, com_local=True,
                      rotulo_local="local (tempo desta máquina)")
    _titulo(
        ax,
        f"Quanto tempo levaria transcrever {_pt(PAGINAS_DO_PROJETO, 0)} páginas",
        "latência medida por página x o acervo · a espera do rate limit não entra",
    )
    return _salvar(fig, destino / "tempo_do_acervo.png")


# --------------------------------------------------------------------------
# 5. Obediência ao formato
# --------------------------------------------------------------------------

def _grafico_obediencia(dados: list[dict], destino: Path) -> Path | None:
    # Só os sistemas que receberam a instrução: sob o prompt sentinela (MinerU)
    # a coluna é vazia, e cobrar obediência a uma ordem não dada seria erro.
    linhas = [d for d in dados if d["html_sem_texto_extra_taxa"] is not None]
    if not linhas:
        return None
    linhas.sort(key=lambda d: d["html_sem_texto_extra_taxa"])
    fig, ax = _figura(len(linhas))
    valores = [d["html_sem_texto_extra_taxa"] * 100 for d in linhas]
    ax.barh(range(len(linhas)), valores, height=0.35,
            color=[COR_POR_TIPO.get(d["tipo"], CINZA) for d in linhas])
    for i, valor in enumerate(valores):
        _rotulo_na_ponta(ax, valor, i, f"{valor:.0f}%")
    ax.set_yticks(range(len(linhas)), [d["rotulo"] for d in linhas])
    _faixas(ax, len(linhas))
    ax.set_xlabel("% das respostas que vieram só com o HTML", color=TINTA_FRACA, fontsize=9)
    ax.set_xlim(0, 108)
    # Ordenado do menor para o maior, então o canto vazio é o de baixo.
    _legenda_por_tipo(ax, linhas, loc="lower right")
    _titulo(ax, "Obediência ao formato pedido",
            'o prompt termina com "retorne apenas o código HTML sem texto extra"')
    return _salvar(fig, destino / "html_sem_texto_extra.png")


# --------------------------------------------------------------------------

def build_graficos(settings: Settings | None = None,
                   run_ids: list[str] | None = None) -> list[Path]:
    """Gera as seis figuras em `results/graficos/`."""
    settings = settings or load_settings()
    dados = _por_modelo(carregar_chamadas(settings, run_ids))
    destino = settings.results_dir / GRAFICOS_DIR
    destino.mkdir(parents=True, exist_ok=True)

    figuras = []
    for construir in (_grafico_cer_wer, _grafico_num_f1, _grafico_custo_beneficio,
                      _grafico_projecao, _grafico_tempo, _grafico_obediencia):
        # Uma figura sem dados (ou que falhe) não derruba as outras.
        try:
            caminho = construir(dados, destino)
        except Exception as exc:  # noqa: BLE001
            print(f"[graficos] {construir.__name__} falhou: {type(exc).__name__}: {exc}")
            continue
        if caminho:
            figuras.append(caminho)
            print(f"[graficos] {caminho.name}")
    if not figuras:
        print("[graficos] nada a desenhar: nenhuma chamada válida.")
    return figuras
