"""Sete figuras sobre TODAS as chamadas já coletadas.

    cer_wer_por_modelo.png      erro por caractere e por palavra, ordenado pelo CER
    num_f1_por_modelo.png       fidelidade dos números
    custo_beneficio.png         CER x custo por página — onde cada sistema cai
    custo_n_paginas.png         a conta em dinheiro do acervo, em quatro papéis
    tempo_do_acervo.png         a conta em tempo do acervo, modelo a modelo
    html_sem_texto_extra.png    obediência ao "retorne apenas o código HTML"
    erro_por_dificuldade.png    quanto a complexidade da página custa em erro

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
* **`n=` no rótulo de TODOS, e em páginas.** Um modelo medido em 1 página não
  pode ser lido como um medido em 16, e a figura precisa dizer isso sem nota de
  rodapé. Marcar só quem tem pouco deixaria implícito que os demais são
  comparáveis entre si; o número é de páginas distintas, não de chamadas.
* **Superlativo só entre quem cobriu o conjunto.** "Melhor qualidade" apoiado
  numa página não é a mesma afirmação que apoiado em dezesseis, e é a figura de
  projeção — a que vira dinheiro — que mais sofre com a confusão.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sem janela: só arquivos
import matplotlib.pyplot as plt  # noqa: E402

from .config import Settings, load_models, load_settings, motivos_de_exclusao  # noqa: E402
from .dataset import ORDEM_DE_DIFICULDADE, discover_documents  # noqa: E402
from .formato import pt, usd  # noqa: E402
from .score import METRICAS, resumir, valida  # noqa: E402


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


# A identidade de uma chamada dentro do desenho experimental. Duas linhas com
# esta mesma chave sao a MESMA celula medida duas vezes, e nao duas evidencias.
_CHAVE_DA_CHAMADA = ("model_id", "doc_id", "prompt_id", "repetition")


def _deduplicar(linhas: list[dict]) -> tuple[list[dict], int]:
    """Uma linha por celula do desenho: modelo x pagina x prompt x repeticao.

    Uma execucao interrompida e retomada, ou simplesmente refeita, grava a mesma
    pagina de novo. Empilhar os `metricas.csv` sem isto faz essa pagina pesar
    duas vezes na media do modelo — e o peso passa a refletir quantas vezes ela
    foi remedida, que nao e propriedade do modelo nem do documento.

    Vence a medicao mais recente ENTRE AS VALIDAS. O `entre as validas` importa:
    remede-se justamente o que falhou, entao a ordem crua faria uma falha
    posterior apagar o sucesso que ela veio consertar. Quando nenhuma e valida,
    fica a mais recente, para a falha continuar visivel nas taxas.

    As linhas chegam na ordem das execucoes, que sao lidas em ordem de
    timestamp: a ultima de cada grupo e a mais nova.
    """
    grupos: dict[tuple, list[dict]] = {}
    for linha in linhas:
        grupos.setdefault(tuple(linha.get(c) for c in _CHAVE_DA_CHAMADA), []).append(linha)
    escolhidas = [
        next((l for l in reversed(grupo) if valida(l)), grupo[-1])
        for grupo in grupos.values()
    ]
    return escolhidas, len(linhas) - len(escolhidas)


def _reconciliar(linhas: list[dict]) -> list[dict]:
    """Confronta os `model_id` medidos com o registro, e aplica as exclusoes.

    Duas situacoes diferentes, tratadas de modo diferente de proposito:

    ORFAO — o `model_id` nao esta em `models.yaml`. Os CSV sao um historico e
    guardam sistemas renomeados ou retirados do registro; sem a conferencia um
    orfao entra nas figuras como se fosse mais um sistema do estudo e ninguem
    tem onde ler o que ele e. Aqui nada e removido, so AVISADO: ninguem decidiu
    nada, e provavelmente foi um acidente de renome que alguem precisa olhar.

    EXCLUIDO — o sistema esta no registro e declara um motivo para estar fora da
    analise. Aqui alguem decidiu, o motivo esta escrito, e as linhas saem das
    figuras, da tabela e do `resumo.csv`. As medicoes seguem no disco: sair da
    analise nao e deixar de ter existido.

    A regra da exclusao NAO mora aqui: mora em `config.motivos_de_exclusao`, e e
    a mesma que o `score` aplica ao montar o `resumo.csv`. Enquanto ela existia
    so neste arquivo, o resumo publicava a media de um sistema que nenhuma
    figura mostrava.
    """
    registro = {m.id: m for m in load_models()}
    orfaos = sorted({l["model_id"] for l in linhas if l.get("model_id") not in registro})
    if orfaos:
        print(f"[graficos] AVISO: {len(orfaos)} model_id fora de models.yaml, "
              f"presente(s) nas figuras sem registro que os descreva: {orfaos}")

    excluidos = motivos_de_exclusao(list(registro.values()))
    if not excluidos:
        return linhas
    mantidas = [l for l in linhas if l["model_id"] not in excluidos]
    for model_id, motivo in sorted(excluidos.items()):
        n = sum(1 for l in linhas if l["model_id"] == model_id)
        if n:
            print(f"[graficos] {model_id} fora da análise ({n} chamada(s) "
                  f"não entram nas figuras): {motivo}")
    return mantidas


def execucoes_avaliadas(settings: Settings, run_ids: list[str] | None = None) -> list[Path]:
    """As execuções que já têm `metricas.csv`, em ordem cronológica.

    O `run_id` é `%Y%m%d-%H%M%S`, então ordem alfabética É ordem de tempo — e é
    disso que o `_deduplicar` depende para saber qual medição é a mais recente.
    """
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
    return execucoes


def carregar_brutas(
    settings: Settings, run_ids: list[str] | None = None
) -> tuple[list[dict], list[Path]]:
    """Todas as linhas dos `metricas.csv`, SEM deduplicar e SEM excluir.

    Responde "quantas chamadas foram feitas e quanto custaram" — que é a
    pergunta do cabeçalho do relatório, e não a das figuras. As duas contagens
    são legitimamente diferentes: aqui conta-se dinheiro gasto, então uma
    remedição é mais uma chamada paga e um sistema fora da análise continua
    tendo custado. As figuras precisam do oposto — uma linha por célula do
    desenho, sem os excluídos —, e é o que `carregar_chamadas` entrega.
    """
    execucoes = execucoes_avaliadas(settings, run_ids)
    linhas: list[dict] = []
    for execucao in execucoes:
        with open(execucao / METRICAS, encoding="utf-8-sig") as fh:
            linhas.extend(_converter(l) for l in csv.DictReader(fh))
    return linhas, execucoes


def carregar_chamadas(settings: Settings, run_ids: list[str] | None = None) -> list[dict]:
    """Junta o `metricas.csv` de todas as execuções (ou das indicadas)."""
    linhas, execucoes = carregar_brutas(settings, run_ids)
    linhas, repetidas = _deduplicar(linhas)
    print(f"[graficos] {len(linhas)} chamada(s) de {len(execucoes)} execução(ões): "
          f"{', '.join(p.name for p in execucoes)}")
    if repetidas:
        print(f"[graficos] {repetidas} remedição(ões) descartada(s): a mesma página "
              f"medida mais de uma vez pelo mesmo sistema, mantida a mais recente válida")
    return _reconciliar(linhas)


def por_modelo(linhas: list[dict], total_de_paginas: int) -> list[dict]:
    """Uma linha por modelo, agregando as chamadas de todas as execuções.

    Marca tambem a COBERTURA, que e a mesma regra para as figuras e para a
    tabela do relatorio: um sistema so esta completo quando mediu todas as
    paginas do conjunto. Ter a regra num lugar so e o que impede o caso em que
    a tabela marca um sistema como `parcial` e a figura, gerada pelo mesmo
    comando e dos mesmos dados, lhe da um superlativo.

    A contagem e de PAGINAS distintas, nao de chamadas: comparar um numero de
    chamadas com um numero de documentos daria "completo" a quem mediu metade
    do conjunto duas vezes.
    """
    _, resumo = resumir(
        linhas, ["model_id", "model_label", "tipo", "empresa", "api_provider"]
    )
    for r in resumo:
        # O `n` aparece em TODOS os rótulos, não só nos poucos: marcar apenas
        # quem tem menos deixaria implícito que os demais são comparáveis entre
        # si, e a média de 16 páginas não é a mesma coisa que a de 1.
        #
        # E conta PAGINAS, nao chamadas: todo o texto do projeto le este numero
        # como pagina, e uma pagina remedida nao e uma pagina a mais.
        r["rotulo"] = f"{r['model_label']} (n={r['n_paginas']})"
        r["completo"] = r["n_paginas"] >= total_de_paginas
    return [r for r in resumo if r["n_validas"]]


def _completos(dados: list[dict]) -> list[dict]:
    """Os sistemas que podem disputar um superlativo, ou todos se nenhum pode.

    Devolver a lista inteira quando ninguem tem cobertura completa e deliberado:
    no comeco de um estudo ninguem tem, e uma figura vazia esconderia o que ja
    foi medido. Quem chama e obrigado a dizer na figura que a disputa foi entre
    parciais — ver o subtitulo de `_grafico_projecao`.
    """
    return [d for d in dados if d["completo"]] or dados


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


def _cor(linha: dict) -> str:
    """A cor de um sistema, a MESMA em todas as figuras.

    O local tem cor propria porque nao e uma terceira categoria de licenca — os
    MinerU sao `tipo: aberto` — e sim uma condicao que muda o que os eixos
    significam: custo por token nao se aplica e o tempo mede esta GPU. Antes
    esta regra existia so em duas das quatro figuras que pintam barras, e o
    mesmo MinerU aparecia cinza numa e azul na outra, como se tivesse mudado de
    categoria entre uma pagina e a seguinte.
    """
    return CINZA if _e_local(linha) else COR_POR_TIPO.get(linha["tipo"], CINZA)


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


def _mediana(ordenados: list[float]) -> float:
    """Mediana de uma lista JÁ ordenada."""
    meio = len(ordenados) // 2
    if len(ordenados) % 2:
        return ordenados[meio]
    return (ordenados[meio - 1] + ordenados[meio]) / 2


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

    medidos = [v for d in linhas for v in (d["cer_media"], d["wer_media"]) if v is not None]
    maior = max(medidos)
    limite = min(maior * 1.15, TETO_DE_ERRO)
    cortou = maior > limite

    for deslocamento, campo, cor, nome in (
        (0.20, "cer_media", VERDE, "CER (caractere)"),
        (-0.20, "wer_media", VIOLETA, "WER (palavra)"),
    ):
        # Sem `or 0.0`: nao medido nao e erro zero. Zero e falso em Python, e
        # trocar ausencia por zero desenharia uma barra vazia com o rotulo
        # "0,000" — que se le como transcricao perfeita, exatamente o oposto do
        # que o dado diz. E a mesma armadilha que `_ordenar_por` documenta.
        valores = [d[campo] for d in linhas]
        presentes = [(p, v) for p, v in zip(posicoes, valores) if v is not None]
        ax.barh([p + deslocamento for p, _ in presentes],
                [min(v, limite) for _, v in presentes],
                height=0.36, color=cor, label=nome)
        for p, valor in zip(posicoes, valores):
            if valor is None:
                _rotulo_na_ponta(ax, 0.0, p + deslocamento, "sem dado")
            elif valor > limite:
                ax.plot(limite, p + deslocamento, marker=">", markersize=7,
                        color=cor, clip_on=False)
                ax.annotate(pt(valor), xy=(limite, p + deslocamento), xytext=(12, 0),
                            textcoords="offset points", va="center",
                            color=TINTA_FRACA, fontsize=8.5, annotation_clip=False)
            else:
                _rotulo_na_ponta(ax, valor, p + deslocamento, pt(valor))

    ax.set_yticks(list(posicoes), [d["rotulo"] for d in linhas])
    _faixas(ax, len(linhas))
    ax.set_xlabel("taxa de erro (0 = perfeito)", color=TINTA_FRACA, fontsize=9)
    ax.set_xlim(0, limite)
    ax.legend(frameon=False, fontsize=9, labelcolor=TINTA_FRACA, loc="upper right")
    _titulo(ax, "Erro de transcrição por modelo",
            "ordenado pelo CER · menor é melhor"
            + (f" · acima de {pt(limite, 1)} a barra é cortada" if cortou else ""))
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
    ax.barh(range(len(linhas)), valores, height=0.35, color=[_cor(d) for d in linhas])
    for i, valor in enumerate(valores):
        _rotulo_na_ponta(ax, valor, i, pt(valor))
    ax.set_yticks(range(len(linhas)), [d["rotulo"] for d in linhas])
    _faixas(ax, len(linhas))
    ax.set_xlabel("F1 dos valores numéricos (1 = todos recuperados)",
                  color=TINTA_FRACA, fontsize=9)
    ax.set_xlim(0, 1.08)
    _legenda_por_tipo(ax, linhas, com_local=True, loc="lower right")
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

    # Os de cobertura parcial ficam na figura — o dado existe — mas vazados: um
    # ponto medido numa pagina nao pode ter o mesmo peso visual de um medido em
    # dezesseis, e o canto inferior esquerdo e justamente onde o olho procura o
    # vencedor.
    for completo in (True, False):
        grupo = [d for d in linhas if bool(d["completo"]) is completo]
        if not grupo:
            continue
        ax.scatter(
            [d["cost_usd_media"] for d in grupo], [d["cer_media"] for d in grupo],
            s=90, zorder=3, linewidths=2,
            color=[_cor(d) for d in grupo] if completo else "none",
            edgecolors=FUNDO if completo else [_cor(d) for d in grupo],
        )
    # Cada ponto é um sistema: sem o nome ao lado, a figura não se lê.
    for d in linhas:
        ax.annotate(d["rotulo"], xy=(d["cost_usd_media"], d["cer_media"]), xytext=(8, 3),
                    textcoords="offset points", color=TINTA_FRACA, fontsize=8)

    maior_custo = max(d["cost_usd_media"] for d in linhas) or 0.001
    ax.set_xlim(-maior_custo * 0.06 - 0.0005, maior_custo * 1.30)
    ax.set_xlabel(r"custo por página (US\$)", color=TINTA_FRACA, fontsize=9)
    ax.set_ylabel("CER", color=TINTA_FRACA, fontsize=9)
    _legenda_por_tipo(ax, linhas, com_local=True, loc="best",
                      rotulo_local="local (sem custo por token)")

    _titulo(ax, "Custo-benefício: erro x preço",
            "canto inferior esquerdo é o melhor custo benefício")
    return _salvar(fig, destino / "custo_beneficio.png")


# --------------------------------------------------------------------------
# 4. A conta de n mil páginas
# --------------------------------------------------------------------------

def escolher_cenarios(dados: list[dict]) -> list[tuple[str, dict]]:
    """Os modelos que respondem à pergunta de orçamento.

    São QUATRO papéis, e um mesmo modelo pode ocupar mais de um. Quando ocupa,
    isso é o achado: significa que ali não há trade-off a discutir.

    Os dois últimos existem porque "de graça" e "o mais barato que se paga" são
    dois pisos diferentes, e a comparação entre eles é o que decide se vale
    abrir a carteira.

    So disputam os sistemas com COBERTURA COMPLETA. E a figura que vira dinheiro:
    ela multiplica o custo medido por milhoes de paginas, e coroar um sistema
    medido numa pagina so projetaria uma unica chamada para o acervo inteiro
    como se fosse medida. E a mesma regra que a tabela do relatorio ja aplica
    para o negrito de melhor valor — aqui ela faltava.
    """
    elegiveis = _completos([d for d in dados
                            if d["cer_media"] is not None and d["cost_usd_media"] is not None])
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
            color=[_cor(m) for _, m in cenarios])
    for i, (valor, (papel, modelo)) in enumerate(zip(valores, cenarios)):
        # O preço sozinho não decide nada: o que ele compra vem junto.
        _rotulo_na_ponta(
            ax, valor, i,
            f"{usd(valor, 0)}   ·   CER {pt(modelo['cer_media'])}"
            f"   ·   {usd(modelo['cost_usd_media'], 4)}/página",
        )
    # O rótulo leva o `n`: "melhor qualidade" medido em 1 página não é a mesma
    # afirmação que "melhor qualidade" medido em 16.
    ax.set_yticks(range(len(cenarios)),
                  [f"{modelo['rotulo']}\n{papel}" for papel, modelo in cenarios])
    _faixas(ax, len(cenarios))
    ax.set_xlabel(rf"custo total de {pt(PAGINAS_DO_PROJETO, 0)} páginas (US\$)",
                  color=TINTA_FRACA, fontsize=9)
    # `or 1` porque um eixo de 0 a 0 nao existe: se todos os elegiveis forem
    # gratuitos o maior valor e zero, e a figura morreria dentro do try/except
    # do `build_graficos` — sumindo com uma mensagem em vez de mostrar que a
    # projecao para aquele conjunto custa nada.
    ax.set_xlim(0, max(valores) * 1.75 or 1)
    # O eixo usa o mesmo formato dos rótulos: com o horizonte em milhões de
    # páginas os valores passam de cinco dígitos, e "45963" ao lado de
    # "US$ 45.963" faria o leitor conferir duas vezes se é o mesmo número.
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: pt(v, 0))
    )
    _legenda_por_tipo(ax, [m for _, m in cenarios], com_local=True,
                      rotulo_local="local (sem custo por token)")
    subtitulo = "projeção linear do custo medido por página · o erro medido vai junto"
    if not all(m["completo"] for _, m in cenarios):
        # Nunca em silencio: um papel decidido entre coberturas parciais e uma
        # afirmacao mais fraca do que a figura aparenta fazer.
        subtitulo += " · nenhum sistema cobriu o conjunto: papéis decididos entre parciais"
    _titulo(ax, f"Quanto custaria transcrever {pt(PAGINAS_DO_PROJETO, 0)} páginas",
            subtitulo)
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
    ax.barh(range(len(linhas)), anos, height=0.35, color=[_cor(d) for d in linhas])
    for i, (d, valor) in enumerate(zip(linhas, anos)):
        dias = d["latency_s_media"] * PAGINAS_DO_PROJETO / CHAMADAS_SIMULTANEAS / SEGUNDOS_POR_DIA
        _rotulo_na_ponta(
            ax, valor, i,
            f"{pt(valor, 1)} anos em série   ·   {pt(d['latency_s_media'], 0)} s/página"
            f"   ·   {pt(dias, 0)} dias com {CHAMADAS_SIMULTANEAS} simultâneas",
        )
    ax.set_yticks(range(len(linhas)), [d["rotulo"] for d in linhas])
    _faixas(ax, len(linhas))
    ax.set_xlabel(f"anos para transcrever {pt(PAGINAS_DO_PROJETO, 0)} páginas, "
                  "uma de cada vez", color=TINTA_FRACA, fontsize=9)
    ax.set_xlim(0, max(anos) * 2.6)
    _legenda_por_tipo(ax, linhas, com_local=True,
                      rotulo_local="local (tempo desta máquina)")
    _titulo(
        ax,
        f"Quanto tempo levaria transcrever {pt(PAGINAS_DO_PROJETO, 0)} páginas",
        "latência medida por página x o acervo · a espera do rate limit não entra",
    )
    return _salvar(fig, destino / "tempo_do_acervo.png")


# --------------------------------------------------------------------------
# 6. Obediência ao formato
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
    ax.barh(range(len(linhas)), valores, height=0.35, color=[_cor(d) for d in linhas])
    for i, valor in enumerate(valores):
        _rotulo_na_ponta(ax, valor, i, f"{valor:.0f}%")
    ax.set_yticks(range(len(linhas)), [d["rotulo"] for d in linhas])
    _faixas(ax, len(linhas))
    ax.set_xlabel("% das respostas que vieram só com o HTML", color=TINTA_FRACA, fontsize=9)
    ax.set_xlim(0, 108)
    # Ordenado do menor para o maior, então o canto vazio é o de baixo.
    _legenda_por_tipo(ax, linhas, com_local=True, loc="lower right")
    _titulo(ax, "Obediência ao formato pedido",
            'o prompt termina com "retorne apenas o código HTML sem texto extra"')
    return _salvar(fig, destino / "html_sem_texto_extra.png")


# --------------------------------------------------------------------------
# 7. Erro por dificuldade da página
# --------------------------------------------------------------------------

def _grafico_dificuldade(chamadas: list[dict], destino: Path,
                         settings: Settings) -> Path | None:
    """Erro médio por grau de dificuldade da página, somando todos os sistemas.

    Esta figura agrega por CHAMADA, não por modelo: a pergunta é o que a página
    faz com o erro, e cada transcrição de cada sistema é uma observação dela.

    A comparação só é honesta se a cobertura for equilibrada: se os sistemas
    que não rodaram tudo faltarem mais num grau que noutro, a categoria mais
    difícil pode estar apenas concentrando os modelos piores, e o efeito medido
    seria dos modelos ausentes e não da página.

    Essa condição é MEDIDA a cada geração e impressa na saída do comando, e não
    afirmada aqui. Um número escrito na docstring envelhece na execução
    seguinte — este envelheceu: dizia "treze dos catorze sistemas rodaram as
    dezesseis páginas" quando já eram doze de quinze.

    Ela sai no console, e não no subtítulo, porque são coisas de leitores
    diferentes: o subtítulo diz o que é preciso para LER a figura e para no
    traço da mediana; a cobertura é condição de VALIDADE, que interessa a quem
    gera a figura e decide se ela pode ser publicada.

    A média vem acompanhada da MEDIANA, marcada por um traço fino sobre a
    barra. As duas juntas dizem o que uma sozinha esconde: quando a média é
    muito maior, o grau não é difícil para todo mundo, é difícil porque poucos
    sistemas desabam nele.

    O `TETO_DE_ERRO` NAO se aplica aqui, e a excecao e proposital. Nas figuras
    por modelo cada barra e um valor, e cortar o eixo esconde apenas o desenho
    de um outlier. Aqui cada barra ja e uma media: o outlier entrou nela e
    cortar o eixo esconderia a contaminacao em vez de mostra-la. Quem cumpre
    esse papel e a mediana ao lado — se as duas se afastam, o grau tem um caso
    extremo dentro, e a figura diz isso sem apagar nada.
    """
    graus = {d.doc_id: d.dificuldade for d in discover_documents(settings)}
    # `valida` — a MESMA função que o `resumir` aplica às demais figuras e ao
    # resumo, e não uma cópia da condição dela. Medir o CER de meia página não
    # mede a qualidade da transcrição, mede o corte, e as respostas cortadas
    # caíram só em médio e difícil, ou seja, inflariam justamente os dois graus
    # que carregam o efeito que a figura existe para mostrar. Com a regra
    # duplicada aqui, mudá-la num lugar deixava esta figura discordando em
    # silêncio de todas as outras.
    validas = [c for c in chamadas
               if valida(c) and graus.get(c["doc_id"]) and c.get("cer") is not None]
    if not validas:
        return None
    sem_grau = {c["doc_id"] for c in chamadas if not graus.get(c["doc_id"])}
    # As paginas efetivamente MEDIDAS, nao todas as do `Dataset/`: com
    # `--run-ids`, ou com `dataset.active` restringindo o conjunto, o rotulo
    # anunciaria paginas que nao entraram em observacao nenhuma.
    medidas = {c["doc_id"] for c in validas}
    paginas = {g: sum(1 for d, v in graus.items() if v == g and d in medidas)
               for g in ORDEM_DE_DIFICULDADE}

    def resumo(grau: str) -> dict:
        do_grau = [c for c in validas if graus[c["doc_id"]] == grau]
        valores = {m: sorted(c[m] for c in do_grau if c.get(m) is not None)
                   for m in ("cer", "wer")}
        return {
            "grau": grau,
            "n": len(do_grau),
            **{f"{m}_media": (sum(v) / len(v) if v else None) for m, v in valores.items()},
            **{f"{m}_mediana": (_mediana(v) if v else None) for m, v in valores.items()},
        }

    # De baixo para cima o eixo cresce, então a lista invertida põe "fácil" no
    # topo e "difícil" embaixo, na ordem em que a dificuldade se lê.
    linhas = [resumo(g) for g in reversed(ORDEM_DE_DIFICULDADE)]
    linhas = [l for l in linhas if l["n"]]
    fig, ax = _figura(len(linhas))
    posicoes = range(len(linhas))

    for deslocamento, campo, cor, nome in (
        (0.20, "cer", VERDE, "CER (caractere)"),
        (-0.20, "wer", VIOLETA, "WER (palavra)"),
    ):
        presentes = [(p, l) for p, l in zip(posicoes, linhas)
                     if l[f"{campo}_media"] is not None]
        ax.barh([p + deslocamento for p, _ in presentes],
                [l[f"{campo}_media"] for _, l in presentes],
                height=0.36, color=cor, label=nome)
        for p, linha in zip(posicoes, linhas):
            media = linha[f"{campo}_media"]
            if media is None:
                _rotulo_na_ponta(ax, 0.0, p + deslocamento, "sem dado")
                continue
            _rotulo_na_ponta(ax, media, p + deslocamento, pt(media))
            mediana = linha[f"{campo}_mediana"]
            if mediana is not None:
                # O traco e claro por cair DENTRO da barra. Quando a mediana
                # passa da media — distribuicao torta para o outro lado — ele
                # cairia no fundo e sumiria justamente no caso que a figura
                # existe para mostrar, entao ali ele veste tinta.
                ax.plot([mediana, mediana],
                        [p + deslocamento - 0.16, p + deslocamento + 0.16],
                        color=FUNDO if mediana <= media else TINTA_FRACA,
                        linewidth=1.8, solid_capstyle="butt")

    ax.set_yticks(list(posicoes),
                  [f"{l['grau']}\n{paginas[l['grau']]} páginas · n={l['n']}"
                   for l in linhas])
    _faixas(ax, len(linhas))
    ax.set_xlabel("taxa de erro média (0 = perfeito)", color=TINTA_FRACA, fontsize=9)
    # O limite olha as DUAS métricas: dimensionar pelo CER cortaria o rótulo do
    # WER, que é sempre o maior dos dois.
    maior = max(max(l["cer_media"] or 0, l["wer_media"] or 0) for l in linhas)
    ax.set_xlim(0, maior * 1.22)
    # Canto superior: as barras crescem com a dificuldade e a área livre fica
    # sempre do lado do grau mais fácil, que está no topo.
    ax.legend(frameon=False, fontsize=9, labelcolor=TINTA_FRACA, loc="upper right")

    # A cobertura, medida agora e dita na figura. `completos` conta sistemas que
    # mediram TODAS as paginas com grau; sem isso o leitor nao tem como saber se
    # a comparacao entre graus e uma comparacao entre paginas ou entre modelos.
    #
    # Contra as paginas MEDIDAS, e nao contra todo o `Dataset/` — a mesma
    # correcao que `paginas`, acima, ja tinha recebido. Com `--run-ids` ou com
    # `dataset.active` restringindo o conjunto, comparar com o dataset inteiro
    # anunciava "0 de N sistemas cobriram" mesmo quando todos cobriram tudo o
    # que rodou.
    paginas_com_grau = {d for d, g in graus.items() if g and d in medidas}
    por_sistema: dict[str, set] = defaultdict(set)
    for c in validas:
        por_sistema[c["model_id"]].add(c["doc_id"])
    completos = sum(1 for docs in por_sistema.values() if docs >= paginas_com_grau)

    # O subtitulo diz o que o leitor precisa para LER a figura, e para no traco
    # da mediana. A cobertura, a razao entre os graus e as paginas sem grau
    # continuam sendo MEDIDAS — sem elas a comparacao entre graus poderia ser
    # uma comparacao entre modelos sem ninguem notar —, mas vao para a saida do
    # comando: sao condicao de validade, que quem gera a figura precisa
    # conferir, e nao legenda, que quem le a figura precisa ver.
    por_grau = {l["grau"]: l for l in linhas}
    aviso = (f"[graficos] erro_por_dificuldade: {completos} de {len(por_sistema)} "
             f"sistema(s) cobriram as {len(paginas_com_grau)} página(s) com grau")
    facil, dificil = por_grau.get("fácil"), por_grau.get("difícil")
    if facil and dificil and facil["cer_media"]:
        aviso += (f" · a página difícil erra "
                  f"{pt(dificil['cer_media'] / facil['cer_media'], 1)} vezes mais que a fácil")
    if sem_grau:
        aviso += f" · {len(sem_grau)} página(s) sem grau no nome ficaram de fora"
    print(aviso)

    _titulo(ax, "Erro por dificuldade da página",
            "média de todas as chamadas · o traço claro é a mediana")
    return _salvar(fig, destino / "erro_por_dificuldade.png")


def build_graficos(settings: Settings | None = None,
                   run_ids: list[str] | None = None) -> list[Path]:
    """Gera as sete figuras em `results/graficos/`."""
    settings = settings or load_settings()
    chamadas = carregar_chamadas(settings, run_ids)
    dados = por_modelo(chamadas, len(discover_documents(settings)))
    destino = settings.results_dir / GRAFICOS_DIR
    destino.mkdir(parents=True, exist_ok=True)

    # Duas famílias, com unidades de agregação diferentes: as seis primeiras
    # comparam MODELOS, a última compara PÁGINAS. Por isso uma recebe o resumo
    # por modelo e a outra recebe as chamadas cruas.
    trabalhos = [(f, (dados, destino)) for f in (
        _grafico_cer_wer, _grafico_num_f1, _grafico_custo_beneficio,
        _grafico_projecao, _grafico_tempo, _grafico_obediencia)]
    trabalhos.append((_grafico_dificuldade, (chamadas, destino, settings)))

    figuras = []
    for construir, argumentos in trabalhos:
        # Uma figura sem dados (ou que falhe) não derruba as outras.
        try:
            caminho = construir(*argumentos)
        except Exception as exc:  # noqa: BLE001
            print(f"[graficos] {construir.__name__} falhou: {type(exc).__name__}: {exc}")
            continue
        if caminho:
            figuras.append(caminho)
            print(f"[graficos] {caminho.name}")
    if not figuras:
        print("[graficos] nada a desenhar: nenhuma chamada válida.")
    return figuras
