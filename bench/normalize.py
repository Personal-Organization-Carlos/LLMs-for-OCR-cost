"""Normalização: HTML/Markdown -> texto -> tokens comparáveis.

Este é o ponto mais sensível do benchmark. A referência é uma transcrição
estilizado em HTML e as hipóteses chegam em HTML (os LLMs) ou em Markdown (o
MinerU). Comparar as duas coisas cruas mediria coincidência de estilo, não
fidelidade de transcrição — a marcação entraria na contagem de palavras como se
fosse palavra da página.

A solução é reduzir os DOIS lados à mesma representação, com uma pipeline só:

    bruto  --to_plain_text-->  texto  --normalize_text-->  normalizado
                                                            |-> tokenize()      (WER)
                                                            |-> char_sequence() (CER)

As regras de normalização são fixas (não há chave em settings.yaml para elas):
NFC, minúsculas, tipografia canônica, pontuação removida só nas BORDAS do token
e desifenização de fim de linha. Para mudar qualquer uma, mexa aqui.

Implementado sobre `html.parser` da biblioteca padrão: sem dependência externa,
comportamento estável e auditável.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from html.parser import HTMLParser

# Tags cujo conteúdo textual nunca faz parte da transcrição.
#
# Deliberadamente NÃO inclui `head`, `meta` nem `link`: as duas últimas são
# tags void (`<meta charset="utf-8">` abre sem nunca fechar) e suprimir por
# profundidade a partir delas engoliria o documento inteiro.
_DROP_CONTENT_TAGS = {"script", "style", "title", "noscript"}

# Tags que forçam quebra de linha ao abrir/fechar.
_BLOCK_TAGS = {
    "address", "article", "aside", "blockquote", "body", "caption", "col",
    "colgroup", "dd", "div", "dl", "dt", "fieldset", "figcaption", "figure",
    "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li",
    "main", "nav", "ol", "p", "pre", "section", "table", "tbody", "tfoot",
    "thead", "tr", "ul",
}

# Células de tabela: separadas por espaço, não por quebra de linha, para que a
# linha da tabela continue sendo uma linha de texto.
#
# `span` entra aqui porque a transcrição de referência o usa para recuo de
# continuação: "Estado de<span>S. Paulo</span>" sem fronteira viraria o token
# "deS.", que nenhum modelo produziria — erro sistemático que não tem nada a
# ver com qualidade de transcrição.
_CELL_TAGS = {"td", "th", "span"}

# Uma cerca de código que envolve a resposta inteira: ```html ... ```
_CODE_FENCE_RE = re.compile(
    r"^\s*```+[ \t]*[A-Za-z0-9_+-]*[ \t]*\r?\n(?P<body>.*?)\r?\n?\s*```+\s*$",
    re.DOTALL,
)

_TYPOGRAPHY_TABLE = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "′": "'", "″": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-",
    " ": " ", " ": " ", " ": " ", " ": " ",
    "​": "", "‌": "", "‍": "", "﻿": "", "­": "",
})

_WS_RUN_RE = re.compile(r"[ \t\f\v]+")
_BLANK_LINES_RE = re.compile(r"\n\s*\n+")

# Hifenização de fim de linha: "San-\ntos" -> "santos". Aceita o hífen do
# jornal e "¬", a convenção do CATMuS. O jornal hifeniza; um modelo pode
# reunir a palavra ou preservar a quebra, e nenhuma das duas escolhas é erro de
# transcrição — por isso a junção é feita nos DOIS lados da comparação.
_HYPHEN_BREAK_RE = re.compile(r"(?<=\w)[-¬]\n(?=\w)")

# Pontuação removida das BORDAS do token. A interna é preservada de propósito:
# valores em réis como "1:770$300" e numerações como "1.º" perderiam o sentido.
_EDGE_PUNCTUATION = ".,;:!?()[]{}\"'«»…"


# --------------------------------------------------------------------------
# HTML -> texto
# --------------------------------------------------------------------------

class _TextExtractor(HTMLParser):
    """Extrai o texto visível de um fragmento HTML preservando quebras lógicas."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._suppress_depth = 0

    def _newline(self) -> None:
        if self.parts and self.parts[-1] != "\n":
            self.parts.append("\n")

    def _space(self) -> None:
        if self.parts and not self.parts[-1].endswith((" ", "\n")):
            self.parts.append(" ")

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag in _DROP_CONTENT_TAGS:
            self._suppress_depth += 1
        elif tag in _BLOCK_TAGS or tag == "br":
            self._newline()
        elif tag in _CELL_TAGS:
            self._space()

    def handle_startendtag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag in {"br", "hr"}:
            self._newline()
        elif tag == "img":
            self._space()

    def handle_endtag(self, tag: str) -> None:
        if tag in _DROP_CONTENT_TAGS:
            self._suppress_depth = max(0, self._suppress_depth - 1)
        elif tag in _BLOCK_TAGS:
            self._newline()
        elif tag in _CELL_TAGS:
            self._space()

    def handle_data(self, data: str) -> None:
        if not self._suppress_depth and data:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    """Converte HTML na sua forma textual visível."""
    extractor = _TextExtractor()
    try:
        extractor.feed(html)
        extractor.close()
    except Exception:  # noqa: BLE001 - html.parser é tolerante, mas não infalível
        return html
    return "".join(extractor.parts)


def strip_code_fences(texto: str) -> str:
    """Remove uma cerca de código que envolva a resposta inteira.

    Modelos frequentemente respondem com ```html ... ```; sem isto, as cercas
    virariam tokens espúrios e a detecção de formato falharia.
    """
    match = _CODE_FENCE_RE.match(texto or "")
    return match.group("body") if match else (texto or "")


# --------------------------------------------------------------------------
# Markdown -> texto
# --------------------------------------------------------------------------

_MD_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
_MD_TABLE_SEP = re.compile(r"^\s*\|?(?:\s*:?-{1,}:?\s*\|)+\s*:?-{0,}:?\s*\|?\s*$")
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MD_ULIST = re.compile(r"^\s{0,3}[-*+]\s+(.*)$")
_MD_OLIST = re.compile(r"^\s{0,3}\d+[.)]\s+(.*)$")
_MD_QUOTE = re.compile(r"^\s{0,3}>\s?")
_MD_FENCE = re.compile(r"^\s*```")
_MD_REGRA = re.compile(r"^\s{0,3}([-*_])(?:\s*\1){2,}\s*$")
# Ênfase: só some quando o marcador está numa borda de palavra. Sem isso,
# `text_image` viraria `textimage`.
_MD_ENFASE = re.compile(r"(?<![\w*_~`])[*_~`]{1,3}|[*_~`]{1,3}(?![\w*_~`])")

_HTML_TABELA = re.compile(r"<table\b.*?</table\s*>", re.IGNORECASE | re.DOTALL)
_HTML_BLOCO = re.compile(
    r"</?(?:html|body|p|div|table|tr|td|th|thead|tbody|h[1-6]|ul|ol|li|br|img|span|section|article)\b",
    re.IGNORECASE,
)
_DOC_START = re.compile(r"<!doctype\s+html|<html[\s>]", re.IGNORECASE)


def markdown_to_text(md: str) -> str:
    """Reduz Markdown ao texto que ele exibe, descartando a marcação.

    Sem isto, `#`, `|`, `---` e `**` entram na contagem de palavras como se
    fossem palavras da página, e a marcação colada à palavra (`**diario`)
    transforma acertos em substituições: o WER passaria a medir escolha de
    formato. As tabelas escritas em HTML dentro do Markdown — o formato do
    MinerU — são entregues ao extrator de HTML, não reimplementadas aqui.
    """
    if not md:
        return ""

    fragmentos: list[str] = []

    def _guardar(match: re.Match) -> str:
        fragmentos.append(match.group(0))
        return f"\n\x00{len(fragmentos) - 1}\x00\n"

    corpo = _HTML_TABELA.sub(_guardar, md)

    saida: list[str] = []
    em_cerca = False
    for linha in corpo.splitlines():
        if _MD_FENCE.match(linha):
            # A cerca some; o conteúdo dela fica — numa resposta de transcrição
            # o bloco cercado costuma ser o documento, não um exemplo.
            em_cerca = not em_cerca
            continue

        marca = re.fullmatch(r"\s*\x00(\d+)\x00\s*", linha)
        if marca:
            saida.append(html_to_text(fragmentos[int(marca.group(1))]))
            continue

        if not em_cerca:
            if _MD_REGRA.match(linha) or _MD_TABLE_SEP.match(linha):
                continue  # régua horizontal e separador de tabela não são texto
            titulo = _MD_HEADING.match(linha)
            if titulo:
                linha = titulo.group(2)
            else:
                item = _MD_ULIST.match(linha) or _MD_OLIST.match(linha)
                if item:
                    linha = item.group(1)
                linha = _MD_QUOTE.sub("", linha)
            if "|" in linha:
                # Célula de tabela: a barra vira espaço, para que a linha da
                # tabela continue sendo uma linha de texto legível.
                linha = linha.replace("\\|", "\x01").replace("|", " ").replace("\x01", "|")

        linha = _MD_IMAGE.sub(" ", linha)   # a imagem não é texto da página
        linha = _MD_LINK.sub(r"\1", linha)  # do link fica o rótulo
        saida.append(_MD_ENFASE.sub("", linha))

    return "\n".join(saida)


# --------------------------------------------------------------------------
# Classificação de formato
# --------------------------------------------------------------------------

def detectar_formato(bruto: str) -> str:
    """Classifica o que foi entregue: html | markdown | texto | vazio.

    A pergunta não é "existe alguma tag?" — um documento Markdown com uma
    tabela em HTML no meio (o formato do MinerU) tem tags e continua sendo
    Markdown. O critério é qual das duas marcações ESTRUTURA o documento.

    Governa duas coisas ao mesmo tempo: como o texto vira palavras (aqui) e com
    que extensão o arquivo de saída é gravado (`runner.py`). Ter um
    classificador só evita que os dois discordem.
    """
    texto = strip_code_fences(bruto or "")
    if not texto.strip():
        return "vazio"
    if _DOC_START.search(texto):
        return "html"

    # Marcas de Markdown que não acontecem por acaso: título ATX, imagem
    # `![]()` e separador de tabela GFM logo abaixo de uma linha com barras.
    linhas = texto.splitlines()
    marcas_md = sum(
        1
        for i, linha in enumerate(linhas)
        if _MD_HEADING.match(linha)
        or _MD_IMAGE.search(linha)
        or (_MD_TABLE_SEP.match(linha) and i and "|" in linhas[i - 1])
    )
    # As tabelas saem da contagem de HTML: no formato do MinerU elas são a
    # única marcação HTML de um documento que no resto é Markdown, e as suas
    # centenas de tags de célula afogariam as marcas de Markdown.
    blocos_html = len(_HTML_BLOCO.findall(_HTML_TABELA.sub("", texto)))

    if blocos_html > marcas_md:
        return "html"
    if marcas_md:
        return "markdown"
    return "texto"


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

def to_plain_text(bruto: str) -> str:
    """Ponto de entrada único: aceita HTML, Markdown ou texto corrido."""
    if not bruto:
        return ""
    formato = detectar_formato(bruto)
    texto = strip_code_fences(bruto)
    if formato == "html":
        texto = html_to_text(texto)
    elif formato == "markdown":
        texto = markdown_to_text(texto)
    return collapse_whitespace(texto)


def collapse_whitespace(texto: str) -> str:
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    texto = _BLANK_LINES_RE.sub("\n", _WS_RUN_RE.sub(" ", texto))
    return "\n".join(l for l in (linha.strip() for linha in texto.split("\n")) if l)


def normalize_text(texto: str) -> str:
    """Normalização de caracteres (sem tokenizar)."""
    texto = unicodedata.normalize("NFC", texto.translate(_TYPOGRAPHY_TABLE)).lower()
    return _HYPHEN_BREAK_RE.sub("", collapse_whitespace(texto))


def tokenize(texto: str) -> list[str]:
    """Divide em tokens de palavra, tirando a pontuação só das bordas."""
    return [tok for tok in (p.strip(_EDGE_PUNCTUATION) for p in texto.split()) if tok]


def char_sequence(texto: str) -> str:
    """Sequência de caracteres usada no CER: uma linha só, espaços simples."""
    return " ".join(texto.split())


@dataclass(frozen=True)
class NormalizedDoc:
    """As representações de um texto que as métricas consomem."""

    normalized: str
    tokens: tuple[str, ...]
    chars: str

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)


def prepare(bruto: str) -> NormalizedDoc:
    """Executa a pipeline completa sobre um texto bruto."""
    normalizado = normalize_text(to_plain_text(bruto))
    return NormalizedDoc(
        normalized=normalizado,
        tokens=tuple(tokenize(normalizado)),
        chars=char_sequence(normalizado),
    )
