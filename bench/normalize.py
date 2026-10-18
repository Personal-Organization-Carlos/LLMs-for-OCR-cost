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
NFC, minúsculas, tipografia canônica, pontuação removida só nas BORDAS do token,
fios e traços soltos fora da contagem de palavras e desifenização. Para mudar
qualquer uma, mexa aqui.

A desifenização é a única regra que olha para a referência — ver
`_resolver_hifens`. As demais tratam os dois lados sem saber qual é qual.

Implementado sobre `html.parser` da biblioteca padrão: sem dependência externa,
comportamento estável e auditável.
"""

from __future__ import annotations

import re
import unicodedata
import warnings
from dataclasses import dataclass
from html import unescape as _decodificar_entidades
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

# Linha que é só uma cerca, em qualquer ponto da resposta. Pega a cerca que
# sobra quando a resposta foi cortada pelo limite de tokens (abre e nunca
# fecha) ou quando vem depois de uma prosa de abertura.
_FENCE_LINE_RE = re.compile(r"^[ \t]*```+[ \t]*[A-Za-z0-9_+-]*[ \t]*$", re.MULTILINE)

_TYPOGRAPHY_TABLE = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "′": "'", "″": '"',
    # O ordinal tem quatro grafias para o mesmo sinal impresso: "1.º", "1.°"
    # (grau e ordinal são o mesmo glifo no tipo da época), "1.<sup>o</sup>"
    # e o `$^{o}$` do MinerU. As duas últimas já chegam aqui como "o" e "a";
    # as outras vão para a mesma letra, senão a escolha de grafia contava como
    # palavra errada.
    "º": "o", "°": "o", "ª": "a",
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-",
    " ": " ", " ": " ", " ": " ", " ": " ",
    "​": "", "‌": "", "‍": "", "﻿": "", "­": "",
})

_WS_RUN_RE = re.compile(r"[ \t\f\v]+")
_BLANK_LINES_RE = re.compile(r"\n\s*\n+")

# Hifenização de fim de linha: "San-\ntos". Aceita o hífen do jornal e "¬", a
# convenção do CATMuS. O jornal hifeniza; um modelo pode reunir a palavra,
# preservar a quebra ou manter o hífen e perder a quebra ("apresen-tando"), e
# nenhuma dessas escolhas é erro de transcrição.
#
# Só entre LETRAS: entre dígitos a quebra costuma ser fronteira de linha de
# tabela, e "12-" numa linha seguido de "15" na outra não é o número "1215".
#
# A quebra não some aqui: vira `_QUEBRA`, e quem decide se ela vira junção ou
# hífen é `_resolver_hifens`.
_HYPHEN_BREAK_RE = re.compile(r"(?<=[^\W\d_])[-¬]\n(?=[^\W\d_])")
_QUEBRA = ""  # área de uso privado: não é \w, não é espaço, não é pontuação
_PALAVRA_HIFENIZADA = re.compile(
    r"(?<![\w\-])[^\W\d_]+(?:[-][^\W\d_]+)+(?![\w\-])"
)

# Pontuação removida das BORDAS do token. A interna é preservada de propósito:
# valores em réis como "1:770$300" e numerações como "1.º" perderiam o sentido.
_EDGE_PUNCTUATION = ".,;:!?()[]{}\"'«»…"

# Token feito só de fio ou traço: "-", "---", "|", "* * *". É desenho da página
# — régua, pontilhado de índice, separador de célula — e não palavra. Sem esta
# regra o ponto final isolado sumia (é pontuação de borda) mas o travessão
# isolado contava como palavra, e quem desenhasse mais réguas errava mais.
# Fica no CER, que mede a página caractere a caractere.
#
# Não inclui "&", "%" nem "$": têm valor.
_SO_FIOS = re.compile(r"^[-_=|*~·•/\\]+$")


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
    except Exception as erro:  # noqa: BLE001 - html.parser é tolerante, mas não infalível
        # Devolver o HTML cru mantém a chamada no estudo, mas as tags entram
        # como palavras: o aviso é o que impede isso de passar calado.
        warnings.warn(f"html_to_text falhou ({erro!r}); usando o texto sem extração")
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

# O texto do título é opcional: um `##` sozinho na linha (o MinerU produz) é
# título vazio, não a palavra "##".
_MD_HEADING = re.compile(r"^\s{0,3}(#{1,6})(?:\s+(.*?))?\s*#*\s*$")
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

# HTML solto no Markdown, fora das tabelas. O MinerU embrulha o texto lido
# dentro de imagens em <details><summary>text_image</summary>...</details>: o
# rótulo do <summary> é dele, não da página, e sai com o conteúdo; o resto do
# bloco é texto da página e fica.
#
# Só tags HTML conhecidas saem. Uma regra genérica `<...>` apagaria texto: o
# OCR lê "«" como "<" e "<City of Santos Improvements Company" é página.
_MD_HTML_RESUMO = re.compile(r"<summary\b[^>]*>.*?</summary\s*>", re.IGNORECASE | re.DOTALL)
_MD_HTML_EM_LINHA = re.compile(  # inline: some sem deixar espaço, como no navegador
    r"</?(?:a|b|i|em|strong|u|s|small|sup|sub|font)\b[^>]*>", re.IGNORECASE
)
_MD_HTML_BLOCO = re.compile(  # separa palavras, como _BLOCK_TAGS/_CELL_TAGS
    r"</?(?:details|summary|div|p|br|hr|center|span|img|section)\b[^>]*/?>", re.IGNORECASE
)

# Matemática inline do MinerU: `$^{a}$` (ordinal), `$_{o}$`, `$\frac{1}{2}$`.
# O corpo tem de começar por `^`, `_` ou `\` e o `$` não pode estar escapado.
# É o que separa a fórmula dos valores em réis, que também usam `$`
# ("1:770$300 e 2:000$000" não pode virar fórmula).
_MD_MATEMATICA = re.compile(r"(?<!\\)\$\s*([\\^_][^$\n]{0,30}?)\s*(?<!\\)\$")
_TEX_FRACAO = re.compile(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")

# Escape de Markdown: `\$`, `\*`, `\_`, `\-`... O MinerU escapa o cifrão dos
# réis, e "7:500\$000" contra "7:500$000" errava a palavra E partia o número em
# dois no num_f1. A barra sai e fica o caractere — mas o caractere precisa
# atravessar as regras de ênfase e de fórmula sem ser lido como marcação: um
# `\*` literal não é ênfase e um `\$` literal não abre fórmula. Por isso ele
# viaja protegido na área de uso privado e só volta ao fim.
_MD_ESCAPE = re.compile(r"\\([!-/:-@\[-`{-~])")  # pontuação ASCII, como no CommonMark
_PROTEGIDO = 0xE100
_DESPROTEGER = {_PROTEGIDO + i: chr(i) for i in range(128)}
_TEX_COMANDO = re.compile(r"\\[A-Za-z]+\s*")


def _tex_para_texto(match: re.Match) -> str:
    """`^{a}` -> `a`, `\\frac{1}{2}` -> `1/2`: fica o que a fórmula exibe."""
    corpo = _TEX_FRACAO.sub(r"\1/\2", match.group(1)).replace("\\circ", "°")
    return re.sub(r"[\^_{}\s]", "", _TEX_COMANDO.sub("", corpo))


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

    md = _MD_ESCAPE.sub(lambda m: chr(_PROTEGIDO + ord(m.group(1))), md)
    corpo = _MD_HTML_RESUMO.sub("", _HTML_TABELA.sub(_guardar, md))

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
            # O MinerU põe fórmula também dentro da célula.
            tabela = html_to_text(fragmentos[int(marca.group(1))])
            saida.append(_MD_MATEMATICA.sub(_tex_para_texto, tabela))
            continue

        if not em_cerca:
            if _MD_REGRA.match(linha) or _MD_TABLE_SEP.match(linha):
                continue  # régua horizontal e separador de tabela não são texto
            titulo = _MD_HEADING.match(linha)
            if titulo:
                linha = titulo.group(2) or ""
            else:
                item = _MD_ULIST.match(linha) or _MD_OLIST.match(linha)
                if item:
                    linha = item.group(1)
                linha = _MD_QUOTE.sub("", linha)
            if "|" in linha:
                # Célula de tabela: a barra vira espaço, para que a linha da
                # tabela continue sendo uma linha de texto legível. Uma `\|`
                # escapada já está protegida e não passa por aqui.
                linha = linha.replace("|", " ")

        linha = _MD_IMAGE.sub(" ", linha)   # a imagem não é texto da página
        linha = _MD_LINK.sub(r"\1", linha)  # do link fica o rótulo
        linha = _MD_HTML_BLOCO.sub(" ", _MD_HTML_EM_LINHA.sub("", linha))
        # Antes da ênfase: ela levaria o `_` de `$_{o}$` e deixaria `${o}$`.
        linha = _MD_MATEMATICA.sub(_tex_para_texto, linha)
        saida.append(_MD_ENFASE.sub("", linha))

    return "\n".join(saida).translate(_DESPROTEGER)


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
    # Uma <table> sem nenhuma outra marcação em volta também é o formato do
    # MinerU: páginas só de tabela, ou de tabela e uma linha solta. Classificá-
    # la como texto corrido mandava as tags inteiras para a contagem de
    # palavras.
    if marcas_md or _HTML_TABELA.search(texto):
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
    if formato == "markdown":
        # O Markdown trata as próprias cercas (o conteúdo delas fica).
        texto = _decodificar_entidades(markdown_to_text(texto))
    else:
        # A cerca é formatação, não palavra; a prosa em volta dela, se houver,
        # continua contando.
        texto = _FENCE_LINE_RE.sub("", texto)
        # Fora do HTML ninguém decodifica `&amp;`: o extrator faz isso no
        # caminho do HTML, e no Markdown e no texto corrido a entidade virava
        # palavra.
        texto = html_to_text(texto) if formato == "html" else _decodificar_entidades(texto)
    return collapse_whitespace(texto)


def collapse_whitespace(texto: str) -> str:
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    texto = _BLANK_LINES_RE.sub("\n", _WS_RUN_RE.sub(" ", texto))
    return "\n".join(l for l in (linha.strip() for linha in texto.split("\n")) if l)


def normalize_text(texto: str) -> str:
    """Normalização de caracteres (sem tokenizar).

    A quebra hifenizada sai daqui marcada com `_QUEBRA`, não resolvida: ver
    `_resolver_hifens`.
    """
    texto = unicodedata.normalize("NFC", texto.translate(_TYPOGRAPHY_TABLE)).lower()
    return _HYPHEN_BREAK_RE.sub(_QUEBRA, collapse_whitespace(texto))


def _resolver_hifens(texto: str, vocabulario: set[str] | None) -> str:
    """Decide, palavra a palavra, entre a forma unida e a hifenizada.

    Sem vocabulário (é o caso da própria referência) a quebra de linha
    hifenizada é unida — é divisão silábica na quase totalidade dos casos — e o
    hífen interno fica como está.

    Com o vocabulário da referência (o caso da hipótese), uma palavra que só
    difere de uma palavra da página pela presença do hífen recebe a grafia da
    página. Isso cobre os três jeitos de transcrever a mesma divisão silábica —
    unir, quebrar a linha no hífen, ou manter o hífen e perder a quebra — e
    também o composto quebrado no fim da linha ("guarda-\\nchuva"), que unir
    às cegas transformaria em "guardachuva". A regra é a mesma para todos os
    sistemas e nunca troca uma letra: só decide sobre o hífen.
    """
    def resolver(match: re.Match) -> str:
        palavra = match.group(0)
        hifenizada = palavra.replace(_QUEBRA, "-")
        if vocabulario is not None:
            if hifenizada in vocabulario:
                return hifenizada
            unida = hifenizada.replace("-", "")
            if unida in vocabulario:
                return unida
        return palavra.replace(_QUEBRA, "")

    # O replace final pega a quebra que a regex não alcançou (colada a dígito).
    return _PALAVRA_HIFENIZADA.sub(resolver, texto).replace(_QUEBRA, "")


def tokenize(texto: str) -> list[str]:
    """Divide em tokens de palavra, tirando a pontuação só das bordas.

    Tokens que são só fio ou traço (`_SO_FIOS`) saem: são desenho da página.
    """
    return [
        tok for tok in (p.strip(_EDGE_PUNCTUATION) for p in texto.split())
        if tok and not _SO_FIOS.match(tok)
    ]


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


def prepare(bruto: str, referencia: NormalizedDoc | None = None) -> NormalizedDoc:
    """Executa a pipeline completa sobre um texto bruto.

    `referencia` é passada ao preparar uma HIPÓTESE: o vocabulário dela decide
    a hifenização (ver `_resolver_hifens`). A referência é preparada sem ela.
    """
    vocabulario = set(referencia.tokens) if referencia is not None else None
    normalizado = _resolver_hifens(normalize_text(to_plain_text(bruto)), vocabulario)
    return NormalizedDoc(
        normalized=normalizado,
        tokens=tuple(tokenize(normalizado)),
        chars=char_sequence(normalizado),
    )
