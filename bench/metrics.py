"""As métricas do estudo.

São quatro funções e um pouco de aritmética:

| métrica                 | o que responde                                        |
|-------------------------|-------------------------------------------------------|
| `wer`                   | quantas PALAVRAS da página o sistema errou            |
| `cer`                   | quantos CARACTERES da página o sistema errou          |
| `num_f1`                | recuperou os NÚMEROS (réis, datas, processos)?        |
| `html_sem_texto_extra`  | obedeceu ao "retorne apenas o código HTML"?           |

O custo em tokens é a quinta medida e não mora aqui: ele vem do provedor junto
da resposta e é gravado pelo `runner.py` (tokens de entrada, tokens de saída e
US$ por chamada).

WER e CER são a mesma conta em unidades diferentes: distância de Levenshtein
dividida pelo tamanho da referência. A distância é o menor número de
substituições, remoções e inserções que transforma a hipótese na referência —
`(S + D + I) / N_ref`, a definição usual. **Pode passar de 1,0**: uma alucinação
longa acrescenta inserções sem aumentar o denominador.
"""

from __future__ import annotations

import re
from collections import Counter

from rapidfuzz.distance import Levenshtein

from .normalize import NormalizedDoc, detectar_formato, strip_code_fences


def _taxa_de_erro(referencia, hipotese) -> float | None:  # noqa: ANN001
    """Distância de edição normalizada pelo tamanho da referência."""
    if not referencia:
        return None
    return Levenshtein.distance(list(referencia), list(hipotese)) / len(referencia)


def wer(ref: NormalizedDoc, hyp: NormalizedDoc) -> float | None:
    """Word Error Rate: erro por palavra, na ordem em que aparecem."""
    return _taxa_de_erro(ref.tokens, hyp.tokens)


def cer(ref: NormalizedDoc, hyp: NormalizedDoc) -> float | None:
    """Character Error Rate: erro por caractere."""
    return _taxa_de_erro(ref.chars, hyp.chars)


# --------------------------------------------------------------------------
# Números
# --------------------------------------------------------------------------

# Sequências numéricas com os separadores internos preservados:
#   "1:770$300"  "761,2"  "5605"  "22,7"  "1.º"
_NUM_RE = re.compile(r"\d+(?:[.,:$/\-]\d+)*")


def num_f1(ref: NormalizedDoc, hyp: NormalizedDoc) -> dict:
    """F1 dos valores numéricos da página.

    O Diário Oficial é denso em valores cujo erro é qualitativamente mais grave
    que o de uma palavra qualquer: réis, leituras meteorológicas, números de
    processo, datas. Um modelo pode ter WER baixo e ser inútil para pesquisa
    histórica se errar sistematicamente os números — e o contrário também
    acontece.

    A comparação é por multiconjunto, não por posição: interessa se os valores
    da página foram recuperados, não onde foram parar.

    Numa página SEM número nenhum na referência a medida não se aplica e
    `num_f1` sai vazio — ver abaixo.
    """
    numeros_ref = Counter(_NUM_RE.findall(ref.normalized))
    numeros_hyp = Counter(_NUM_RE.findall(hyp.normalized))

    n_ref = sum(numeros_ref.values())
    n_hyp = sum(numeros_hyp.values())
    acertos = sum((numeros_ref & numeros_hyp).values())

    # Referência sem número nenhum: não há o que recuperar, e a medida NÃO SE
    # APLICA. Devolver 0,0 aqui dava nota zero a todos os sistemas — inclusive
    # a quem acertou justamente por não inventar número que a página não tem —
    # e essa nota entrava na média do modelo como se fosse falha de
    # transcrição. Vazio é o que a coluna significa ali, exatamente como
    # `html_sem_texto_extra` sob o prompt sentinela.
    if not n_ref:
        return {"num_n_ref": 0, "num_n_hyp": n_hyp, "num_acertos": 0, "num_f1": None}

    precisao = acertos / n_hyp if n_hyp else 0.0
    revocacao = acertos / n_ref
    # Referência COM números e hipótese sem nenhum continua sendo 0,0, e está
    # certo: o sistema não recuperou nada.
    f1 = 2 * precisao * revocacao / (precisao + revocacao) if (precisao + revocacao) else 0.0

    return {"num_n_ref": n_ref, "num_n_hyp": n_hyp, "num_acertos": acertos, "num_f1": f1}


# --------------------------------------------------------------------------
# Obediência ao formato pedido
# --------------------------------------------------------------------------

def html_sem_texto_extra(bruto: str) -> bool:
    """A resposta é só o documento HTML, sem prosa em volta?

    O prompt termina com "retorne apenas o código HTML sem texto extra", e nem
    todo modelo obedece: os desobedientes abrem com "Claro! Aqui está a
    transcrição:" ou fecham com "Espero ter ajudado". Isso não é erro de
    transcrição — o WER quase não sente — mas é trabalho manual para quem for
    usar a saída, e é por isso que vira uma medida própria.

    O critério é deliberadamente simples e verificável a olho: depois de tirar
    uma cerca de código que envolva a resposta inteira (```html ... ```, que é
    formatação e não texto), o que sobra tem de ser HTML e tem de começar em
    `<` e terminar em `>`. Qualquer prosa antes ou depois quebra uma das duas
    pontas.
    """
    texto = strip_code_fences(bruto or "").strip()
    if not texto or detectar_formato(texto) != "html":
        return False
    return texto.startswith("<") and texto.endswith(">")


# --------------------------------------------------------------------------

def calcular(
    ref: NormalizedDoc, hyp: NormalizedDoc, hyp_bruto: str, pede_html: bool = True
) -> dict:
    """Todas as métricas de uma chamada, num dicionário só.

    `pede_html=False` para os sistemas que rodam sob o prompt sentinela `p0` (o
    MinerU): eles nunca receberam a instrução "retorne apenas o código HTML", e
    marcá-los como False seria registrar desobediência a uma ordem que não foi
    dada. A coluna sai vazia, e vazio é o que ela significa: não se aplica.
    """
    return {
        "wer": wer(ref, hyp),
        "cer": cer(ref, hyp),
        **num_f1(ref, hyp),
        "html_sem_texto_extra": html_sem_texto_extra(hyp_bruto) if pede_html else None,
        "saida_formato": detectar_formato(hyp_bruto),
        "ref_n_tokens": ref.n_tokens,
        "hyp_n_tokens": hyp.n_tokens,
    }
