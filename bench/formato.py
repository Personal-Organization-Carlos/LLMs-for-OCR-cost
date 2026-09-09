"""Formatação de número para leitura em português.

Existe como módulo próprio porque `graficos.py` e `relatorio.py` formatam os
MESMOS números, dos mesmos dados, para o mesmo leitor — e mantinham cada um a
sua cópia da função, com regra de milhar diferente. A divergência não aparecia
(todo valor da tabela é menor que mil, faixa em que as duas coincidem) e
apareceria no dia em que um valor passasse disso, numa figura só e não na
outra.
"""

from __future__ import annotations


def pt(valor: float, casas: int = 3) -> str:
    """Número em português: vírgula decimal, ponto de milhar.

    O caractere nulo é um pivô: `format` produz o padrão inglês (`1,234.56`) e
    trocar vírgula por ponto diretamente sobrescreveria o que a troca anterior
    acabou de escrever.
    """
    return f"{valor:,.{casas}f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def usd(valor: float, casas: int = 2) -> str:
    r"""Valor em dólar, com o cifrão escapado.

    Entre dois `$` o matplotlib entra em modo fórmula: "US$ 383 · US$ 0,0038"
    saía em itálico matemático, como se fosse uma equação. A barra invertida
    desliga a interpretação. É inofensivo fora do matplotlib.
    """
    return rf"US\$ {pt(valor, casas)}"
