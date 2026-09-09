"""Orquestração das chamadas.

Três decisões explicam o arquivo inteiro:

* **A coleta é separada do cálculo.** `run` só chama o modelo e grava a
  resposta bruta; `score` calcula as métricas depois, offline, quantas vezes for
  preciso. A definição de uma métrica muda ao longo do trabalho; os dados
  coletados — que custaram dinheiro — não podem depender dela.

* **Concorrência só ENTRE modelos.** Dentro de um mesmo modelo as chamadas são
  serializadas, e todos os modelos locais dividem uma fila só — dois MinerU
  simultâneos disputariam a mesma GPU.

* **Retomada idempotente.** Uma execução interrompida (queda de rede, crédito
  esgotado) volta com `--resume`, refazendo só o que faltou ou falhou. Nenhuma
  chamada bem-sucedida é repetida — e paga duas vezes.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .config import ModelSpec, PromptSpec
from .dataset import Document
from .normalize import detectar_formato

CHAMADAS = "chamadas.jsonl"   # o índice tabular de todas as chamadas
CHAMADAS_DIR = "chamadas"     # uma pasta por chamada, com as evidências dela

# Os quatro arquivos de uma chamada. Os nomes são fixos: dentro da pasta, o que
# identifica o arquivo é o papel dele, não o nome do modelo ou do documento —
# esses já estão no caminho da pasta.
RESPOSTA = "resposta"          # + a extensão do formato devolvido
BRUTO = "bruto.json"
NORMALIZADO = "normalizado.txt"
REFERENCIA = "referencia.txt"


@dataclass(frozen=True)
class Task:
    """Uma chamada: modelo x documento x prompt x repetição."""

    model: ModelSpec
    document: Document
    prompt: PromptSpec
    repetition: int

    @property
    def key(self) -> str:
        return f"{self.model.id}|{self.document.doc_id}|{self.prompt.id}|{self.repetition}"

    @property
    def stem(self) -> str:
        return f"{self.document.doc_id}__{self.prompt.id}__r{self.repetition}"

    @property
    def pasta(self) -> str:
        """Pasta desta chamada, relativa à raiz da execução.

        Tudo que pertence a uma chamada mora junto: a resposta, o corpo bruto do
        provedor, o texto normalizado e a referência normalizada. Antes os
        quatro estavam em árvores separadas por tipo de arquivo, e conferir uma
        transcrição à mão exigia abrir três pastas diferentes.
        """
        return f"{CHAMADAS_DIR}/{self.model.slug}/{self.stem}"


def new_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def prompts_do_modelo(modelo: ModelSpec, prompts: list[PromptSpec]) -> list[PromptSpec]:
    """Prompts aplicáveis a um modelo.

    O MinerU não recebe instrução: roda UMA vez, sob o prompt sentinela `p0`.
    Os modelos instruídos nunca rodam sob ele. Sem essa regra o eixo de prompt
    viraria repetição idêntica para o MinerU e cada sentinela acrescentado ao
    registro dispararia uma rodada paga a mais em todos os modelos de API.
    """
    return [p for p in prompts if p.prompt_agnostic == modelo.prompt_agnostic]


def build_tasks(
    modelos: list[ModelSpec],
    documentos: list[Document],
    prompts: list[PromptSpec],
    repeticoes: int,
) -> list[Task]:
    # `load_models` devolve o registro inteiro; é aqui que os dois estados que
    # impedem uma chamada são aplicados. `enabled: false` diz "não chame";
    # `excluido` diz "não conte" — e um sistema que ninguém vai contar também
    # não vale ser pago, então nenhum dos dois é chamado.
    #
    # Nenhum dos dois em silêncio: quem lê a saída do `run` precisa ver por que
    # um sistema que está no registro não apareceu na execução.
    for modelo in modelos:
        if not modelo.enabled:
            print(f"[run] {modelo.id} desabilitado, não será chamado.")
        elif modelo.excluido:
            print(f"[run] {modelo.id} fora da análise, não será chamado: {modelo.excluido}")
    modelos = [m for m in modelos if m.enabled and not m.excluido]
    return [
        Task(model=modelo, document=doc, prompt=prompt, repetition=rep)
        for modelo in modelos
        for doc in documentos
        for prompt in prompts_do_modelo(modelo, prompts)
        for rep in range(repeticoes)
    ]


def ler_chamadas(run_dir: Path) -> list[dict]:
    """Lê o `chamadas.jsonl` de uma execução.

    O arquivo é escrito como JSON Lines — um registro compacto por linha —, mas
    a leitura NÃO exige isso. Basta abrir um `.jsonl` num editor com formatação
    automática para ele voltar ao disco indentado, com um registro espalhado por
    trinta linhas; foi o que aconteceu com a primeira execução deste projeto. Um
    leitor estrito quebraria ali, e `chaves_concluidas` devolveria um conjunto
    vazio *em silêncio* — fazendo o `--resume` refazer, e pagar de novo, tudo
    que já estava pronto.

    Por isso a leitura é por valores JSON consecutivos, e não por linhas: aceita
    as duas formas. O arquivo custou dinheiro; ele não deve depender de nunca
    ter sido aberto num editor.
    """
    caminho = run_dir / CHAMADAS
    if not caminho.exists():
        return []
    texto = caminho.read_text(encoding="utf-8").strip()
    if not texto:
        return []

    decodificador = json.JSONDecoder()
    registros: list[dict] = []
    posicao = 0
    while posicao < len(texto):
        try:
            registro, posicao = decodificador.raw_decode(texto, posicao)
        except json.JSONDecodeError as exc:
            linha = texto.count("\n", 0, posicao) + 1
            raise ValueError(
                f"{caminho} está corrompido perto da linha {linha}: {exc.msg}"
            ) from exc
        registros.append(registro)
        while posicao < len(texto) and texto[posicao] in " \t\r\n":
            posicao += 1
    return registros


def chaves_concluidas(run_dir: Path) -> set[str]:
    """Chaves que já deram certo numa execução anterior."""
    return {
        f"{r['model_id']}|{r['doc_id']}|{r['prompt_id']}|{r['repetition']}"
        for r in ler_chamadas(run_dir)
        if r.get("ok")
    }


class Escritor:
    """Grava o que cada chamada deixou, tudo na pasta dela.

        chamadas/<modelo>/<doc>__<prompt>__r<n>/
            resposta.<ext>    o texto devolvido, e só ele
            bruto.json        a resposta do provedor verbatim, uma por tentativa
            normalizado.txt   (escrito depois, pelo `score`)
            referencia.txt    (idem)

    Mais uma linha em `chamadas.jsonl`, que é o índice: ele responde "quanto
    custou, quanto demorou, o que falhou" sem abrir pasta nenhuma, e a pasta
    responde "o que exatamente esse modelo devolveu".
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self._lock = threading.Lock()

    def escrever(self, task: Task, resultado) -> None:  # noqa: ANN001
        pasta = self.run_dir / task.pasta
        pasta.mkdir(parents=True, exist_ok=True)

        resposta = None
        if resultado.text:
            # A extensão segue o formato do que o modelo REALMENTE devolveu — o
            # mesmo classificador que as métricas usam. Chamar de `.html` o
            # Markdown do MinerU obrigaria a renomear na mão para lê-lo.
            extensao = {"html": ".html", "markdown": ".md"}.get(
                detectar_formato(resultado.text), ".txt"
            )
            resposta = f"{RESPOSTA}{extensao}"
            (pasta / resposta).write_text(resultado.text, encoding="utf-8")

        (pasta / BRUTO).write_text(
            json.dumps(
                {
                    "chave": resultado.key,
                    "registro": resultado.to_record(),
                    # Uma entrada por tentativa: um sucesso na terceira mantém
                    # visíveis os dois erros que vieram antes.
                    "respostas_brutas": resultado.respostas_brutas,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        registro = {
            **resultado.to_record(),
            # Metadados do modelo e do prompt viajam com a chamada: assim o
            # `score` e as tabelas não dependem de o models.yaml continuar
            # igual meses depois.
            "empresa": task.model.empresa,
            "tipo": task.model.tipo,
            "model_label": task.model.label,
            "prompt_label": task.prompt.label,
            "prompt_agnostic": task.prompt.prompt_agnostic,
            # Sempre com "/", inclusive no Windows: o caminho vai para um CSV
            # que pode ser lido noutra máquina, e `Path` aceita os dois.
            "pasta": task.pasta,
            # Só o nome: a extensão varia com o formato devolvido, os outros
            # três arquivos da pasta têm nome fixo.
            "resposta": resposta,
        }
        with self._lock:
            with open(self.run_dir / CHAMADAS, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(registro, ensure_ascii=False) + "\n")


def executar(
    cliente,  # noqa: ANN001
    tasks: list[Task],
    escritor: Escritor,
    max_parallel_models: int = 3,
    sleep_between_calls: float = 1.0,
    ao_terminar=None,  # noqa: ANN001
) -> None:
    """Roda as tarefas: serial dentro do modelo, paralelo entre modelos.

    Os modelos locais compartilham uma fila só (`__local__`): dois processos do
    MinerU simultâneos ou estouram a VRAM ou medem um tempo que é de contenção
    pela GPU, não do backend. Modelos de API continuam paralelos entre si e
    paralelos à fila local, porque ali quem espera é a rede.
    """
    por_modelo: dict[str, list[Task]] = {}
    for task in tasks:
        chave = "__local__" if task.model.local else task.model.id
        por_modelo.setdefault(chave, []).append(task)

    def rodar(fila: list[Task]) -> None:
        for i, task in enumerate(fila):
            resultado = cliente.transcribe(
                task.model, task.prompt, task.document, task.repetition
            )
            escritor.escrever(task, resultado)
            if ao_terminar:
                ao_terminar(task, resultado)
            if sleep_between_calls and i < len(fila) - 1:
                time.sleep(sleep_between_calls)

    workers = max(1, min(max_parallel_models, len(por_modelo)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(rodar, por_modelo.values()))
