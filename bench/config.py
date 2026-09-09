"""Carga dos três arquivos de configuração.

`settings.yaml` é lido como dicionário cru (com atalhos para o que se usa
sempre); `models.yaml` e `prompts.yaml` viram listas de dataclasses. Campos que
o código não usa — licença, família, notas, número de parâmetros — continuam
nos YAML e são simplesmente ignorados aqui.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROVEDORES = {"openrouter", "gemini", "mineru"}


def project_root() -> Path:
    """Raiz do projeto (a pasta que contém `bench/` e `config/`)."""
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelSpec:
    id: str
    empresa: str
    tipo: str  # "aberto" | "fechado"
    label: str
    enabled: bool = True

    # Por onde o modelo é chamado. Os Gemini proprietários vão pela API nativa
    # do Google; o MinerU roda nesta máquina; o resto vai pelo OpenRouter.
    provider: str = "openrouter"
    # ID no provedor, quando difere do `id` usado como identidade no estudo.
    api_model_id: str | None = None

    # Endpoint exigido dentro do OpenRouter: casa E faixa de serviço, no
    # formato do catálogo ("openai", "anthropic", "alibaba",
    # "moonshotai/mxfp4"). Só faz sentido com `provider: openrouter`.
    #
    # Precisa fixar as DUAS coisas. A casa, porque as casas de um mesmo modelo
    # aberto diferem em quantização — o Kimi K3 é servido em fp4, mxfp4, fp8 e
    # bf16 conforme quem atende, e trocar de casa no meio da execução mudaria o
    # WER como se fosse propriedade do modelo. A faixa, porque a mesma casa
    # vende o mesmo modelo em níveis de serviço com preços diferentes: o
    # gpt-5.6-sol sai a 1/5 em `openai/flex`, 2/10 em `openai` e 4/20 em
    # `openai/fast`. Fixar só a casa deixa a faixa por conta do roteador, que
    # entrega a padrão — o dobro do preço da flex, sem avisar.
    #
    # A tag exata vem de `GET /api/v1/models/<id>/endpoints` e é conferida por
    # `bench check`.
    endpoint_tag: str | None = None

    # Preço de tabela (US$ por milhão de tokens). Só é usado quando o provedor
    # não devolve o custo real da chamada — o caso da API nativa do Gemini, que
    # reporta tokens mas não o valor cobrado.
    preco_entrada_usd_mtok: float | None = None
    preco_saida_usd_mtok: float | None = None

    # Teto de chamadas por minuto para este modelo.
    rpm_limit: float | None = None

    # Motivo pelo qual este sistema saiu da ANALISE, ou None se esta nela.
    #
    # Excluir e diferente de desabilitar. `enabled: false` diz "nao chame";
    # `excluido` diz "nao conte" — o sistema nao e chamado, nao aparece em
    # figura nem em tabela, e as medicoes que ele ja produziu continuam no
    # disco, intactas, porque custaram dinheiro e continuam sendo evidencia.
    #
    # O campo guarda o MOTIVO, e nao um booleano, de proposito: "por que este
    # sistema nao esta nos resultados?" e uma pergunta que um leitor vai fazer,
    # e a resposta tem que morar junto do sistema e nao na memoria de quem
    # tomou a decisao.
    excluido: str | None = None

    # "none" | "minimal" | "low" | "medium" | "high" | "max". Só é enviado
    # quando declarado: nos modelos que raciocinam por padrão, os tokens de
    # pensamento são cobrados como saída e distorcem o custo por página.
    reasoning_effort: str | None = None

    # Parâmetros do backend local (MinerU): backend, effort, method...
    backend_options: dict[str, Any] = field(default_factory=dict)

    @property
    def slug(self) -> str:
        """Identificador seguro para nome de pasta."""
        return self.id.replace("/", "__").replace(":", "-")

    @property
    def remote_id(self) -> str:
        return self.api_model_id or self.id

    @property
    def prompt_agnostic(self) -> bool:
        """O sistema não recebe instrução em linguagem natural.

        É o caso do MinerU, que é um analisador de documentos e não um modelo
        instruído. Ele roda uma única vez, sob o prompt sentinela `p0`, e nunca
        sob `p1` — ver `runner.prompts_do_modelo`.
        """
        return self.provider == "mineru"

    @property
    def local(self) -> bool:
        return self.provider == "mineru"


@dataclass(frozen=True)
class PromptSpec:
    id: str
    text: str
    label: str
    enabled: bool = True
    # Marcador de "este sistema não recebeu instrução nenhuma" (o p0).
    prompt_agnostic: bool = False


@dataclass
class Settings:
    raw: dict[str, Any]

    def section(self, name: str) -> dict[str, Any]:
        return self.raw.get(name, {}) or {}

    @property
    def dataset_root(self) -> Path:
        root = Path(self.section("dataset").get("root", "Dataset"))
        return root if root.is_absolute() else project_root() / root

    @property
    def active_documents(self) -> list[str]:
        return list(self.section("dataset").get("active") or [])

    @property
    def results_dir(self) -> Path:
        return project_root() / "results"

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.section("api").get("api_key_env", "OPENROUTER_API_KEY"))

    @property
    def gemini_api_key(self) -> str | None:
        return os.environ.get(self.section("gemini").get("api_key_env", "GEMINI_API_KEY"))


@dataclass
class Config:
    settings: Settings
    models: list[ModelSpec]
    prompts: list[PromptSpec]


def _ler_yaml(path: Path):  # noqa: ANN201
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _float_ou_none(valor) -> float | None:  # noqa: ANN001
    return None if valor in (None, "") else float(valor)


def load_settings(path: Path | None = None) -> Settings:
    path = path or project_root() / "config" / "settings.yaml"
    return Settings(raw=_ler_yaml(path) or {})


def load_models(path: Path | None = None) -> list[ModelSpec]:
    """Lê models.yaml validando na carga, e devolve o registro COMO ESCRITO.

    A validação é aqui, e não no meio de uma execução paga: um `tipo` errado ou
    um MinerU sem backend declarado devem falhar no `check`.

    Nada é filtrado aqui — nem `enabled: false`, nem `excluido`. Quem decide o
    que fazer com cada estado é quem chama: `runner.build_tasks` não chama os
    desabilitados nem os excluídos, e a análise (`score`, `graficos`) não conta
    os excluídos. Filtrar na carga desligava a exclusão em silêncio: um modelo
    marcado `excluido` E `enabled: false` sumia do registro, e as linhas que ele
    já tinha produzido voltavam para as figuras como se fossem de um sistema
    desconhecido, sem ninguém para explicar o que eram.
    """
    path = path or project_root() / "config" / "models.yaml"
    entradas = (_ler_yaml(path) or {}).get("models") or []

    modelos: list[ModelSpec] = []
    vistos: set[str] = set()
    for entrada in entradas:
        model_id = str(entrada.get("id") or "").strip()
        if not model_id:
            raise ValueError(f"{path}: há uma entrada sem 'id'.")
        if model_id in vistos:
            raise ValueError(f"{path}: modelo duplicado '{model_id}'.")
        vistos.add(model_id)

        tipo = str(entrada.get("tipo", "")).strip().lower()
        if tipo not in {"aberto", "fechado"}:
            raise ValueError(f"{path}: '{model_id}' tem tipo='{tipo}'; use 'aberto' ou 'fechado'.")

        provider = str(entrada.get("provider") or "openrouter").strip().lower()
        if provider not in PROVEDORES:
            raise ValueError(
                f"{path}: '{model_id}' tem provider='{provider}'; use um de {sorted(PROVEDORES)}."
            )

        backend_options = dict(entrada.get("backend_options") or {})
        if provider == "mineru" and not backend_options.get("backend"):
            raise ValueError(
                f"{path}: '{model_id}' usa provider 'mineru' mas não declara "
                "backend_options.backend (pipeline | vlm-engine | hybrid-engine)."
            )

        modelos.append(
            ModelSpec(
                id=model_id,
                empresa=str(entrada.get("empresa") or "").strip(),
                tipo=tipo,
                label=str(entrada.get("label") or model_id),
                enabled=bool(entrada.get("enabled", True)),
                provider=provider,
                api_model_id=entrada.get("api_model_id"),
                endpoint_tag=entrada.get("endpoint_tag"),
                preco_entrada_usd_mtok=_float_ou_none(entrada.get("preco_entrada_usd_mtok")),
                preco_saida_usd_mtok=_float_ou_none(entrada.get("preco_saida_usd_mtok")),
                rpm_limit=_float_ou_none(entrada.get("rpm_limit")),
                excluido=(str(entrada["excluido"]).strip()
                          if entrada.get("excluido") else None),
                reasoning_effort=entrada.get("reasoning_effort"),
                backend_options=backend_options,
            )
        )
    return modelos


def motivos_de_exclusao(modelos: list[ModelSpec] | None = None) -> dict[str, str]:
    """`model_id` -> motivo, para os sistemas que saíram da ANÁLISE.

    Uma regra num lugar só. Ela decide quem fica de fora do `resumo.csv`, das
    figuras e da tabela do relatório — três números que se apresentam como o
    mesmo e que, com a regra duplicada, passariam a responder a perguntas
    diferentes. Foi o que aconteceu: a exclusão valia nas figuras e não no
    `score`, e o `resumo.csv` publicava a média de um sistema que a análise já
    tinha dispensado.

    Excluir não é apagar: as medições continuam no `chamadas.jsonl`, nas pastas
    de chamada e no `metricas.csv`, que é a tabela de auditoria. O que a
    exclusão tira é o direito de virar média apresentada.
    """
    return {m.id: m.excluido for m in (modelos or load_models()) if m.excluido}


def load_prompts(path: Path | None = None) -> list[PromptSpec]:
    path = path or project_root() / "config" / "prompts.yaml"
    entradas = _ler_yaml(path) or []
    prompts = [
        PromptSpec(
            id=str(e["id"]).strip(),
            text=str(e["text"]),
            label=str(e.get("label") or e["id"]),
            enabled=bool(e.get("enabled", True)),
            prompt_agnostic=bool(e.get("prompt_agnostic", False)),
        )
        for e in entradas
    ]
    return [p for p in prompts if p.enabled]


def load_all() -> Config:
    return Config(settings=load_settings(), models=load_models(), prompts=load_prompts())
