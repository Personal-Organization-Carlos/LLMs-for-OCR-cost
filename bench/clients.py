"""Os três jeitos de transcrever uma página.

    OpenRouter  ->  a maioria dos modelos, por HTTP
    Gemini      ->  os proprietários do Google, pela API nativa
    MinerU      ->  o analisador de documentos que roda nesta máquina

Os três expõem o mesmo método — `transcribe(modelo, prompt, documento, rep)` —
e devolvem o mesmo `Resultado`. O `Roteador` escolhe qual usar a partir do
`provider` declarado em `models.yaml`, e é ele que o `runner` recebe.

As chamadas são feitas SEM streaming. É mais simples e, para o que se mede
aqui, melhor: o `finish_reason` sempre chega, e é ele que denuncia a resposta
truncada — a falha mais perigosa do experimento, porque produz um WER
catastrófico que parece incompetência do modelo mas é teto de tokens.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import requests

from .config import ModelSpec, PromptSpec, Settings, project_root
from .dataset import Document

_MIME_POR_EXTENSAO = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

_MOTIVOS_DE_TRUNCAMENTO = {"length", "max_tokens", "MAX_TOKENS"}


@dataclass
class Resultado:
    """O que uma chamada devolve, tenha ela dado certo ou não."""

    model_id: str
    prompt_id: str
    doc_id: str
    repetition: int

    ok: bool = False
    text: str = field(default="", repr=False)
    error: str | None = None
    error_kind: str | None = None  # api_error | rate_limit | timeout | vazio | recusa | excecao

    latency_s: float | None = None

    # Custo em tokens: as três contagens e o valor em dólar.
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    # De onde saiu o valor: "usage" e "generation_api" são o que o OpenRouter
    # cobrou de fato; "tabela_precos" é preço declarado em models.yaml x tokens
    # (o caso do Gemini, cuja API não informa valor); "local" é o MinerU.
    cost_source: str | None = None

    finish_reason: str | None = None
    truncated: bool = False

    api_provider: str | None = None   # openrouter | gemini | mineru
    provider_upstream: str | None = None
    generation_id: str | None = None
    attempts: int = 0
    timestamp_utc: str = ""

    # A resposta como ela chegou, sem interpretação — uma entrada por TENTATIVA,
    # na ordem em que aconteceram. Não entra no `chamadas.jsonl`: vai inteira
    # para `chamadas/<modelo>/<doc>__<prompt>__r<n>/bruto.json`.
    #
    # Guardar isto é assimétrico e por isso vale a pena: o disco é barato e a
    # chamada não. Todo campo que o `Resultado` não extrai — tokens nativos,
    # desconto de cache, o id upstream, o corpo de um erro 429 — só existe aqui;
    # sem este arquivo, recuperá-lo exigiria refazer a chamada e pagar de novo.
    respostas_brutas: list[dict] = field(default_factory=list, repr=False)

    @property
    def key(self) -> str:
        return f"{self.model_id}|{self.doc_id}|{self.prompt_id}|{self.repetition}"

    def to_record(self) -> dict:
        """Linha do `chamadas.jsonl`.

        Sem o texto (que vai para `resposta.<ext>`, na pasta da chamada) e sem
        as respostas brutas (que vão para `bruto.json`, na mesma pasta): os
        dois são grandes e tornariam o JSONL ilegível.
        """
        registro = asdict(self)
        registro.pop("text", None)
        registro.pop("respostas_brutas", None)
        return registro


def _agora_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _imagem_base64(caminho: Path) -> tuple[str, str]:
    mime = _MIME_POR_EXTENSAO.get(caminho.suffix.lower(), "image/jpeg")
    return mime, base64.b64encode(caminho.read_bytes()).decode("ascii")


def _classificar(exc: Exception) -> str:
    """429 é limite de cota, não incapacidade do modelo: vale distinguir."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    texto = str(exc)
    if status == 429 or "429" in texto or "RESOURCE_EXHAUSTED" in texto:
        return "rate_limit"
    if status and 500 <= status < 600:
        return "servidor"
    return "api_error"


def _custo_pela_tabela(modelo: ModelSpec, entrada: int | None, saida: int | None) -> float | None:
    """Custo estimado quando o provedor não devolve o valor cobrado."""
    if modelo.preco_entrada_usd_mtok is None and modelo.preco_saida_usd_mtok is None:
        return None
    return (
        (entrada or 0) * (modelo.preco_entrada_usd_mtok or 0.0)
        + (saida or 0) * (modelo.preco_saida_usd_mtok or 0.0)
    ) / 1e6


class _Espera:
    """Espaçamento mínimo entre chamadas de um mesmo modelo.

    Cotas de camada gratuita se contam em requisições por minuto, e estourá-las
    devolve 429 — que o retry trata, mas ao custo de uma espera que seria
    medida como latência do modelo. Segurar a chamada antes de fazê-la é mais
    barato e mantém o tempo limpo.
    """

    def __init__(self) -> None:
        self._proxima: dict[str, float] = {}

    def aguardar(self, model_id: str, rpm: float | None) -> None:
        if not rpm or rpm <= 0:
            return
        agora = time.monotonic()
        atraso = max(0.0, self._proxima.get(model_id, 0.0) - agora)
        self._proxima[model_id] = max(agora, self._proxima.get(model_id, 0.0)) + 60.0 / rpm
        if atraso:
            time.sleep(atraso)


class _ClienteHTTP:
    """O que OpenRouter e Gemini têm em comum: retry, espera e contabilidade."""

    api_provider = "http"

    def __init__(self, settings: Settings, secao: str) -> None:
        self.settings = settings
        self.cfg = settings.section(secao)
        self.timeout = float(self.cfg.get("timeout_seconds", 900))
        self.max_retries = int(self.cfg.get("max_retries", 3))
        self.backoff = float(self.cfg.get("retry_backoff_seconds", 5))
        geracao = settings.section("generation")
        self.temperature = float(geracao.get("temperature", 0.0))
        self.max_tokens = int(geracao.get("max_tokens", 32000))
        self.session = requests.Session()
        self.espera = _Espera()

    def transcribe(
        self, modelo: ModelSpec, prompt: PromptSpec, documento: Document, repetition: int = 0
    ) -> Resultado:
        resultado = Resultado(
            model_id=modelo.id,
            prompt_id=prompt.id,
            doc_id=documento.doc_id,
            repetition=repetition,
            api_provider=self.api_provider,
            timestamp_utc=_agora_utc(),
        )
        erro = tipo = None
        for tentativa in range(1, self.max_retries + 1):
            resultado.attempts = tentativa
            self.espera.aguardar(modelo.id, self._rpm(modelo))
            inicio = time.perf_counter()
            try:
                corpo = self._chamar(modelo, prompt, documento, resultado)
                resultado.latency_s = time.perf_counter() - inicio
                self._preencher(corpo, resultado, modelo)
            except requests.Timeout as exc:
                erro, tipo = str(exc), "timeout"
            except requests.RequestException as exc:
                erro, tipo = str(exc), _classificar(exc)
            except Exception as exc:  # noqa: BLE001
                erro, tipo = f"{type(exc).__name__}: {exc}", "excecao"
            else:
                if resultado.text.strip():
                    resultado.ok = True
                    return resultado
                erro, tipo = self._motivo_da_resposta_vazia(resultado)

            if tentativa < self.max_retries:
                # 429 espera mais: a cota se recompõe por janela de tempo.
                time.sleep(self.backoff * tentativa * (4 if tipo == "rate_limit" else 1))

        resultado.ok = False
        resultado.error, resultado.error_kind = erro, tipo
        resultado.latency_s = None
        return resultado

    # Preenchidos pelas subclasses -------------------------------------
    def _rpm(self, modelo: ModelSpec) -> float | None:
        return modelo.rpm_limit

    def _motivo_da_resposta_vazia(self, resultado: Resultado) -> tuple[str, str]:
        return f"resposta vazia (finish_reason={resultado.finish_reason})", "vazio"

    def _chamar(self, modelo, prompt, documento, resultado) -> dict:  # noqa: ANN001
        raise NotImplementedError

    def _preencher(self, corpo: dict, resultado: Resultado, modelo: ModelSpec) -> None:
        raise NotImplementedError

    @staticmethod
    def _guardar_bruto(resposta, resultado: Resultado, etapa: str) -> dict:  # noqa: ANN001
        """Registra a resposta verbatim e devolve o corpo já convertido.

        Roda ANTES de qualquer verificação de status: o corpo de um erro é
        justamente o que explica a falha (qual cota estourou, qual campo o
        provedor recusou), e é o que se perde se a exceção subir primeiro.
        """
        try:
            corpo = resposta.json()
        except ValueError:
            corpo = {"corpo_nao_e_json": resposta.text[:8000]}
        resultado.respostas_brutas.append(
            {
                "etapa": etapa,
                "tentativa": resultado.attempts,
                "http_status": resposta.status_code,
                "corpo": corpo,
            }
        )
        return corpo if isinstance(corpo, dict) else {}


class OpenRouterClient(_ClienteHTTP):
    """A maioria dos modelos, com o custo real cobrado pela chamada."""

    api_provider = "openrouter"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings, "api")
        self.base_url = self.cfg.get("base_url", "https://openrouter.ai/api/v1").rstrip("/")
        if not settings.api_key:
            env = self.cfg.get("api_key_env", "OPENROUTER_API_KEY")
            raise RuntimeError(
                f"Chave de API não encontrada. Defina a variável de ambiente {env}.\n"
                f"  PowerShell:  $env:{env} = 'sk-or-...'"
            )
        self.session.headers.update(
            {"Authorization": f"Bearer {settings.api_key}", "Content-Type": "application/json"}
        )
        # Com a casa de hospedagem fixada, um desvio para outra casa é FALHA, não
        # alternativa: a chamada que voltasse de outra casa teria outra
        # quantização e entraria na média como se fosse o mesmo sistema.
        self.permitir_fallback = bool(self.cfg.get("permitir_fallback_de_hospedagem", False))

    def _chamar(
        self, modelo: ModelSpec, prompt: PromptSpec, documento: Document, resultado: Resultado
    ) -> dict:
        mime, dados = _imagem_base64(documento.image_path)
        payload: dict[str, Any] = {
            "model": modelo.remote_id,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            # Pede ao OpenRouter que devolva o custo junto do uso, evitando uma
            # segunda consulta em quase todas as chamadas.
            "usage": {"include": True},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt.text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{dados}"},
                        },
                    ],
                }
            ],
        }
        if modelo.reasoning_effort:
            # Modelos que raciocinam por padrão cobram os tokens de pensamento
            # como saída. Sem fixar o esforço, o custo por página deixa de ser
            # comparável entre modelos.
            payload["reasoning"] = (
                {"enabled": False}
                if modelo.reasoning_effort == "none"
                else {"effort": modelo.reasoning_effort}
            )
        if modelo.endpoint_tag:
            # Um endpoint só — casa e faixa —, declarado em models.yaml. O
            # `provider_upstream` do resultado devolve quem serviu de fato, e o
            # custo cobrado confirma a faixa: é a prova de que a fixação pegou.
            payload["provider"] = {
                "order": [modelo.endpoint_tag],
                "allow_fallbacks": self.permitir_fallback,
            }

        resposta = self.session.post(
            f"{self.base_url}/chat/completions", data=json.dumps(payload), timeout=self.timeout
        )
        corpo = self._guardar_bruto(resposta, resultado, "chat/completions")
        resposta.raise_for_status()
        if corpo.get("error") and not corpo.get("choices"):
            raise requests.RequestException(json.dumps(corpo["error"])[:500])
        return corpo

    def _preencher(self, corpo: dict, resultado: Resultado, modelo: ModelSpec) -> None:
        escolha = (corpo.get("choices") or [{}])[0]
        resultado.text = (escolha.get("message") or {}).get("content") or ""
        resultado.finish_reason = escolha.get("finish_reason") or escolha.get(
            "native_finish_reason"
        )
        resultado.truncated = resultado.finish_reason in _MOTIVOS_DE_TRUNCAMENTO
        resultado.generation_id = corpo.get("id")
        resultado.provider_upstream = corpo.get("provider")

        uso = corpo.get("usage") or {}
        resultado.prompt_tokens = uso.get("prompt_tokens")
        resultado.completion_tokens = uso.get("completion_tokens")
        resultado.total_tokens = uso.get("total_tokens")
        if uso.get("cost") is not None:
            resultado.cost_usd, resultado.cost_source = float(uso["cost"]), "usage"
        else:
            self._buscar_custo(resultado)
        if resultado.cost_usd is None:
            resultado.cost_usd = _custo_pela_tabela(
                modelo, resultado.prompt_tokens, resultado.completion_tokens
            )
            if resultado.cost_usd is not None:
                resultado.cost_source = "tabela_precos"

    def _buscar_custo(self, resultado: Resultado) -> None:
        """Rede de segurança: o custo real pelo endpoint /generation.

        Melhor do que preço de tabela x tokens, porque tokens de imagem têm
        precificação própria e variam por provedor. O registro só fica
        consultável alguns segundos depois da geração, então aqui vai uma
        tentativa só — se falhar, sobra a tabela de preços.
        """
        if not resultado.generation_id:
            return
        time.sleep(2.0)
        try:
            resposta = self.session.get(
                f"{self.base_url}/generation",
                params={"id": resultado.generation_id},
                timeout=30,
            )
            # Também guardado: este registro traz tokens nativos, desconto de
            # cache e tempos decompostos que o `Resultado` não extrai.
            dados = self._guardar_bruto(resposta, resultado, "generation").get("data") or {}
            if dados.get("total_cost") is not None:
                resultado.cost_usd = float(dados["total_cost"])
                resultado.cost_source = "generation_api"
                resultado.provider_upstream = (
                    dados.get("provider_name") or resultado.provider_upstream
                )
        except Exception:  # noqa: BLE001 - custo pela tabela é o plano B
            pass


class GeminiClient(_ClienteHTTP):
    """API nativa do Google.

    Ela devolve contagem de tokens mas **não** o valor cobrado: o custo vem da
    tabela declarada em `models.yaml` e fica marcado com
    `cost_source = "tabela_precos"`. Ao comparar custo entre provedores, essa
    diferença de procedência precisa ser dita.
    """

    api_provider = "gemini"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings, "gemini")
        self.base_url = self.cfg.get(
            "base_url", "https://generativelanguage.googleapis.com/v1beta"
        ).rstrip("/")
        self.rpm_padrao = float(self.cfg.get("rpm_limit") or 0)
        self.api_key = settings.gemini_api_key
        if not self.api_key:
            env = self.cfg.get("api_key_env", "GEMINI_API_KEY")
            raise RuntimeError(
                f"Chave do Gemini não encontrada. Defina a variável de ambiente {env}.\n"
                f"  PowerShell:  $env:{env} = '...'"
            )
        self.session.headers.update({"Content-Type": "application/json"})

    def _rpm(self, modelo: ModelSpec) -> float | None:
        return modelo.rpm_limit or self.rpm_padrao

    def _chamar(
        self, modelo: ModelSpec, prompt: PromptSpec, documento: Document, resultado: Resultado
    ) -> dict:
        mime, dados = _imagem_base64(documento.image_path)
        payload: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt.text},
                        {"inline_data": {"mime_type": mime, "data": dados}},
                    ],
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
            },
        }
        if modelo.reasoning_effort:
            # O Gemini expõe o mesmo controle como thinkingLevel; "none" desliga.
            payload["generationConfig"]["thinkingConfig"] = (
                {"thinkingBudget": 0}
                if modelo.reasoning_effort == "none"
                else {"thinkingLevel": modelo.reasoning_effort}
            )

        resposta = self.session.post(
            f"{self.base_url}/models/{modelo.remote_id}:generateContent",
            params={"key": self.api_key},
            data=json.dumps(payload),
            timeout=self.timeout,
        )
        corpo = self._guardar_bruto(resposta, resultado, "generateContent")
        if resposta.status_code >= 400:
            raise requests.HTTPError(
                f"HTTP {resposta.status_code}: {resposta.text[:500]}", response=resposta
            )
        return corpo

    def _preencher(self, corpo: dict, resultado: Resultado, modelo: ModelSpec) -> None:
        candidatos = corpo.get("candidates") or []
        if candidatos:
            resultado.finish_reason = candidatos[0].get("finishReason")
            partes = (candidatos[0].get("content") or {}).get("parts") or []
            resultado.text = "".join(p.get("text", "") for p in partes)
        elif (corpo.get("promptFeedback") or {}).get("blockReason"):
            resultado.finish_reason = corpo["promptFeedback"]["blockReason"]
        resultado.truncated = resultado.finish_reason in _MOTIVOS_DE_TRUNCAMENTO
        resultado.provider_upstream = "google"

        uso = corpo.get("usageMetadata") or {}
        visiveis = uso.get("candidatesTokenCount") or 0
        # Tokens de raciocínio são cobrados como saída: entram no total para que
        # o custo por página fique correto.
        pensamento = uso.get("thoughtsTokenCount") or 0
        resultado.prompt_tokens = uso.get("promptTokenCount")
        resultado.completion_tokens = (visiveis + pensamento) or None
        resultado.total_tokens = uso.get("totalTokenCount")
        resultado.cost_usd = _custo_pela_tabela(
            modelo, resultado.prompt_tokens, resultado.completion_tokens
        )
        resultado.cost_source = "tabela_precos" if resultado.cost_usd is not None else None

    def _motivo_da_resposta_vazia(self, resultado: Resultado) -> tuple[str, str]:
        recusou = resultado.finish_reason in {"SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT"}
        return (
            f"resposta sem texto (finishReason={resultado.finish_reason})",
            "recusa" if recusou else "vazio",
        )


class MinerUClient:
    """O MinerU, executado localmente, como se fosse mais um modelo.

    Não é um LLM: é um analisador de documentos (layout, OCR, tabelas) que roda
    na máquina, sem chave, sem custo por token e sem receber instrução em
    linguagem natural. Ainda assim responde à mesma pergunta que os LLMs
    respondem aqui, e é o concorrente natural deles nesta tarefa.

    Duas coisas mudam de sentido quando o "modelo" é local:

    * **Custo.** Não há preço por token: `cost_usd` é 0 e os campos de token
      ficam nulos DE PROPÓSITO. Não existe tokenizador comum entre o MinerU e
      os LLMs, e inventar uma contagem o faria entrar nas médias de custo por
      token como se fosse comparável. O custo real dele é tempo, e está em
      `latency_s`.

    * **Prompt.** Não existe. Por isso o MinerU é `prompt_agnostic` e roda uma
      única vez, sob o prompt sentinela `p0`.

    Os artefatos de cada chamada (o Markdown, o `content_list.json`, o
    `layout.pdf`, o log) ficam em `results/<run_id>/mineru/`: são eles que
    tornam o resultado auditável.
    """

    api_provider = "mineru"

    def __init__(self, settings: Settings) -> None:
        self.cfg = settings.section("mineru")
        self.timeout_s = float(self.cfg.get("timeout_seconds", 3600))
        self.run_dir = project_root() / "results" / "mineru_avulso"

    @property
    def executavel(self) -> Path:
        """Caminho do `mineru`, no venv dedicado.

        O MinerU exige Python 3.10-3.12; o venv separado é o que permite rodar
        os dois no mesmo projeto. O caminho é configuração (`mineru.venv`), não
        descoberta automática, para que a execução seja reproduzível noutra
        máquina.
        """
        base = Path(self.cfg.get("venv") or ".venv-mineru")
        if not base.is_absolute():
            base = project_root() / base
        for relativo in ("Scripts/mineru.exe", "bin/mineru"):
            if (base / relativo).exists():
                return base / relativo
        raise RuntimeError(
            f"Executável do MinerU não encontrado em {base}. "
            "Rode ferramentas/mineru/setup_mineru.ps1."
        )

    def _comando(self, modelo: ModelSpec, entrada: Path, saida: Path) -> list[str]:
        opcoes = modelo.backend_options
        cmd = [str(self.executavel), "-p", str(entrada), "-o", str(saida),
               "-b", str(opcoes["backend"])]
        # Repassadas só quando declaradas: o padrão do MinerU é o que o usuário
        # comum obtém, e mudá-lo em silêncio tornaria o resultado incomparável
        # com a documentação da ferramenta.
        for chave, flag in (("effort", "--effort"), ("method", "-m"), ("lang", "-l")):
            if opcoes.get(chave) is not None:
                cmd += [flag, str(opcoes[chave])]
        return cmd

    def _ambiente(self) -> dict[str, str]:
        env = dict(os.environ)
        # modelscope em vez de huggingface: o cache do HF usa symlinks, e
        # criá-los no Windows exige um privilégio que nem toda máquina concede
        # (WinError 1314). Não é preferência, é a fonte que funciona.
        env.setdefault("MINERU_MODEL_SOURCE", str(self.cfg.get("model_source", "modelscope")))
        return env

    def transcribe(
        self, modelo: ModelSpec, prompt: PromptSpec, documento: Document, repetition: int = 0
    ) -> Resultado:
        resultado = Resultado(
            model_id=modelo.id,
            prompt_id=prompt.id,
            doc_id=documento.doc_id,
            repetition=repetition,
            api_provider="mineru",
            provider_upstream="local",
            cost_usd=0.0,
            cost_source="local",
            attempts=1,
            timestamp_utc=_agora_utc(),
        )

        destino = self.run_dir / "mineru" / modelo.slug / f"{documento.doc_id}__r{repetition}"
        shutil.rmtree(destino, ignore_errors=True)
        entrada_dir, saida_dir = destino / "entrada", destino / "saida"
        entrada_dir.mkdir(parents=True, exist_ok=True)

        # A imagem é copiada com um nome previsível: o MinerU batiza a pasta de
        # saída com o nome do arquivo, e os nomes do dataset têm espaços e
        # sufixos de identificador que tornariam o caminho ilegível.
        entrada = entrada_dir / f"{documento.doc_id}{documento.image_path.suffix.lower()}"
        shutil.copy2(documento.image_path, entrada)

        inicio = time.perf_counter()
        try:
            comando = self._comando(modelo, entrada, saida_dir)
            processo = subprocess.run(  # noqa: S603 - comando montado aqui, sem shell
                comando,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=self._ambiente(),
                timeout=self.timeout_s,
                check=False,
            )
            log = (processo.stdout or "") + (processo.stderr or "")
        except subprocess.TimeoutExpired:
            resultado.latency_s = time.perf_counter() - inicio
            resultado.error_kind = "timeout"
            resultado.error = f"MinerU excedeu {self.timeout_s:.0f}s"
            return resultado
        except (OSError, RuntimeError) as exc:
            resultado.latency_s = time.perf_counter() - inicio
            resultado.error_kind = "excecao"
            resultado.error = f"não foi possível executar o MinerU: {exc}"
            return resultado

        # O equivalente local da resposta bruta: não há corpo HTTP, e o que
        # explica uma execução é a linha de comando exata mais o log. O log
        # inteiro fica em `mineru/.../mineru.log`; aqui vai o fim dele, que é
        # onde as mensagens de erro aparecem.
        resultado.respostas_brutas.append(
            {
                "etapa": "subprocess",
                "tentativa": 1,
                "comando": comando,
                "returncode": processo.returncode,
                "backend": modelo.backend_options.get("backend"),
                "artefatos": str(destino),
                "log_final": log[-4000:],
            }
        )

        resultado.latency_s = time.perf_counter() - inicio
        (destino / "mineru.log").write_text(log, encoding="utf-8", errors="replace")

        if processo.returncode != 0:
            resultado.error_kind = "api_error"
            resultado.error = f"MinerU terminou com código {processo.returncode}: {log.strip()[-400:]}"
            return resultado

        markdown = self._localizar_markdown(saida_dir)
        if markdown is None or not markdown.read_text(encoding="utf-8", errors="replace").strip():
            resultado.error_kind = "vazio"
            resultado.error = f"MinerU não gerou Markdown com conteúdo em {saida_dir}"
            return resultado

        resultado.ok = True
        resultado.text = markdown.read_text(encoding="utf-8", errors="replace")
        resultado.finish_reason = "stop"
        resultado.generation_id = f"mineru-{modelo.backend_options.get('backend')}-{destino.name}"
        return resultado

    @staticmethod
    def _localizar_markdown(saida_dir: Path) -> Path | None:
        """Acha o .md gerado, sem depender do nome da subpasta do backend.

        A subpasta muda com o backend (`auto`, `vlm`, `hybrid_auto`) e já mudou
        entre versões do MinerU. Buscar por glob e ficar com o maior arquivo
        sobrevive a isso; casar o nome exato não sobreviveria.
        """
        candidatos = [p for p in saida_dir.rglob("*.md") if p.stat().st_size > 0]
        return max(candidatos, key=lambda p: p.stat().st_size) if candidatos else None


class Roteador:
    """Despacha cada modelo para o provedor declarado em `models.yaml`.

    Os clientes são construídos sob demanda: uma execução só com MinerU não
    exige chave de API nenhuma.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._clientes: dict[str, Any] = {}

    def _cliente(self, provider: str):  # noqa: ANN201
        if provider not in self._clientes:
            classe = {
                "openrouter": OpenRouterClient,
                "gemini": GeminiClient,
                "mineru": MinerUClient,
            }[provider]
            self._clientes[provider] = classe(self.settings)
        return self._clientes[provider]

    def set_run_dir(self, run_dir: Path) -> None:
        """Diz ao MinerU onde guardar os artefatos desta execução."""
        self.run_dir = Path(run_dir)
        for cliente in self._clientes.values():
            if isinstance(cliente, MinerUClient):
                cliente.run_dir = self.run_dir

    def transcribe(
        self, modelo: ModelSpec, prompt: PromptSpec, documento: Document, repetition: int = 0
    ) -> Resultado:
        cliente = self._cliente(modelo.provider)
        if isinstance(cliente, MinerUClient) and getattr(self, "run_dir", None):
            cliente.run_dir = self.run_dir
        return cliente.transcribe(modelo, prompt, documento, repetition)
