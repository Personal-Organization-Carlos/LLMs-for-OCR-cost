"""Interface de linha de comando.

    python -m bench check    confere dataset, modelos, prompts e chaves
    python -m bench run      faz as chamadas e grava as respostas
    python -m bench score    calcula as métricas de uma execução
    python -m bench graficos gera as figuras sobre todas as execuções
    python -m bench all      run + score + gráficos
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Config, load_all, load_settings, project_root
from .dataset import discover_documents, select_documents


def _config(args) -> Config:  # noqa: ANN001
    config = load_all()
    for atributo, campo, arquivo in (
        ("models", "id", "models.yaml"),
        ("prompts", "id", "prompts.yaml"),
    ):
        escolhidos = getattr(args, atributo, None)
        if not escolhidos:
            continue
        itens = [i for i in getattr(config, atributo) if getattr(i, campo) in set(escolhidos)]
        faltando = set(escolhidos) - {getattr(i, campo) for i in itens}
        if faltando:
            raise SystemExit(f"Não encontrado(s) em {arquivo}: {sorted(faltando)}")
        setattr(config, atributo, itens)
    return config


def _documentos(settings, args):  # noqa: ANN001, ANN201
    """Precedência: `--docs` > `--all-docs` > `dataset.active` > todos."""
    documentos = discover_documents(settings)
    if getattr(args, "docs", None):
        return select_documents(documentos, args.docs)
    if getattr(args, "all_docs", False):
        return documentos
    ativos = settings.active_documents
    if ativos:
        escolhidos = select_documents(documentos, ativos)
        print(
            f"[dataset] usando {len(escolhidos)} de {len(documentos)} documento(s) "
            f"(dataset.active em settings.yaml; --all-docs usa todos)"
        )
        return escolhidos
    return documentos


def _run_dir(settings, run_id: str | None) -> Path:  # noqa: ANN001
    """A execução pedida, ou a mais recente.

    Uma execução é uma pasta com `chamadas.jsonl` dentro — não qualquer pasta
    em `results/`. A distinção importa porque `results/graficos/` também mora
    ali, e sem ela seria escolhida como "a mais recente" por ordem alfabética.
    """
    from .runner import CHAMADAS

    raiz = settings.results_dir
    if run_id:
        caminho = raiz / run_id
        if not caminho.is_dir():
            raise SystemExit(f"Execução não encontrada: {caminho}")
        return caminho
    candidatos = (
        sorted(p for p in raiz.iterdir() if p.is_dir() and (p / CHAMADAS).exists())
        if raiz.is_dir()
        else []
    )
    if not candidatos:
        raise SystemExit(f"Nenhuma execução em {raiz}. Rode `run` antes.")
    return candidatos[-1]


# --------------------------------------------------------------------------

def cmd_check(args) -> int:  # noqa: ANN001
    settings = load_settings()
    config = _config(args)
    ativos = set(settings.active_documents)

    print(f"Raiz do projeto : {project_root()}")
    print(f"Dataset         : {settings.dataset_root}")

    documentos = discover_documents(settings)
    print(f"\nDocumentos ({len(documentos)}):")
    for doc in documentos:
        marca = "[ativo]" if not ativos or doc.doc_id in ativos else "[  -  ]"
        print(f"  {marca} {doc.doc_id:22s} imagem={doc.image_path.name}")

    print(f"\nModelos habilitados ({len(config.models)}):")
    for modelo in config.models:
        print(f"  - {modelo.id:42s} {modelo.empresa:12s} {modelo.tipo:8s} via {modelo.provider}")
    print(f"\nPrompts habilitados : {[p.id for p in config.prompts]}")

    provedores = {m.provider for m in config.models}
    print("\nChaves de API:")
    for rotulo, chave, provedor in (
        ("OPENROUTER_API_KEY", settings.api_key, "openrouter"),
        ("GEMINI_API_KEY", settings.gemini_api_key, "gemini"),
    ):
        necessaria = "  <- necessária" if provedor in provedores else ""
        print(f"  {rotulo:19s}: {'definida' if chave else 'AUSENTE'}{necessaria}")

    if "mineru" in provedores:
        from .clients import MinerUClient

        try:
            print(f"\nMinerU: {MinerUClient(settings).executavel}")
        except RuntimeError as exc:
            print(f"\nMinerU: {exc}")

    return _conferir_hospedagem(config)


def _conferir_hospedagem(config: Config) -> int:  # noqa: ANN001
    """Confere que o endpoint fixado em `endpoint_tag` existe para o modelo.

    Vale a consulta de rede: com `permitir_fallback_de_hospedagem: false`, uma
    tag errada faz TODAS as chamadas daquele modelo falharem, e a mensagem do
    OpenRouter não diz que o problema é o nome. Mostra também o preço da faixa,
    que é a segunda metade do que a tag fixa e a que passa despercebida com mais
    facilidade. O catálogo de endpoints é público e não consome cota.
    """
    fixados = [m for m in config.models if m.endpoint_tag]
    if not fixados:
        return 0

    import requests

    print("\nEndpoint fixado (casa + faixa de serviço):")
    problemas = 0
    for modelo in fixados:
        try:
            resposta = requests.get(
                f"https://openrouter.ai/api/v1/models/{modelo.id}/endpoints", timeout=30
            )
            endpoints = (resposta.json().get("data") or {}).get("endpoints") or []
        except Exception as exc:  # noqa: BLE001 - sem rede, isto é aviso, não erro
            print(f"  {modelo.id:36s} {modelo.endpoint_tag:20s} (não verificado: {exc})")
            continue

        escolhido = next((e for e in endpoints if e.get("tag") == modelo.endpoint_tag), None)
        if escolhido:
            p = escolhido.get("pricing") or {}
            preco = (f"{float(p.get('prompt', 0)) * 1e6:.3f}/"
                     f"{float(p.get('completion', 0)) * 1e6:.3f} US$/Mtok")
            print(f"  {modelo.id:36s} {modelo.endpoint_tag:20s} ok · "
                  f"{escolhido.get('provider_name')} · {preco} "
                  f"({len(endpoints)} endpoint(s) no total)")
        else:
            problemas += 1
            print(f"  {modelo.id:36s} {modelo.endpoint_tag:20s} NÃO EXISTE · "
                  f"disponíveis: {sorted(e.get('tag') for e in endpoints)}")
    return 1 if problemas else 0


def cmd_run(args) -> int:  # noqa: ANN001
    from .clients import Roteador
    from .runner import Escritor, build_tasks, chaves_concluidas, executar, new_run_id

    settings = load_settings()
    config = _config(args)
    documentos = _documentos(settings, args)
    if not config.models:
        raise SystemExit("Nenhum modelo selecionado (veja config/models.yaml).")
    if not config.prompts:
        raise SystemExit("Nenhum prompt habilitado (veja config/prompts.yaml).")

    execucao = settings.section("execution")
    repeticoes = args.repetitions or int(execucao.get("repetitions", 1))
    tasks = build_tasks(config.models, documentos, config.prompts, repeticoes)

    if args.resume:
        run_dir = _run_dir(settings, args.resume)
        concluidas = chaves_concluidas(run_dir)
        tasks = [t for t in tasks if t.key not in concluidas]
        print(f"[run] retomando {run_dir.name}: {len(concluidas)} pronta(s), {len(tasks)} restante(s)")
    else:
        run_dir = settings.results_dir / new_run_id()

    print(
        f"[run] {len(config.models)} modelo(s) x {len(documentos)} documento(s) x "
        f"{repeticoes} repetição(ões) = {len(tasks)} chamada(s)"
    )
    if args.dry_run:
        for task in tasks:
            print(f"  {task.key}")
        print(f"\n[dry-run] nada foi executado. Pasta prevista: {run_dir}")
        return 0
    if not tasks:
        print("[run] nada a fazer.")
        return 0

    # O cliente é construído antes de qualquer pasta ser criada: uma chave
    # ausente não deve deixar execuções vazias para trás.
    cliente = Roteador(settings)
    run_dir.mkdir(parents=True, exist_ok=True)
    cliente.set_run_dir(run_dir)

    contador = {"n": 0, "ok": 0, "custo": 0.0}

    def ao_terminar(task, resultado) -> None:  # noqa: ANN001
        contador["n"] += 1
        contador["ok"] += 1 if resultado.ok else 0
        contador["custo"] += resultado.cost_usd or 0.0
        estado = "TRUNC" if resultado.truncated else ("ok   " if resultado.ok else "FALHA")
        detalhe = (
            f"{resultado.latency_s or 0:6.1f}s  {resultado.completion_tokens or 0:6d} tok"
            if resultado.ok
            else f"({resultado.error_kind}: {str(resultado.error)[:60]})"
        )
        print(f"  [{contador['n']:4d}/{len(tasks)}] {estado} {task.key:68s} {detalhe}")

    executar(
        cliente,
        tasks,
        Escritor(run_dir),
        max_parallel_models=int(execucao.get("max_parallel_models", 3)),
        sleep_between_calls=float(execucao.get("sleep_between_calls_seconds", 1.0)),
        ao_terminar=ao_terminar,
    )
    print(
        f"\n[run] {contador['ok']}/{contador['n']} com sucesso · "
        f"custo US$ {contador['custo']:.4f} · {run_dir}"
    )
    args._run_dir = run_dir
    return 0


def cmd_score(args) -> int:  # noqa: ANN001
    from .score import score_run

    settings = load_settings()
    score_run(_run_dir(settings, args.run_id), settings)
    return 0


def cmd_graficos(args) -> int:  # noqa: ANN001
    from .graficos import build_graficos

    build_graficos(load_settings(), run_ids=args.run_ids)
    return 0


def cmd_all(args) -> int:  # noqa: ANN001
    from .graficos import build_graficos
    from .score import score_run

    codigo = cmd_run(args)
    if codigo != 0 or args.dry_run:
        return codigo
    settings = load_settings()
    run_dir = getattr(args, "_run_dir", None) or _run_dir(settings, None)
    score_run(run_dir, settings)
    # Os gráficos são de TODAS as execuções, não só da que acabou de rodar.
    build_graficos(settings)
    return 0


# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench",
        description="Benchmark de LLMs para transcrição de documentos históricos brasileiros.",
    )
    sub = parser.add_subparsers(dest="comando", required=True)

    def selecao(p: argparse.ArgumentParser) -> None:
        p.add_argument("--models", nargs="*", help="Restringe a estes IDs de modelo.")
        p.add_argument("--docs", nargs="*", help="Restringe a estes doc_ids.")
        p.add_argument("--all-docs", action="store_true", help="Usa todos os documentos.")
        p.add_argument("--prompts", nargs="*", help="Restringe a estes prompt_ids.")

    def execucao(p: argparse.ArgumentParser) -> None:
        selecao(p)
        p.add_argument("--repetitions", type=int)
        p.add_argument("--resume", metavar="RUN_ID", help="Retoma uma execução interrompida.")
        p.add_argument("--dry-run", action="store_true", help="Lista as chamadas sem executá-las.")

    p_check = sub.add_parser("check", help="Confere dataset, modelos, prompts e chaves.")
    selecao(p_check)
    p_check.set_defaults(func=cmd_check)

    p_run = sub.add_parser("run", help="Executa as chamadas aos modelos.")
    execucao(p_run)
    p_run.set_defaults(func=cmd_run)

    p_score = sub.add_parser("score", help="Calcula as métricas de uma execução.")
    p_score.add_argument("--run-id", help="Padrão: a execução mais recente.")
    p_score.set_defaults(func=cmd_score)

    p_graf = sub.add_parser(
        "graficos", help="Gera as figuras sobre todas as execuções."
    )
    p_graf.add_argument(
        "--run-ids", nargs="*",
        help="Restringe a estas execuções. Padrão: todas.",
    )
    p_graf.set_defaults(func=cmd_graficos)

    p_all = sub.add_parser("all", help="run + score + gráficos.")
    execucao(p_all)
    p_all.set_defaults(func=cmd_all)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrompido. Use `run --resume <RUN_ID>` para retomar.", file=sys.stderr)
        return 130
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        # Erro de configuração ou de ambiente é do usuário, não bug: mensagem
        # limpa em vez de traceback.
        print(f"\nErro: {exc}", file=sys.stderr)
        return 1
