"""Descoberta do conjunto de testes.

Cada documento é uma pasta em `Dataset/` com exatamente dois arquivos que
importam: a imagem (entrada do experimento) e um `.html` (a referência,
transcrita à mão). Os demais arquivos da pasta são ignorados.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from .config import Settings


@dataclass
class Document:
    doc_id: str
    image_path: Path
    reference_path: Path

    @cached_property
    def reference_html(self) -> str:
        """A referência, lida uma vez por processo e não uma vez por chamada."""
        return self.reference_path.read_text(encoding="utf-8", errors="replace")


def discover_documents(settings: Settings) -> list[Document]:
    """Varre a pasta do dataset e devolve os documentos encontrados, ordenados.

    Duas imagens ou dois HTML na mesma pasta é erro, não escolha silenciosa:
    qual dos dois foi usado mudaria o resultado sem aparecer em lugar nenhum.
    """
    root = settings.dataset_root
    if not root.is_dir():
        raise FileNotFoundError(f"Pasta do dataset não encontrada: {root}")

    exts = {e.lower() for e in settings.section("dataset").get("image_extensions", [".jpg"])}

    documentos: list[Document] = []
    for pasta in sorted(root.iterdir()):
        if not pasta.is_dir() or pasta.name.startswith("."):
            continue
        imagens = sorted(p for p in pasta.iterdir() if p.suffix.lower() in exts)
        referencias = sorted(p for p in pasta.iterdir() if p.suffix.lower() == ".html")
        if not imagens or not referencias:
            continue
        for achados, oque in ((imagens, "imagem"), (referencias, "HTML de referência")):
            if len(achados) > 1:
                raise ValueError(
                    f"{pasta}: esperava um único arquivo de {oque}, encontrei "
                    f"{[p.name for p in achados]}"
                )
        documentos.append(
            Document(
                doc_id=pasta.name,
                image_path=imagens[0],
                reference_path=referencias[0],
            )
        )

    if not documentos:
        raise FileNotFoundError(
            f"Nenhum documento válido em {root}. Cada documento é uma pasta com "
            "uma imagem e um .html de referência."
        )
    return documentos


def select_documents(documentos: list[Document], apenas: list[str] | None) -> list[Document]:
    """Filtra por doc_id, preservando a ordem pedida."""
    if not apenas:
        return documentos
    por_id = {d.doc_id: d for d in documentos}
    desconhecidos = [d for d in apenas if d not in por_id]
    if desconhecidos:
        raise ValueError(
            f"Documento(s) desconhecido(s): {desconhecidos}. Disponíveis: {sorted(por_id)}"
        )
    return [por_id[d] for d in apenas]
