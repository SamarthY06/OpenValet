"""The model downloader against a local HTTP server: what counts as a bad source and what is
left on disk after one (no real model is fetched)."""

from __future__ import annotations

import hashlib
import io
import re
import tarfile
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from macd.speech import models
from macd.speech.models import ModelSpec, Source

NAME = "tiny-model"
FILES = {"tokens.txt": b"a 0\nb 1\n" * 400, "encoder.onnx": bytes(range(256)) * 900}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tarball(files: dict[str, bytes], top: str = NAME) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:bz2") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(f"{top}/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


@dataclass
class Page:
    body: bytes
    send: int | None = None  # bytes really sent before the connection is closed (None = all)
    length: bool = True  # whether Content-Length is sent
    status: int = 200


class Site:
    """A small web server; `pages` maps a path to what it answers, `hits` lists (path, Range)."""

    def __init__(self) -> None:
        self.pages: dict[str, Page] = {}
        self.hits: list[tuple[str, str | None]] = []
        site = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                rng = self.headers.get("Range")
                site.hits.append((self.path, rng))
                page = site.pages.get(self.path)
                if page is None or page.status != 200:
                    self.send_error(page.status if page else 404)
                    return
                start = int(rng.split("=")[1].rstrip("-")) if rng else 0
                if start >= len(page.body) and rng:
                    self.send_error(416)
                    return
                body = page.body[start:]
                self.send_response(206 if rng else 200)
                if page.length:
                    self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body if page.send is None else body[:page.send])

            def log_message(self, *a: object) -> None:
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def site():
    s = Site()
    yield s
    s.close()


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(models.config, "MODELS_DIR", tmp_path)
    return tmp_path


def use(monkeypatch: pytest.MonkeyPatch, spec: ModelSpec) -> None:
    monkeypatch.setattr(models, "SPECS", (spec,))
    monkeypatch.setattr(models, "SPEC_BY_KEY", {spec.key: spec})


def mirror(site: Site, files: dict[str, bytes] = FILES, at: str = "/hf") -> Source:
    for name, data in files.items():
        site.pages[f"{at}/{name}"] = Page(data)
    return Source(site.base + at, tuple(files), tuple(sha(d) for d in files.values()))


def dir_spec(*sources: Source) -> ModelSpec:
    return ModelSpec("tiny", NAME, sources, required=("tokens.txt", "encoder*.onnx"))


def leftovers(root: Path) -> list[str]:
    return sorted(p.name for p in root.rglob("*") if p.name.endswith((".part", ".tar.bz2")) or p.name.startswith(".extract-"))


def installed(root: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in (root / NAME).iterdir()}


def test_a_truncated_archive_behind_a_200_fails_its_checksum_and_the_mirror_is_used(site, root, monkeypatch):
    good = tarball(FILES)
    site.pages["/gh/tiny.tar.bz2"] = Page(good, send=len(good) // 2, length=False)  # 200, no length, cut short
    use(monkeypatch, dir_spec(Source(site.base + "/gh/tiny.tar.bz2", sha256=(sha(good),)), mirror(site)))
    assert models.download_models(keys=["tiny"]) == ["tiny"]
    assert installed(root) == FILES and leftovers(root) == []
    assert [h[0] for h in site.hits] == ["/gh/tiny.tar.bz2", "/hf/tokens.txt", "/hf/encoder.onnx"]


def test_an_archive_download_that_dies_leaves_nothing_behind_once_the_mirror_installed(site, root, monkeypatch):
    good = tarball(FILES)
    site.pages["/gh/tiny.tar.bz2"] = Page(good, send=len(good) // 2)  # the length is announced, half arrives
    gh = Source(site.base + "/gh/tiny.tar.bz2", sha256=(sha(good),))
    use(monkeypatch, dir_spec(gh))
    with pytest.raises(OSError, match="incomplete download"):
        models.download_models(keys=["tiny"])
    assert leftovers(root) == [models._part(root / f"{NAME}.tar.bz2", gh.url).name]  # kept: the next run resumes it
    use(monkeypatch, dir_spec(gh, mirror(site)))
    assert models.download_models(keys=["tiny"]) == ["tiny"]
    assert site.hits[1] == ("/gh/tiny.tar.bz2", f"bytes={len(good) // 2}-")
    assert installed(root) == FILES and leftovers(root) == []


def test_an_archive_that_does_not_extract_is_one_more_reason_to_try_the_mirror(site, root, monkeypatch):
    good = tarball(FILES)
    site.pages["/gh/tiny.tar.bz2"] = Page(good[: len(good) // 2])  # whole as far as HTTP goes, no digest published
    use(monkeypatch, dir_spec(Source(site.base + "/gh/tiny.tar.bz2"), mirror(site)))
    assert models.download_models(keys=["tiny"]) == ["tiny"]
    assert installed(root) == FILES and leftovers(root) == []


def test_an_archive_without_tokens_txt_never_replaces_the_installed_copy(site, root, monkeypatch):
    bad = tarball({"encoder.onnx": FILES["encoder.onnx"]})
    site.pages["/gh/tiny.tar.bz2"] = Page(bad)
    use(monkeypatch, dir_spec(Source(site.base + "/gh/tiny.tar.bz2", sha256=(sha(bad),))))
    (root / NAME).mkdir()
    (root / NAME / "encoder.onnx").write_bytes(b"the copy from before")
    with pytest.raises(OSError, match=r"lacks tokens\.txt"):
        models.download_models(keys=["tiny"])
    assert installed(root) == {"encoder.onnx": b"the copy from before"} and leftovers(root) == []
    use(monkeypatch, dir_spec(Source(site.base + "/gh/tiny.tar.bz2", sha256=(sha(bad),)), mirror(site)))
    assert models.download_models(keys=["tiny"]) == ["tiny"]  # with a mirror: its files, the stale one replaced
    assert installed(root) == FILES


def test_a_good_archive_is_verified_and_installed_without_touching_the_mirror(site, root, monkeypatch):
    good = tarball(FILES)
    site.pages["/gh/tiny.tar.bz2"] = Page(good)
    use(monkeypatch, dir_spec(Source(site.base + "/gh/tiny.tar.bz2", sha256=(sha(good),)), mirror(site)))
    assert models.download_models(keys=["tiny"]) == ["tiny"]
    assert installed(root) == FILES and leftovers(root) == [] and len(site.hits) == 1


def test_a_file_that_is_not_the_published_one_is_never_installed(site, root, monkeypatch):
    real = b"model" * 5000
    site.pages["/a/vad.onnx"] = Page(b"<html>sign in to continue</html>")
    site.pages["/b/vad.onnx"] = Page(real[:-7])
    spec = ModelSpec("vad", "vad.onnx", (Source(site.base + "/a/vad.onnx", sha256=(sha(real),)),
                                         Source(site.base + "/b/vad.onnx", sha256=(sha(real),))))
    use(monkeypatch, spec)
    with pytest.raises(OSError, match=r"(?s)a/vad\.onnx: checksum mismatch.*b/vad\.onnx: checksum mismatch"):
        models.download_models(keys=["vad"])
    assert list(root.iterdir()) == []
    site.pages["/b/vad.onnx"] = Page(real)
    assert models.download_models(keys=["vad"]) == ["vad"] and (root / "vad.onnx").read_bytes() == real


def test_a_resume_never_continues_another_sources_bytes(site, root, monkeypatch):
    first, second = b"A" * 60_000, b"B" * 50_000  # the mirrors of one model are not always the same file
    site.pages["/a/voice.onnx"] = Page(first, send=20_000)  # dies a third of the way in
    site.pages["/b/voice.onnx"] = Page(second)
    spec = ModelSpec("voice", "voice.onnx", (Source(site.base + "/a/voice.onnx", sha256=(sha(first),)),
                                             Source(site.base + "/b/voice.onnx", sha256=(sha(second),))))
    use(monkeypatch, spec)
    assert models.download_models(keys=["voice"]) == ["voice"]
    assert (root / "voice.onnx").read_bytes() == second
    assert site.hits == [("/a/voice.onnx", None), ("/b/voice.onnx", None)]  # the mirror was asked for the whole file
    assert leftovers(root) == []  # and the first source's partial file went with the install


def test_a_resume_continues_its_own_source_and_a_bad_partial_file_is_fetched_again(site, root, monkeypatch):
    body = bytes(range(256)) * 300
    site.pages["/a/turn.onnx"] = Page(body, send=30_000)
    src = Source(site.base + "/a/turn.onnx", sha256=(sha(body),))
    use(monkeypatch, ModelSpec("turn", "turn.onnx", (src,)))
    with pytest.raises(OSError, match="incomplete download"):
        models.download_models(keys=["turn"])
    part = models._part(root / "turn.onnx", src.url)
    assert part.stat().st_size == 30_000 and not (root / "turn.onnx").exists()
    site.pages["/a/turn.onnx"] = Page(body)
    assert models.download_models(keys=["turn"]) == ["turn"]
    assert site.hits[-1] == ("/a/turn.onnx", "bytes=30000-") and (root / "turn.onnx").read_bytes() == body

    (root / "turn.onnx").unlink()
    part.write_bytes(b"\0" * 1000)  # a partial file that is not the start of this file
    assert models.download_models(keys=["turn"]) == ["turn"]
    assert site.hits[-2:] == [("/a/turn.onnx", "bytes=1000-"), ("/a/turn.onnx", None)]
    assert (root / "turn.onnx").read_bytes() == body and leftovers(root) == []

    (root / "turn.onnx").unlink()
    part.write_bytes(body)  # already whole: the server answers 416 and the file is still verified
    assert models.download_models(keys=["turn"]) == ["turn"]
    assert (root / "turn.onnx").read_bytes() == body and leftovers(root) == []


def test_files_already_there_are_kept_only_if_they_match(site, root, monkeypatch):
    use(monkeypatch, dir_spec(mirror(site)))
    (root / NAME).mkdir()
    (root / NAME / "encoder.onnx").write_bytes(FILES["encoder.onnx"])
    assert models.download_models(keys=["tiny"]) == ["tiny"]
    assert [h[0] for h in site.hits] == ["/hf/tokens.txt"]  # the matching file was not fetched again
    (root / NAME / "tokens.txt").unlink()
    (root / NAME / "encoder.onnx").write_bytes(b"cut short")
    assert models.download_models(keys=["tiny"]) == ["tiny"]
    assert installed(root) == FILES


def test_every_real_source_has_a_digest_and_every_mirror_a_pinned_commit():
    for spec in models.SPECS:
        for src in spec.sources:
            assert len(src.sha256) == (len(src.files) or 1), src.url
            assert all(re.fullmatch(r"[0-9a-f]{64}", d) for d in src.sha256), src.url
            if src.url.startswith("https://huggingface.co/"):
                assert re.search(r"/resolve/[0-9a-f]{40}(/|$)", src.url), src.url  # never a moving branch
