"""Model locations and a resumable, mirrored, safe downloader.

Each model has an ordered list of sources (GitHub release first, HuggingFace mirrors
next). A source is tried until one works: a timeout, an HTTP error, a file that does not
match its published SHA-256, an archive that does not extract or lacks a required file
all fall through to the next. Downloads go to a ``.part`` file of their own per source,
with HTTP Range resume, and are renamed into place only once verified; archives are
extracted with path checks.

``python -m macd.speech.models`` downloads whatever is missing into config.MODELS_DIR.
"""

from __future__ import annotations

import glob
import hashlib
import logging
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from macd import config

log = logging.getLogger(__name__)

KWS_NAME = "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
PARAKEET_NAME = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
POCKET_NAME = "kyutai-pocket-tts-int8"
TURN_NAME = "smart-turn-v3.2-cpu.onnx"
TURN_REV = "f766f81d3cfdf7737ac64aad813d91bbfd56bf93"  # pinned pipecat-ai/smart-turn-v3 commit
CONNECT_TIMEOUT_S = 10.0
_GH = "https://github.com"
_HF = "https://huggingface.co"


@dataclass(frozen=True)
class Source:
    """Where to get a model. ``files`` empty = a single file (or a .tar.bz2 archive for
    directory models) at ``url``; otherwise ``url`` is a base and each file is fetched from
    ``url/<file>``. Every source is saved under the spec's canonical ``target`` name.
    ``sha256`` is the published digest of the file or archive at ``url``, or one per name in
    ``files``; a download that does not match is thrown away."""

    url: str
    files: tuple[str, ...] = ()
    sha256: tuple[str, ...] = ()

    def digest(self, name: str = "") -> str:
        if not self.sha256:
            return ""
        return self.sha256[self.files.index(name)] if name else self.sha256[0]

    @property
    def archive(self) -> bool:
        return not self.files and self.url.endswith((".tar.bz2", ".tar.gz", ".tgz"))


@dataclass(frozen=True)
class ModelSpec:
    key: str
    target: str  # file or directory name under MODELS_DIR
    sources: tuple[Source, ...]
    required: tuple[str, ...] = ()  # glob patterns that must exist inside a directory target
    optional: bool = False  # not needed by default (only when a feature using it is enabled)

    @property
    def is_dir(self) -> bool:
        return bool(self.required)


def _hf(repo: str, rev: str, files: dict[str, str]) -> Source:
    """Files of a HuggingFace repo at one commit, each with its SHA-256."""
    return Source(f"{_HF}/{repo}/resolve/{rev}", tuple(files), tuple(files.values()))


_KWS_FILES = {
    "tokens.txt": "fd2ded4050a55d2b1578870ba8697d02371980217806b7558bd0a5cc60f3ba53",
    "bpe.model": "c8a2a0129c4ab8e463164c142f82d25649661b122c8cd0b7aab5c9e80b90ad24",
    "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx": "1e721676515bcd42a186979733981213c66c80db680e1cc582dfedf3be76e678",
    "decoder-epoch-12-avg-2-chunk-16-left-64.onnx": "f61ebd3eed3773a44d088d53dfae92dbb6aec4839f4dcaee2d402414741663a3",
    "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx": "eae9da0c7e1e6c6a3f4cc42d167899c388f6c6701b94cb96320e4f55df79624c",
}
_PARAKEET_FILES = {
    "encoder.int8.onnx": "92a10bd9c23936ea0566c29f963bc57c2a5df4261b6a8921c39abea66557ca8b",
    "decoder.int8.onnx": "4d097bae3bac5c96018d5293ba39f1b657296102b142affda40378f38b020f41",
    "joiner.int8.onnx": "77a29f3e5ac7d37074e0af465391b80f314015d5b97f0f90c6c7767f1d56f428",
    "tokens.txt": "ec182b70dd42113aff6c5372c75cac58c952443eb22322f57bbd7f53977d497d",
}
_POCKET_FILES = {
    "lm_main.int8.onnx": "bfc0c7e7e3d72864fa3bb2ee499f62f21ddc1474b885f5f3ca570f8be73e787e",
    "lm_flow.int8.onnx": "8d627d235c44a597da908e1085ebe241cbbe358964c502c5a5063d18851a5529",
    "encoder.onnx": "e8f2f6d301ffb96e398b138a7dc6d3038622d236044636b73d920bab85890260",
    "decoder.int8.onnx": "12b0857402d31aead94df19d6783b4350d1f740e811f3a3202c70ad89ae11eea",
    "text_conditioner.onnx": "0b84e837d7bfaf2c896627b03e3f080320309f37f4fc7df7698c644f7ba5e6b1",
    "vocab.json": "6fb646346cf931016f70c4921aab0900ce7a304b893cb02135c74e294abfea01",
    "token_scores.json": "5be2f278caf9b9800741f0fd82bff677f4943ec764c356f907213434b622d958",
    "LICENSE": "fe7b4ce83b8381cc5b216bbb4af73c570688d1b819c73bbaed8ca401f4677cd6",
}

# GitHub digests are the ones the sherpa-onnx release tags publish in checksum.txt (the kokoro-onnx
# release publishes none: those two are of the release files themselves); HuggingFace ones are the
# repo's own (LFS) digests at the pinned commit.
SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        "kws",
        KWS_NAME,
        (
            Source(f"{_GH}/k2-fsa/sherpa-onnx/releases/download/kws-models/{KWS_NAME}.tar.bz2",
                   sha256=("f170013b4716e41b62b9bfd809687c207cef798ef9bc6534d524e17af9b6561a",)),
            _hf("mobilebytesensei/sherpa-onnx-kws-zipformer-gigaspeech-3.3M",
                "e8ca74a794ce66e38a100e7c60c0214189ff4100", _KWS_FILES),
        ),
        required=("tokens.txt", "bpe.model", "encoder*.onnx", "decoder*.onnx", "joiner*.onnx"),
    ),
    ModelSpec(
        "vad",
        "silero_vad.onnx",
        (
            Source(f"{_GH}/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx",
                   sha256=("9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6",)),
            Source(f"{_HF}/lquint/silero-vad-v4-onnx/resolve/559ac3a7d56f309c04a601ff0ad7f0bcb2b56ad0/silero_vad.onnx",
                   sha256=("a35ebf52fd3ce5f1469b2a36158dba761bc47b973ea3382b3186ca15b1f5af28",)),
            Source(f"{_HF}/deepghs/silero-vad-onnx/resolve/193243f7d961b15e6de789d3f90cb0ee867e7b62/silero_vad.onnx",
                   sha256=("2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f",)),
        ),
    ),
    ModelSpec(
        "kokoro",
        "kokoro-v1.0.int8.onnx",
        (
            Source(f"{_GH}/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx",
                   sha256=("6e742170d309016e5891a994e1ce1559c702a2ccd0075e67ef7157974f6406cb",)),
            # No mirror of the exact int8 file exists; this is an 8-bit quantisation of the same v1.0 export.
            Source(f"{_HF}/Arxell/kokoro-v1.0.onnx/resolve/1919b3738f2ba7692d18a6b5784c26acab5714f2/model_quantized.onnx",
                   sha256=("fbae9257e1e05ffc727e951ef9b9c98418e6d79f1c9b6b13bd59f5c9028a1478",)),
        ),
    ),
    ModelSpec(
        "voices",
        "voices-v1.0.bin",
        (
            Source(f"{_GH}/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
                   sha256=("bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",)),
            Source(f"{_HF}/fastrtc/kokoro-onnx/resolve/8d07950c9b6c87ce6809e9bba7bd494336217c2a/voices-v1.0.bin",
                   sha256=("bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",)),
        ),
    ),
    ModelSpec(
        "parakeet",
        PARAKEET_NAME,
        (
            Source(f"{_GH}/k2-fsa/sherpa-onnx/releases/download/asr-models/{PARAKEET_NAME}.tar.bz2",
                   sha256=("6fc4426b54aa549a0019b8a75b8dea59164e71790a37ac56a045ad57f915f714",)),
            _hf(f"csukuangfj/{PARAKEET_NAME}", "84b478d541cdbf0c296a7ab5f1e20a6f8f998114", _PARAKEET_FILES),
        ),
        required=tuple(_PARAKEET_FILES),
    ),
    ModelSpec(  # end-of-turn detection (Pipecat Smart Turn v3.2, BSD-2-Clause); listener falls back to silence without it
        "turn",
        TURN_NAME,
        (Source(f"{_HF}/pipecat-ai/smart-turn-v3/resolve/{TURN_REV}/{TURN_NAME}",
                sha256=("2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f",)),),
    ),
    ModelSpec(  # voice cloning (tts.engine = "clone"), opt-in; weights are CC-BY-4.0 (see LICENSE)
        "pocket",
        POCKET_NAME,
        # HF first: the GitHub tarball extracts to sherpa-onnx-pocket-tts-int8-2026-01-26/, not POCKET_NAME.
        (_hf("csukuangfj2/sherpa-onnx-pocket-tts-int8-2026-01-26", "e715955cf50d18d919d37231513c0e914b83661a",
             _POCKET_FILES),),
        required=tuple(_POCKET_FILES)[:-1],
        optional=True,
    ),
)
SPEC_BY_KEY = {s.key: s for s in SPECS}


def models_dir() -> Path:
    return config.MODELS_DIR


def path(key: str) -> Path:
    """Installed location of a model (file or directory) under MODELS_DIR."""
    return models_dir() / SPEC_BY_KEY[key].target


def is_present(spec: ModelSpec) -> bool:
    p = models_dir() / spec.target
    if spec.is_dir:
        return p.is_dir() and all(any(p.glob(pat)) for pat in spec.required)
    return p.is_file() and p.stat().st_size > 0


def missing_models(keys: list[str] | None = None) -> list[str]:
    """Keys of models that are not (fully) installed (optional ones only when named in ``keys``)."""
    specs = [SPEC_BY_KEY[k] for k in keys] if keys else [s for s in SPECS if not s.optional]
    return [s.key for s in specs if not is_present(s)]


ProgressCb = Callable[[str, int, int | None], None]  # (key, bytes_done, bytes_total)


def _sha256(file: Path) -> str:
    h = hashlib.sha256()
    with open(file, "rb") as f:
        while buf := f.read(1 << 20):
            h.update(buf)
    return h.hexdigest()


def _part(dest: Path, url: str) -> Path:
    """The partial file of one source: a resume never continues another source's bytes
    (the mirrors of one model are not all the same file)."""
    return dest.with_name(f"{dest.name}.{hashlib.sha256(url.encode()).hexdigest()[:8]}.part")


def _download(url: str, dest: Path, key: str, progress_cb: ProgressCb | None, sha256: str = "",
              chunk: int = 1 << 20) -> None:
    """Download to ``dest`` through a ``.part`` file with HTTP Range resume. The file is renamed
    into place only when it is whole and, with ``sha256`` given, matches it."""
    part = _part(dest, url)
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "macd-model-downloader"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    try:
        resp = urllib.request.urlopen(req, timeout=CONNECT_TIMEOUT_S)
    except urllib.error.HTTPError as e:
        if not (e.code == 416 and have):  # 416: the part is already the whole file
            raise
    else:
        with resp:
            if have and resp.status != 206:  # server ignored Range: restart
                have = 0
            length = resp.headers.get("Content-Length")
            total = (int(length) + have) if length is not None else None
            done = have
            with open(part, "ab" if have else "wb") as f:
                while buf := resp.read(chunk):
                    f.write(buf)
                    done += len(buf)
                    if progress_cb:
                        progress_cb(key, done, total)
        if total is not None and done < total:
            raise OSError(f"incomplete download of {url}: {done}/{total} bytes")
    if sha256 and _sha256(part) != sha256:  # cut short behind a 200, or not the published file
        part.unlink()
        if have:  # the fault may be in the part this resumed: once more from the first byte
            _download(url, dest, key, progress_cb, sha256, chunk)
            return
        raise ValueError(f"checksum mismatch for {url}")
    os.replace(part, dest)


def _safe_members(tar: tarfile.TarFile, root: Path) -> list[tarfile.TarInfo]:
    root = root.resolve()
    members = []
    for m in tar.getmembers():
        name = m.name
        if os.path.isabs(name) or name.startswith("\\") or ".." in Path(name).parts:
            raise ValueError(f"unsafe path in archive: {name!r}")
        if m.issym() or m.islnk():
            raise ValueError(f"links not allowed in archive: {name!r}")
        if not (m.isfile() or m.isdir()):
            continue
        if not (root / name).resolve().is_relative_to(root):
            raise ValueError(f"path escapes target: {name!r}")
        members.append(m)
    return members


def safe_extract(archive: Path, dest_dir: Path) -> None:
    """Extract a tarball rejecting absolute paths, '..', links and special files."""
    with tarfile.open(archive, "r:*") as tar:
        members = _safe_members(tar, dest_dir)
        tar.extractall(dest_dir, members=members, filter="data")


def _install_archive(spec: ModelSpec, src: Source, root: Path, progress_cb: ProgressCb | None) -> None:
    archive = root / (spec.target + ".tar.bz2")
    _download(src.url, archive, spec.key, progress_cb, src.digest())  # verified before anything is extracted
    try:
        with tempfile.TemporaryDirectory(dir=root, prefix=".extract-") as tmp:
            try:
                safe_extract(archive, Path(tmp))
            except (tarfile.TarError, EOFError) as e:  # truncated or not an archive
                raise ValueError(f"archive {archive.name} could not be extracted: {e}") from e
            extracted = Path(tmp) / spec.target
            if not extracted.is_dir():
                raise ValueError(f"archive {archive.name} did not contain {spec.target}/")
            lacks = [pat for pat in spec.required if not any(extracted.glob(pat))]
            if lacks:  # checked before the installed copy is replaced
                raise ValueError(f"archive {archive.name} lacks {', '.join(lacks)}")
            final = root / spec.target
            if final.exists():
                shutil.rmtree(final)
            os.replace(extracted, final)
    finally:
        archive.unlink(missing_ok=True)  # a bad archive is not kept for the next source or the next run


def _install_files(spec: ModelSpec, src: Source, root: Path, progress_cb: ProgressCb | None) -> None:
    target = root / spec.target
    target.mkdir(parents=True, exist_ok=True)
    for name in src.files:
        dest, want = target / name, src.digest(name)
        if dest.is_file() and dest.stat().st_size > 0 and (not want or _sha256(dest) == want):
            continue  # already here (an earlier run, or an archive that had it)
        _download(f"{src.url}/{name}", dest, spec.key, progress_cb, want)


def _install(spec: ModelSpec, src: Source, root: Path, progress_cb: ProgressCb | None) -> None:
    if src.files:
        _install_files(spec, src, root, progress_cb)
    elif src.archive:
        _install_archive(spec, src, root, progress_cb)
    else:
        _download(src.url, root / spec.target, spec.key, progress_cb, src.digest())


def download_models(progress_cb: ProgressCb | None = None, keys: list[str] | None = None) -> list[str]:
    """Download (and extract) missing models, trying each source in order. Returns installed keys."""
    root = models_dir()
    root.mkdir(parents=True, exist_ok=True)
    installed: list[str] = []
    for key in missing_models(keys):
        spec = SPEC_BY_KEY[key]
        errors: list[str] = []
        for src in spec.sources:
            log.info("downloading model %s from %s", key, src.url)
            try:
                _install(spec, src, root, progress_cb)
            except Exception as e:  # noqa: BLE001 - whatever failed at this source, the next one may work
                errors.append(f"{src.url}: {e}")
                log.warning("source failed for %s (%s); trying next", key, e)
                continue
            if is_present(spec):
                for old in root.glob(glob.escape(spec.target) + ".*part"):  # partial files of sources that failed
                    old.unlink(missing_ok=True)
                installed.append(key)
                break
            errors.append(f"{src.url}: incomplete")
        else:
            raise OSError(f"could not download model {key}:\n  " + "\n  ".join(errors))
    return installed


def _cli_progress(key: str, done: int, total: int | None) -> None:
    mb = done / 1e6
    if total:
        sys.stderr.write(f"\r{key}: {mb:7.1f} / {total / 1e6:.1f} MB ({100 * done / total:5.1f}%)")
    else:
        sys.stderr.write(f"\r{key}: {mb:7.1f} MB")
    sys.stderr.flush()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    keys = [k for k in (argv if argv is not None else sys.argv[1:]) if k in SPEC_BY_KEY] or None
    missing = missing_models(keys)
    if not missing:
        print(f"All models present in {models_dir()}")
        return 0
    print(f"Downloading {', '.join(missing)} into {models_dir()}")
    download_models(_cli_progress, keys)
    sys.stderr.write("\n")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
