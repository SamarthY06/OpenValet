"""ListenerImpl: KWS -> stage-2 verify -> VAD endpointing -> STT for one AudioInput.

See contracts.Listener and ARCHITECTURE §3/§4.

Concurrency: bookkeeping and VAD (~0.1 ms) run on the event loop. KWS runs pipelined on a
dedicated single-thread executor ("kws_pool"): 80 ms batches are submitted without
waiting and their results collected on later frames, so a slow KWS step (the zipformer
encoder costs ~10 ms every 320 ms) never stalls VAD/endpointing. Only when the executor
falls more than ``MAX_KWS_INFLIGHT`` batches behind (file feeding faster than real time)
does the frame loop wait for it. STT always goes to the STT executor. All endpointing decisions are made in *sample time* (absolute sample indices
of the source), never wall-clock, so file-fed tests are deterministic at any feed speed.

Jobs that need STT (verify, utterance) are chained in capture order (each awaits its
predecessor before emitting), so events come out in the order the audio happened. The one
STT call outside the chain is the barge-in probe (speech over Mac pauses it only once words
are heard): its own task, at most one per capture, and its answer is applied on the frame
path, so a capture is still only changed from there and from set_speaking.

Capture model: one ``_Capture`` at a time. A woke capture stays open until its no-speech
deadline: when a segment ends in silence it is transcribed and the capture *restarts*
(same object, fresh audio) in case the transcript was just the wake phrase ("Hey Mac …
pause … open mail"). When a segment hits ``max_utterance_s`` it is transcribed and the
capture restarts with speech already in progress; that text is a ``continuation``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import re
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from macd.audio.ring import RingBuffer
from macd.config import Config
from macd.contracts import (
    FRAME_SAMPLES,
    SAMPLE_RATE,
    STT,
    VAD,
    AudioInput,
    ListenerEvent,
    ListenerEventKind,
    Utterance,
    WakeDetector,
    WakeHit,
    WindowReason,
)
from macd.eventlog import Metrics
from macd.phrases import (
    is_backchannel,
    is_echo,
    is_interjection,
    is_stop,
    normalize,
    word_count,
)
from macd.speech.wake import InputGain, WakePhrases

log = logging.getLogger(__name__)

PREROLL_S = 2.5
VERIFY_WINDOW_S = 2.0
VERIFY_TAIL_S = 0.1  # audio after the hit before verifying (the hit already follows a trailing blank)
WAKE_LOOKBACK_S = 1.6  # utterance audio starts this far before the hit (wake phrase is stripped)
WINDOW_PREROLL_S = 0.3
WINDOW_START_FRAMES = 3  # consecutive speech frames needed to open a window capture
SAME_BREATH_S = 0.6  # speech starting within this of the hit = same breath (no wake blip needed)
STOP_CAPTURE_S = 2.0  # stop-word captures while speaking (no AEC) are at most this long
TRAILING_KEEP_S = 0.3
KWS_BATCH_FRAMES = 4  # feed KWS 80 ms at a time: the encoder steps every 16 frames anyway, 4x fewer thread hops
MAX_KWS_INFLIGHT = 2  # batches the executor may lag before the frame loop waits (never in real time)
TURN_DECIDE_FRAMES = 3  # Smart Turn runs off-loop (~35 ms); its answer is applied exactly this many frames later
TURN_AUDIO_S = 8.0  # the model reads the last 8 s
REOPENABLE = (WindowReason.FOLLOWUP, WindowReason.CONFIRM)  # an empty capture keeps these windows open
FRAME_MS = FRAME_SAMPLES * 1000 // SAMPLE_RATE
# Barge-in confirmation (listen.barge_in_confirm): speech over Mac pauses it only once a quick STT
# probe of the capture so far hears words that are not Mac's own.
PROBE_AUDIO_S = 4.0  # a probe reads at most this much of the capture's tail
PROBE_DECIDE_FRAMES = 3  # a probe's answer is applied no sooner than this many frames after its start
PROBE_GAP_MS = 400  # voiced audio needed after a probe that heard no interruption; doubles each time
PROBE_GAP_MAX_MS = 1600
PROBE_MORE_MS = 200  # after a probe that heard one or two words

def _samples(seconds: float) -> int:
    return round(seconds * SAMPLE_RATE)


def _tail(parts: list[np.ndarray], n: int) -> np.ndarray:
    """The last ``n`` samples of ``parts`` without joining all of them."""
    out: list[np.ndarray] = []
    have = 0
    for p in reversed(parts):
        out.append(p)
        have += p.shape[0]
        if have >= n:
            break
    return np.concatenate(out[::-1])[-n:]


def _dbfs(x: float) -> float:
    return round(20 * math.log10(x), 1) if x > 1e-6 else -120.0


def _levels(audio: np.ndarray) -> dict[str, float | None]:
    if not audio.size:
        return {"peak_dbfs": None, "rms_dbfs": None}
    x = audio.astype(np.float64, copy=False)
    return {"peak_dbfs": _dbfs(float(np.abs(x).max())), "rms_dbfs": _dbfs(float(np.sqrt(np.mean(x * x))))}


@dataclass
class _Capture:
    woke: bool
    start: int  # absolute index of the first captured sample of the current segment
    deadline: int  # woke: give up (NO_SPEECH) if no speech by then
    max_end: int
    hit: WakeHit | None = None
    window: WindowReason | None = None
    window_end: int | None = None  # FOLLOWUP/CONFIRM: the window's end, reopened if this capture is empty
    window_token: object | None = None
    parts: list[np.ndarray] = field(default_factory=list)
    end: int = 0  # absolute index after the last captured sample
    speech_started: bool = False  # endpointing armed (silence after last_speech ends the segment)
    speech_seen: bool = False  # VAD saw speech in this segment (SPEECH_START, same_breath)
    speech_start: int = 0
    last_speech: int = 0
    verify_at: int | None = None  # launch stage-2 verify once end >= verify_at
    verified: bool = False
    rejected: bool = False
    speech_start_pending: bool = False  # SPEECH_START held back until WAKE is emitted
    capped: bool = False  # current segment continues one that hit max_utterance_s
    produced_text: bool = False  # set by the utterance job once this capture yielded text
    barge: bool = False  # SPEAKING capture with AEC: any words count (the orchestrator decides)
    onset_sent: bool = False  # BARGE_IN already emitted for this capture
    probe: asyncio.Task[str | None] | None = None  # barge-in STT probe in flight (at most one)
    probe_at: int = 0  # sample index from which its answer is applied
    probes: int = 0  # probes started for this capture
    probe_voiced: int = 0  # `voiced` when the last probe started
    probe_gap: int = 0  # voiced frames needed after it before the next probe
    probe_ms: float = 0.0  # wall time spent in probes (diagnostics)
    probe_heard: list[str] = field(default_factory=list)  # what each probe heard, as a category (diagnostics)
    faint_words: bool = False  # the last probe heard an interruption, in a sound too faint to act on (yet)
    began: str = "onset"  # how it came to be speech over Mac: "onset" (VAD) | "item_start" (Mac spoke into it)
    over_from: int = 0  # sample index from which the user and Mac were both talking (for pause_after_ms)
    pause_at: int | None = None  # sample index of the first BARGE_IN
    frames: int = 0  # frames seen by this segment, and VAD numbers over them (diagnostics)
    voiced: int = 0
    prob_sum: float = 0.0
    prob_max: float = 0.0
    wall_hit: float = 0.0
    turn_fut: asyncio.Future[float] | None = None  # Smart Turn check in flight for the current pause
    turn_at: int = 0  # sample index at which its answer is applied
    turn_wait: bool = False  # the model said "not finished" for this pause: wait for more speech

    def restart(self, pos: int, *, speaking: bool, max_end: int) -> None:
        self.parts, self.start, self.end = [], pos, pos
        self.speech_started = self.speech_seen = speaking
        self.speech_start = self.last_speech = pos
        self.capped = speaking
        self.turn_fut, self.turn_wait = None, False
        self.max_end = max_end
        self.frames, self.voiced, self.prob_sum, self.prob_max = 0, 0, 0.0, 0.0


class ListenerImpl:
    """contracts.Listener implementation for one AudioInput."""

    def __init__(
        self,
        source: AudioInput,
        wake: WakeDetector | None,
        vad: VAD,
        stt: STT,
        cfg: Config,
        aec: bool | Callable[[], bool] = False,
        metrics: Metrics | None = None,
        eventlog: Any = None,
        turn: Any = None,
    ) -> None:
        self.source = source
        self.source_name = source.name
        self.speaking = False
        self.metrics = metrics or Metrics()
        self.eventlog = eventlog  # anything with .emit(type, **fields); None = no event log
        self.frame_ms_max = 0.0  # loop time per frame (KWS executor wait excluded)
        self.kws_ms_max = 0.0
        self._wake = wake
        self._vad = vad
        self._stt = stt
        self._turn = turn  # SmartTurn (complete_prob(audio) -> float) or None: silence-only endpointing
        self._turn_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="turn_pool") if turn is not None else None
        self._cfg = cfg
        self._aec = aec
        self._phrases = WakePhrases(cfg.wake.phrases, accept_similar_names=cfg.wake.accept_similar_names)
        self._requires_wake = source.requires_wake_word and wake is not None
        self._ring = RingBuffer(_samples(PREROLL_S))
        self._pos = 0  # absolute index of the next sample
        self._capture: _Capture | None = None
        self._window_end: int | None = None
        self._window_reason: WindowReason | None = None
        self._window_token: object | None = None  # identity of the open window (None: closed)
        self._speech_run = 0  # consecutive VAD speech frames (for window starts)
        self._muted = False
        self._current_text = ""
        self._spoken: deque[str] = deque(maxlen=2)  # Mac's last items, for the probe's echo check
        # A source that is not paced in real time (a file fed as fast as it is read) waits for a
        # probe's answer at a fixed sample offset, so those runs are deterministic. A live source
        # never waits: the answer is applied on the first frame after it arrives.
        self._probe_wait = getattr(source, "realtime", True) is False
        self._guard_until = 0
        self._queue: asyncio.Queue[ListenerEvent | None] = asyncio.Queue()
        self._chain: asyncio.Task[None] | None = None
        self._jobs: set[asyncio.Task[Any]] = set()
        self._kws_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kws_pool")
        self._kws_batch: list[np.ndarray] = []
        self._kws_gain = InputGain()
        self._kws_inflight: deque[tuple[asyncio.Future[WakeHit | None], float]] = deque()
        self._vad_active = False  # VAD only runs while something can use it (idle CPU)
        self._stt_missing_logged = False
        lc = cfg.listen
        self._silence = _samples(lc.silence_ms / 1000)
        self._max_utt = _samples(lc.max_utterance_s)
        self._no_speech = _samples(lc.no_speech_timeout_s)
        self._vad_th = lc.vad_threshold
        self._tail_guard = _samples(lc.tail_guard_ms / 1000)
        self._barge_onset_frames = max(1, round(lc.barge_in_onset_ms / 20))
        self._barge_conf = lc.barge_in_min_confidence

    # ------------------------------------------------------------------ public API

    @property
    def aec(self) -> bool:
        return bool(self._aec() if callable(self._aec) else self._aec)

    async def run(self) -> None:
        try:
            async for frame in self.source.frames():
                await self._on_frame(np.asarray(frame, dtype=np.float32))
            while self._kws_inflight:  # late hits at end of input still count (file sources)
                await self._collect_kws(wait=True)
        finally:
            self._kws_pool.shutdown(wait=False)
            if self._turn_pool is not None:
                self._turn_pool.shutdown(wait=False)
            self._end_of_input()
            while self._chain is not None and not self._chain.done():
                await asyncio.shield(self._chain)
            if self._jobs:  # a barge-in probe still in flight (its answer no longer matters)
                await asyncio.wait(list(self._jobs))
            self._queue.put_nowait(None)

    async def events(self) -> AsyncIterator[ListenerEvent]:
        while True:
            ev = await self._queue.get()
            if ev is None:
                return
            yield ev

    def open_window(self, seconds: float, reason: WindowReason) -> None:
        self._window_end = self._pos + _samples(seconds)
        self._window_reason = reason
        self._window_token = object()
        # PTT starts on the very first speech frame; others need a short run.
        self._speech_run = WINDOW_START_FRAMES if reason is WindowReason.PTT else 0

    def close_window(self) -> None:
        self._window_end = None
        self._window_reason = None
        self._window_token = None
        c = self._capture
        if c is not None and not c.woke and c.window is not WindowReason.SPEAKING:
            self._capture = None

    def set_muted(self, muted: bool) -> None:
        self._muted = muted
        if muted:
            self._capture = None
            self._window_end = None
            self._window_token = None
            if self._wake is not None:
                self._kws_pool.submit(self._wake.reset)  # same thread as accept(): no race

    def set_wake_detector(self, det: WakeDetector | None) -> None:
        """Swap the KWS detector between frames (batches already in flight finish on the old one)."""
        self._wake = det
        self._requires_wake = self.source.requires_wake_word and det is not None
        self._kws_batch.clear()

    def set_wake_phrases(self, phrases: WakePhrases) -> None:
        """Swap the stage-2 verifier/stripper (single attribute store: safe from any thread)."""
        self._phrases = phrases

    def set_speaking(self, speaking: bool, current_text: str = "") -> None:
        c = self._capture
        lc = self._cfg.listen
        probe = False
        if speaking and not self.speaking and c is not None and c.pause_at is None:
            c.over_from = self._pos  # Mac was silent until now: the two overlap from here (pause_after_ms)
        if speaking and c is not None and c.speech_seen and self.aec and lc.barge_in and (not c.woke or c.produced_text):
            # Mac starts an item while the user is mid-utterance (live 2026-11-11: Smart Turn kept a
            # thinking pause open, the reply started and talked over them). The capture becomes a
            # barge-in. A wake capture that already yielded its command (kept open for a follow-on)
            # is the same.
            if c.onset_sent or (not lc.barge_in_confirm and c.window is not WindowReason.SPEAKING):
                # BARGE_IN goes out now, so Mac pauses before its first word: for a capture that
                # already paused Mac once, at every later item start while it is open (the first
                # one may have landed on an earcon or system line, which the orchestrator does not
                # pause, and the answer sentences that follow must not play over the user either);
                # with barge_in_confirm off, also for a capture that has not paused it yet.
                self._convert(c)
                self._pause(c)
            elif lc.barge_in_confirm and (c.window is not WindowReason.SPEAKING or c.barge):
                # Not confirmed yet: what the capture holds may be a cough or noise. Ask STT now
                # (not after another onset) and pause only if it hears words.
                self._convert(c)
                probe = c.probe is None
        if self.speaking and not speaking:
            if not self.aec:
                self._guard_until = self._pos + self._tail_guard
            c = self._capture
            if c is not None and c.window is WindowReason.SPEAKING and not c.barge:
                self._capture = None  # playback ended: no more stop-word captures (a barge-in finishes)
        self.speaking = speaking
        self._current_text = current_text if speaking else ""
        if speaking and current_text:
            self._spoken.append(current_text)
        if probe and c is not None:
            self._start_probe(c)

    def _convert(self, c: _Capture) -> None:
        """An open capture becomes speech over Mac (a barge-in capture)."""
        if c.window is not WindowReason.SPEAKING:
            c.began, c.over_from = "item_start", self._pos
        c.window, c.barge, c.woke = WindowReason.SPEAKING, True, False

    def _confident(self, c: _Capture) -> bool:
        """The VAD has been sure, at least once, that this capture holds a voice at the mic. Echo
        that leaks through AEC and voices far from the mic stay under it, yet STT still makes
        sentences of them (live 2026-11-24: a full sentence, close to what Mac had just said). A
        capture that began while Mac was silent is not judged: it was good enough for the open
        window, and a quiet speaker must not lose the sentence because the reply started into it."""
        return c.began == "item_start" or c.prob_max >= self._barge_conf

    def _pause(self, c: _Capture) -> None:
        """BARGE_IN, straight onto the queue (not chained behind STT jobs): the pause must be instant."""
        c.onset_sent = True
        if c.pause_at is None:
            c.pause_at = self._pos
        self._queue.put_nowait(ListenerEvent(ListenerEventKind.BARGE_IN, self.source_name, window=WindowReason.SPEAKING))

    # ------------------------------------------------------------------ frame path

    async def _on_frame(self, frame: np.ndarray) -> None:
        t0 = time.perf_counter()
        idx = self._pos
        self._pos += frame.shape[0]
        self._ring.append(frame)
        if not self._muted and self._requires_wake and self._wake is not None:
            self._kws_batch.append(frame)
            if len(self._kws_batch) >= KWS_BATCH_FRAMES:
                chunk = self._kws_gain(np.concatenate(self._kws_batch))
                self._kws_batch.clear()
                fut = asyncio.get_running_loop().run_in_executor(self._kws_pool, self._wake.accept, chunk, self._pos - chunk.shape[0])
                self._kws_inflight.append((fut, time.perf_counter()))
        t_wait = time.perf_counter()
        await self._collect_kws(wait=len(self._kws_inflight) > MAX_KWS_INFLIGHT)
        waited = time.perf_counter() - t_wait
        vad_needed = self._capture is not None or self._window_end is not None or self.speaking
        if vad_needed and not self._vad_active:
            self._vad.reset()
        self._vad_active = vad_needed
        prob = float(self._vad.accept(frame)) if vad_needed else 0.0
        is_speech = vad_needed and prob >= self._vad_th
        self._speech_run = self._speech_run + 1 if is_speech else 0
        if not self._muted:
            self._step_capture(frame, idx, is_speech, prob)
            t_wait = time.perf_counter()
            await self._apply_turn_check()
            await self._apply_probe()
            waited += time.perf_counter() - t_wait
            self._step_window(is_speech)
        dt = (time.perf_counter() - t0 - waited) * 1000
        self.frame_ms_max = max(self.frame_ms_max, dt)
        self.metrics.observe("listener_frame", dt)
        if dt > 15:
            log.warning("listener frame took %.1f ms (budget 20 ms)", dt)

    async def _collect_kws(self, *, wait: bool) -> None:
        """Harvest finished KWS batches in order (``wait``: block for the oldest one)."""
        while self._kws_inflight and (wait or self._kws_inflight[0][0].done()):
            fut, t0 = self._kws_inflight.popleft()
            wait = False
            try:
                hit = await fut
            except Exception:
                log.exception("KWS failed on a batch")
                continue
            ms = (time.perf_counter() - t0) * 1000  # submit -> result, incl. queueing
            self.kws_ms_max = max(self.kws_ms_max, ms)
            self.metrics.observe("listener_kws", ms)
            if hit is not None and not self._muted:
                self._on_hit(hit)

    def _emit_log(self, type_: str, **fields: Any) -> None:
        if self.eventlog is not None:
            self.eventlog.emit(type_, source=self.source_name, **fields)

    def _on_hit(self, hit: WakeHit) -> None:
        self.metrics.incr("kws_hits")
        self._emit_log("kws_hit", phrase=hit.phrase, at_s=round(hit.sample_index / SAMPLE_RATE, 2), speaking=self.speaking)
        if self.speaking and not self.aec and self._phrases.mentions(self._current_text):
            self.metrics.incr("wakes_suppressed")
            self._emit_log("wake_suppressed", reason="self_trigger", spoken=self._current_text[:80])
            log.info("wake hit suppressed (self-trigger: speaking %r)", self._current_text[:40])
            return
        cur = self._capture
        if cur is not None and cur.woke and not cur.rejected:
            self._emit_log("wake_suppressed", reason="already_capturing")
            return  # already capturing after a wake word
        start = max(self._ring.oldest, hit.sample_index - _samples(WAKE_LOOKBACK_S))
        # The KWS hit can land after a short command fused with the wake phrase ("Hey Mac, stop"),
        # so endpointing is armed right away: silence after the hit transcribes the look-back.
        cap = _Capture(
            woke=True,
            start=start,
            deadline=hit.sample_index + self._no_speech,
            max_end=hit.sample_index + self._max_utt,
            hit=hit,
            speech_started=True,
            speech_start=hit.sample_index,
            last_speech=hit.sample_index,
            wall_hit=time.monotonic(),
        )
        cap.parts.append(self._ring.slice(start, self._pos))
        cap.end = self._pos
        if self._cfg.wake.verify:
            cap.verify_at = hit.sample_index + _samples(VERIFY_TAIL_S)
        else:
            cap.verified = True
            self._emit_log("wake_verify", accepted=True, transcript=None, reason="verify_disabled")
            self._enqueue(self._emit_wake(cap))
        self._capture = cap

    def _step_capture(self, frame: np.ndarray, idx: int, is_speech: bool, prob: float = 0.0) -> None:
        c = self._capture
        if c is None:
            return
        if c.rejected:
            self._capture = None
            return
        c.frames += 1
        c.voiced += is_speech
        c.prob_sum += prob
        c.prob_max = max(c.prob_max, prob)
        if idx >= c.end:  # frames already included via the pre-roll slice are skipped
            c.parts.append(frame.copy())
            c.end = idx + frame.shape[0]
        if c.verify_at is not None and c.end >= c.verify_at:
            c.verify_at = None
            self._enqueue(self._verify(c, self._ring.slice(c.end - _samples(VERIFY_WINDOW_S), c.end)))
        after_hit = c.hit is None or idx >= c.hit.sample_index
        if is_speech and after_hit:
            c.last_speech = c.end
            c.turn_fut, c.turn_wait = None, False  # talking again: any verdict on the pause is moot
            if not c.speech_seen:
                c.speech_seen = c.speech_started = True
                c.speech_start = idx - (self._speech_run - 1) * FRAME_SAMPLES
                if c.woke and not c.verified:
                    c.speech_start_pending = True  # emitted right after WAKE (keeps event order)
                else:
                    self._enqueue(self._emit(ListenerEvent(ListenerEventKind.SPEECH_START, self.source_name)))
        if c.barge and not c.onset_sent and is_speech:
            if not self._cfg.listen.barge_in_confirm:
                if self._speech_run >= self._barge_onset_frames:
                    self._pause(c)  # the old rule: any sound that lasts barge_in_onset_ms pauses Mac
            elif self.speaking and c.probe is None and self._probe_due(c):
                self._start_probe(c)
        if c.speech_started:
            silent = c.end - c.last_speech
            if c.end >= c.max_end:
                self._finalize(c, capped=is_speech)  # still talking: the rest continues as dictation
            elif self._smart(c):
                lc = self._cfg.listen
                if silent >= _samples(lc.turn_max_silence_ms / 1000):
                    self._finalize(c, capped=False)  # the model said "not finished" but nothing followed
                elif c.turn_fut is None and not c.turn_wait and silent >= _samples(lc.turn_check_ms / 1000):
                    assert self._turn is not None and self._turn_pool is not None
                    audio = np.concatenate(c.parts)[-_samples(TURN_AUDIO_S):]
                    c.turn_fut = asyncio.get_running_loop().run_in_executor(self._turn_pool, self._turn.complete_prob, audio)
                    c.turn_at = c.end + TURN_DECIDE_FRAMES * FRAME_SAMPLES
            elif silent >= self._silence:
                self._finalize(c, capped=False)
        elif c.end >= c.deadline:
            self._capture = None
            self._enqueue(self._no_speech_job(c))

    def _smart(self, c: _Capture) -> bool:
        """Semantic end of turn for this capture (not for short no-AEC stop-word captures)."""
        return (
            self._turn is not None and self._cfg.listen.smart_turn
            and not (c.window is WindowReason.SPEAKING and not c.barge)
        )

    async def _apply_turn_check(self) -> None:
        """Act on the Smart Turn verdict for the current pause, at a fixed sample offset so that
        file-fed runs are deterministic (in real time the answer is normally already there)."""
        c = self._capture
        if c is None or c.turn_fut is None or c.end < c.turn_at:
            return
        fut, c.turn_fut = c.turn_fut, None
        t0 = time.perf_counter()
        try:
            prob = await fut
        except Exception:
            log.exception("Smart Turn failed; ending the turn on silence")
            prob = 1.0
        self.metrics.observe("turn_check_wait", (time.perf_counter() - t0) * 1000)
        if self._capture is not c or c.rejected or c.turn_fut is not None:
            return
        done = prob >= self._cfg.listen.turn_threshold
        self._emit_log("turn_check", prob=round(prob, 3), complete=done, silence_ms=round((c.end - c.last_speech) * 1000 / SAMPLE_RATE))
        if done:
            self._finalize(c, capped=False)
        else:
            c.turn_wait = True

    # ------------------------------------------------------------------ barge-in confirmation

    def _probe_due(self, c: _Capture) -> bool:
        """First probe after barge_in_onset_ms of unbroken speech; after an empty one, only once
        more voiced audio has arrived (400 ms, then twice as much each time, up to 1.6 s), or at once
        when words were heard in a sound that was faint then and is clearly a voice now."""
        if c.probes == 0:
            return self._speech_run >= self._barge_onset_frames
        if c.faint_words and self._confident(c):
            return True  # words were heard while it was faint, and now it is clearly a voice: listen again at once
        return c.voiced - c.probe_voiced >= c.probe_gap

    def _start_probe(self, c: _Capture) -> None:
        """Transcribe the capture so far, off the frame path and not behind the chained jobs."""
        audio = _tail(c.parts, _samples(PROBE_AUDIO_S))
        c.probes += 1
        c.faint_words = False
        c.probe_voiced = c.voiced
        c.probe_gap = PROBE_GAP_MS // FRAME_MS if c.probes == 1 else min(c.probe_gap * 2, PROBE_GAP_MAX_MS // FRAME_MS)
        c.probe_at = c.end + PROBE_DECIDE_FRAMES * FRAME_SAMPLES
        c.probe = asyncio.get_running_loop().create_task(self._probe_stt(c, audio))
        self._jobs.add(c.probe)
        c.probe.add_done_callback(self._jobs.discard)

    async def _probe_stt(self, c: _Capture, audio: np.ndarray) -> str | None:
        """The probe's transcript: "" for no words (or a failed STT), None if the model is missing."""
        t0 = time.monotonic()
        try:
            return await self._transcribe(audio, "barge_probe_stt")
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("barge-in probe failed")
            return ""
        finally:
            c.probe_ms += (time.monotonic() - t0) * 1000

    async def _apply_probe(self) -> None:
        """Act on a finished probe: an interruption (see `_heard`) pauses Mac (BARGE_IN), from a faint
        capture (see `_confident`) only if it is a stop word. All of it
        happens here on the frame path, so the capture is only ever changed from one place."""
        c = self._capture
        if c is None or c.probe is None or c.end < c.probe_at:
            return
        task = c.probe
        if not task.done():
            if not self._probe_wait:
                return  # live audio: look again on the next frame
            await asyncio.wait([task])
            if self._capture is not c or c.probe is not task:
                return  # muted, closed or replaced while waiting
        c.probe = None
        if task.cancelled() or c.rejected or c.onset_sent or not self.speaking:
            return  # Mac stopped by itself meanwhile: nothing to pause
        text = task.result()
        if text is None:
            self._pause(c)  # no STT model: the old rule (pause on the sound, the verdict resumes it)
            return
        heard = self._heard(text)
        c.probe_heard.append(heard)
        if heard == "words":
            if self._confident(c) or is_stop(text):
                self._pause(c)
            else:  # a faint sound pauses Mac for "stop" only, whatever else STT made of it
                c.faint_words = True
        elif heard == "few":
            c.probe_gap = PROBE_MORE_MS // FRAME_MS  # maybe the start of a sentence ("Can you"): listen again soon

    def _heard(self, text: str) -> str:
        """What a probe's transcript is. Only "words" pauses Mac: a stop word or "wait", or as many words
        as an interruption needs (barge_in_min_words, the rule the orchestrator applies to the whole
        utterance) with two of them not Mac's own. Less than that is also what STT makes of half a
        second of noise, or of leaked echo cut off mid-word ("Here is a" for "Here is what I found")."""
        words = normalize(text).split()
        spoken = " ".join(self._spoken)
        if not words:
            return "none"
        if is_echo(text, spoken):
            return "echo"
        if is_backchannel(text):
            return "backchannel"  # "yeah" alone lets Mac talk on; words that follow it are heard by the next probe
        if is_stop(text) or is_interjection(text):
            return "words"
        said = set(normalize(spoken).split())
        if len(words) >= self._cfg.listen.barge_in_min_words and sum(w not in said for w in words) >= 2:
            return "words"
        return "few"

    def _step_window(self, is_speech: bool) -> None:
        if self._capture is not None:
            return
        reason = self._window_reason if self._window_end is not None else None
        if self.speaking and reason in (None, WindowReason.TURN):
            if is_speech and self._speech_run >= WINDOW_START_FRAMES:
                # Speech over Mac. With AEC it is a possible barge-in (a full utterance, BARGE_IN once
                # words are heard); without AEC our own voice is in the mic, so only a short stop-word capture.
                barge = self.aec and self._cfg.listen.barge_in
                end = self._pos + (self._max_utt if barge else _samples(STOP_CAPTURE_S))
                self._start_vad_capture(WindowReason.SPEAKING, end, barge=barge)
            return
        if self._window_end is None:
            return
        suppressed = (self.speaking or self._pos < self._guard_until) and not self.aec
        if is_speech and not suppressed and self._speech_run >= WINDOW_START_FRAMES:
            window_end = self._window_end
            self._start_vad_capture(reason, self._pos + self._max_utt)
            if reason is not WindowReason.TURN:
                self._window_end = None  # one utterance per window (a TURN window stays open)
                if reason in REOPENABLE and self._capture is not None:
                    self._capture.window_end, self._capture.window_token = window_end, self._window_token
        elif self._pos >= self._window_end:
            self._window_end, self._window_reason = None, None
            self._enqueue(self._emit(ListenerEvent(ListenerEventKind.NO_SPEECH, self.source_name, window=reason)))

    def _start_vad_capture(self, window: WindowReason | None, max_end: int, *, barge: bool = False) -> None:
        start = max(self._ring.oldest, self._pos - _samples(WINDOW_PREROLL_S) - self._speech_run * FRAME_SAMPLES)
        cap = _Capture(
            woke=False,
            start=start,
            deadline=max_end,
            max_end=max_end,
            window=window,
            speech_started=True,
            speech_seen=True,
            speech_start=start,
            last_speech=self._pos,
            barge=barge,
            voiced=self._speech_run,
            over_from=self._pos - self._speech_run * FRAME_SAMPLES,
        )
        cap.parts.append(self._ring.slice(start, self._pos))
        cap.end = self._pos
        self._capture = cap
        if window is not WindowReason.SPEAKING:
            self._enqueue(self._emit(ListenerEvent(ListenerEventKind.SPEECH_START, self.source_name)))

    def _finalize(self, c: _Capture, *, capped: bool) -> None:
        """Transcribe the current segment; ``capped`` = speech is still going at max_utterance_s."""
        keep_to = min(c.end, c.last_speech + _samples(TRAILING_KEEP_S))
        audio = np.concatenate(c.parts)[: max(0, keep_to - c.start)]
        same_breath = c.woke and c.hit is not None and not c.capped and c.speech_start - c.hit.sample_index <= _samples(SAME_BREATH_S)
        # (no post-hit speech at all = the command was fused with the wake phrase: same breath)
        faint = c.barge and not self._confident(c)
        diag = self._capture_numbers(c, faint) if c.window is WindowReason.SPEAKING else None
        now = time.monotonic()
        # when the capture began, on the clock: `audio` is trimmed, and a pause that was waited out is not in it
        self._enqueue(self._utterance_job(c, audio, now, started=now - (c.end - c.start) / SAMPLE_RATE,
                                          continuation=c.capped, same_breath=same_breath, faint=faint, diag=diag))
        if capped and c.window is not WindowReason.SPEAKING:
            c.restart(self._pos, speaking=True, max_end=self._pos + self._max_utt)
        elif c.woke and self._pos < c.deadline:
            c.restart(self._pos, speaking=False, max_end=self._pos + self._max_utt)  # maybe only "hey mac" so far
        else:
            self._capture = None

    def _capture_numbers(self, c: _Capture, faint: bool) -> dict[str, Any]:
        """What a finished SPEAKING capture looked like, for the event log (numbers only)."""
        ms = 1000 / SAMPLE_RATE
        return {
            "aec": self.aec,
            "barge": c.barge,
            "paused": c.onset_sent,
            "faint": faint,
            "began": c.began,
            "capture_ms": round((c.end - c.start) * ms),
            "voiced_ms": c.voiced * FRAME_MS,
            "vad_max": round(c.prob_max, 3),
            "vad_mean": round(c.prob_sum / c.frames, 3) if c.frames else None,
            "probes": c.probes,
            "probe_heard": list(c.probe_heard),
            "pause_after_ms": None if c.pause_at is None else round((c.pause_at - c.over_from) * ms),
        }

    def _end_of_input(self) -> None:
        c, self._capture = self._capture, None
        if c is not None and c.speech_started:
            self._finalize(c, capped=False)
            self._capture = None

    # ------------------------------------------------------------------ jobs (chained)

    def _enqueue(self, coro: Coroutine[Any, Any, None]) -> None:
        prev = self._chain

        async def job() -> None:
            if prev is not None:
                with contextlib.suppress(Exception):  # predecessor already logged its error
                    await prev
            try:
                await coro
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("listener job failed")

        task = asyncio.get_running_loop().create_task(job())
        self._chain = task
        self._jobs.add(task)
        task.add_done_callback(self._jobs.discard)

    async def _emit(self, ev: ListenerEvent) -> None:
        self._queue.put_nowait(ev)

    async def _emit_wake(self, c: _Capture) -> None:
        if c.wall_hit:
            self.metrics.observe_since("kws_hit_to_wake", c.wall_hit)
        self._window_end = self._window_token = None  # a verified wake supersedes any open window
        self.metrics.incr("wakes")
        self._queue.put_nowait(ListenerEvent(ListenerEventKind.WAKE, self.source_name, wake=c.hit))
        if c.speech_start_pending:
            c.speech_start_pending = False
            self._queue.put_nowait(ListenerEvent(ListenerEventKind.SPEECH_START, self.source_name))

    async def _transcribe(self, audio: np.ndarray, metric: str) -> str | None:
        """STT through the executor; None if the model is not installed (logged once)."""
        t0 = time.monotonic()
        try:
            text = await self._stt.transcribe(audio)
        except FileNotFoundError:
            if not self._stt_missing_logged:
                self._stt_missing_logged = True
                log.warning("STT model missing: wake hits are unverified and utterances are dropped")
            return None
        self.metrics.observe_since(metric, t0)
        return text.strip()

    async def _verify(self, c: _Capture, audio: np.ndarray) -> None:
        text = await self._transcribe(audio, "wake_verify_stt")
        unheard = text == "" and self._cfg.wake.accept_unheard
        if text is None or unheard or self._phrases.verify(text):
            c.verified = True  # no STT -> lighter path: trust the KWS hit
            reason = "no_stt" if text is None else "no_words" if unheard else "phrase_found"
            self._emit_log("wake_verify", accepted=True, transcript=text, reason=reason)
            await self._emit_wake(c)
        else:
            c.rejected = True
            self.metrics.incr("false_wakes")
            self._emit_log("wake_verify", accepted=False, transcript=text, reason="phrase_not_found")
            log.info("false wake rejected by stage-2 (heard %r)", text)
            self._queue.put_nowait(ListenerEvent(ListenerEventKind.FALSE_WAKE, self.source_name, wake=c.hit))

    async def _utterance_job(self, c: _Capture, audio: np.ndarray, wall_end: float, *, started: float, continuation: bool,
                             same_breath: bool, faint: bool = False, diag: dict[str, Any] | None = None) -> None:
        if c.rejected:
            return
        text = await self._transcribe(audio, "listener_stt")
        if diag is not None:  # numbers only: never what was heard or what Mac was saying
            self._emit_log("barge_capture", **diag, **_levels(audio), probe_ms=round(c.probe_ms),
                           words=None if text is None else word_count(text))
        if c.barge and not text:  # a false barge-in (cough, echo, no STT): Mac resumes
            self._queue.put_nowait(ListenerEvent(ListenerEventKind.NO_SPEECH, self.source_name, window=c.window))
            return
        if text is None:
            return
        if c.woke and not c.produced_text:
            text = self._phrases.remove(text)
        if c.window is WindowReason.SPEAKING:
            if not c.barge and (not is_stop(text) or (not self.aec and self._own_voice(text))):
                return  # no AEC: anything but a bare stop word (or our own voice) is dropped silently
            if is_stop(text):
                self.metrics.incr("stop_words")
        if not text:
            if not c.woke and not self._reopen_window(c):
                self._queue.put_nowait(ListenerEvent(ListenerEventKind.NO_SPEECH, self.source_name, window=c.window))
            return  # woke: the restarted capture keeps listening until the deadline
        woke = c.woke and not c.produced_text
        c.produced_text = True
        dur = audio.shape[0] / SAMPLE_RATE
        utt = Utterance(
            text=text, source=self.source_name, woke=woke, started_at=started, ended_at=wall_end, audio_s=dur,
            window=c.window, same_breath=woke and same_breath, continuation=continuation, faint=faint,
        )
        self._queue.put_nowait(ListenerEvent(ListenerEventKind.UTTERANCE, self.source_name, utterance=utt, wake=c.hit if woke else None))
        if self._capture is c and not c.speech_seen:
            self._capture = None  # got the command; no need to keep waiting for one

    def _reopen_window(self, c: _Capture) -> bool:
        """An empty capture (cough, click, echo) in a follow-up or confirm window: keep listening
        for the rest of that window instead of ending it, so the real answer still counts."""
        if c.window_end is None or c.window_token is None or c.window_token is not self._window_token:
            return False  # the window was closed or replaced meanwhile
        if self._window_end is not None or self._capture is not None or self._muted or self._pos >= c.window_end:
            return False
        self._window_end, self._window_reason = c.window_end, c.window
        self._speech_run = 0
        self._emit_log("window_reopened", window=c.window.value if c.window else None)
        return True

    def _own_voice(self, heard: str) -> bool:
        """Self-trigger check without AEC: is Mac itself saying a word of the heard stop phrase?"""
        spoken = set(re.findall(r"[a-z']{4,}", self._current_text.lower()))
        return any(w in spoken for w in re.findall(r"[a-z']{4,}", heard.lower()))

    async def _no_speech_job(self, c: _Capture) -> None:
        if not c.rejected and not c.produced_text:
            self._queue.put_nowait(ListenerEvent(ListenerEventKind.NO_SPEECH, self.source_name))
