"""ListenerImpl with fake KWS/VAD/STT: deterministic, sample-time endpointing."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from macd.audio.file_backend import FileInput
from macd.config import Config
from macd.contracts import SAMPLE_RATE, ListenerEventKind, WakeHit, WindowReason
from macd.speech.listener import ListenerImpl

K = ListenerEventKind


class FakeWake:
    """Fires once when the stream passes ``hit_at`` samples."""

    def __init__(self, hit_at: int | None) -> None:
        self.hit_at = hit_at
        self.fired = False

    def accept(self, frame, sample_index):
        end = sample_index + len(frame)
        if self.hit_at is not None and not self.fired and end >= self.hit_at:
            self.fired = True
            return WakeHit("hey mac", end)
        return None

    def reset(self):
        self.fired = False


class FakeVAD:
    """Speech = frame energy above zero (test audio is 0.5 in speech regions)."""

    def accept(self, frame):
        return 1.0 if float(np.abs(frame).max()) > 0.1 else 0.0

    def reset(self):
        pass


class FakeSTT:
    def __init__(self, text: str = "hey mac what time is it", missing: bool = False) -> None:
        self.text, self.missing, self.calls = text, missing, []

    async def transcribe(self, pcm):
        await asyncio.sleep(0)
        if self.missing:
            raise FileNotFoundError("no model")
        self.calls.append(len(pcm) / SAMPLE_RATE)
        return "" if len(pcm) < SAMPLE_RATE // 5 else self.text


def audio(*regions: tuple[float, float], total_s: float = 6.0) -> np.ndarray:
    x = np.zeros(int(total_s * SAMPLE_RATE), np.float32)
    for a, b in regions:
        x[int(a * SAMPLE_RATE) : int(b * SAMPLE_RATE)] = 0.5
    return x


async def run(pcm, wake_at: float | None, stt=None, cfg=None, before=None, trailing=0.5, turn=None, aec=False):
    cfg = cfg or Config()
    src = FileInput(pcm, trailing_silence_s=trailing)
    stt = stt or FakeSTT()
    lis = ListenerImpl(src, FakeWake(int(wake_at * SAMPLE_RATE) if wake_at is not None else None), FakeVAD(), stt, cfg, aec=aec, turn=turn)
    if before:
        before(lis)
    events = []

    async def collect():
        async for ev in lis.events():
            events.append(ev)

    await asyncio.wait_for(asyncio.gather(lis.run(), collect()), 10)
    return [e.kind for e in events], events, lis


async def test_wake_then_utterance_same_breath():
    # "hey mac" 1.0-1.6 s (hit at 1.6), command 1.7-2.6 s, then silence
    kinds, events, lis = await run(audio((1.0, 1.6), (1.7, 2.6)), wake_at=1.6)
    assert kinds == [K.WAKE, K.SPEECH_START, K.UTTERANCE]
    utt = events[-1].utterance
    assert utt.woke and utt.text == "What time is it"
    assert utt.same_breath and not utt.continuation and utt.window is None
    assert events[0].wake.sample_index == int(1.6 * SAMPLE_RATE)
    assert 2.8 < utt.audio_s < 3.0  # 1.6 s look-back before the hit (clamped at 0) + command + 0.3 s tail
    assert lis.frame_ms_max < 20
    assert lis.metrics.latency("listener_frame")["n"] > 100


async def test_false_wake_rejected():
    kinds, _, lis = await run(audio((1.0, 1.6), (1.7, 2.6)), wake_at=1.6, stt=FakeSTT("I ate a big mac"))
    assert kinds == [K.FALSE_WAKE]
    assert lis.metrics.counters["false_wakes"] == 1


async def test_verify_hearing_no_words_trusts_kws():
    """Parakeet often returns "" for a lone short "Hey Mac": that is no evidence against the hit."""
    events_seen = []

    class Log:
        def emit(self, type_, **f):
            events_seen.append((type_, f))

    stt = SeqSTT("", "open mail")
    kinds, _, _ = await run(audio((1.0, 1.6), (2.4, 3.2)), wake_at=1.6, stt=stt, before=lambda l: setattr(l, "eventlog", Log()))
    assert kinds[0] is K.WAKE and kinds[-1] is K.UTTERANCE and K.FALSE_WAKE not in kinds
    assert [f["reason"] for t, f in events_seen if t == "wake_verify"] == ["no_words"]


async def test_verify_hearing_no_words_rejects_when_disabled():
    cfg = Config()
    cfg.wake.accept_unheard = False
    kinds, _, _ = await run(audio((1.0, 1.6)), wake_at=1.6, stt=FakeSTT(""), cfg=cfg)
    assert kinds == [K.FALSE_WAKE]


async def test_verify_disabled_trusts_kws():
    cfg = Config()
    cfg.wake.verify = False
    kinds, _, _ = await run(audio((1.0, 1.6), (1.7, 2.6)), wake_at=1.6, cfg=cfg)
    assert kinds == [K.WAKE, K.SPEECH_START, K.UTTERANCE]


async def test_wake_without_speech_times_out():
    cfg = Config()
    cfg.listen.no_speech_timeout_s = 2.0
    kinds, _, _ = await run(audio((1.0, 1.6)), wake_at=1.6, cfg=cfg, stt=FakeSTT("hey mac"))
    assert kinds == [K.WAKE, K.NO_SPEECH]


async def test_command_fused_before_late_hit():
    """KWS hit lands after a short command ("Hey Mac, stop" -> hit at 1.9 s): silence after the
    hit transcribes the look-back, and the utterance counts as same breath."""
    kinds, events, _ = await run(audio((1.0, 1.8)), wake_at=1.9, stt=FakeSTT("Hey Macstop!"))
    assert kinds == [K.WAKE, K.UTTERANCE]
    assert events[-1].utterance.text == "Stop!" and events[-1].utterance.same_breath


async def test_wake_pause_then_command():
    """Wake alone (KWS fires mid-word, transcript strips to nothing) -> keep listening ->
    the command after the pause is still the woke utterance."""
    stt = FakeSTT("hey mac")
    orig = stt.transcribe

    async def transcribe(pcm):
        # second call is the continuation (short audio)
        return "open mail" if len(stt.calls) >= 2 else await orig(pcm)

    stt.transcribe = transcribe  # type: ignore[method-assign]
    kinds, events, _ = await run(audio((1.0, 1.7), (3.0, 3.8)), wake_at=1.5, stt=stt)
    assert kinds == [K.WAKE, K.SPEECH_START, K.SPEECH_START, K.UTTERANCE]
    assert events[-1].utterance.woke and events[-1].utterance.text == "Open mail"
    assert not events[-1].utterance.same_breath


async def test_window_mode_and_expiry():
    kinds, events, _ = await run(audio((1.0, 2.0)), wake_at=None, stt=FakeSTT("yes"), before=lambda l: l.open_window(3.0, WindowReason.CONFIRM))
    assert kinds == [K.SPEECH_START, K.UTTERANCE]
    assert not events[-1].utterance.woke and events[-1].utterance.text == "yes"
    assert events[-1].utterance.window is WindowReason.CONFIRM
    kinds, events, _ = await run(audio(total_s=3.0), wake_at=None, before=lambda l: l.open_window(1.0, WindowReason.FOLLOWUP))
    assert kinds == [K.NO_SPEECH] and events[0].window is WindowReason.FOLLOWUP


async def test_window_suppressed_while_speaking_without_aec():
    def before(l):
        l.open_window(5.0, WindowReason.FOLLOWUP)
        l.set_speaking(True, "Here is the answer")

    kinds, _, _ = await run(audio((1.0, 2.0)), wake_at=None, before=before)
    assert kinds == [K.NO_SPEECH]


async def test_self_trigger_suppressed_when_spoken_text_mentions_mac():
    kinds, _, lis = await run(audio((1.0, 1.6), (1.7, 2.6)), wake_at=1.6, before=lambda l: l.set_speaking(True, "I'm Mac, your assistant"))
    assert kinds == [] and lis.metrics.counters["wakes_suppressed"] == 1
    # with AEC the same hit is accepted
    kinds, _, _ = await run(audio((1.0, 1.6), (1.7, 2.6)), wake_at=1.6, before=lambda l: (setattr(l, "_aec", True), l.set_speaking(True, "I'm Mac")))
    assert K.WAKE in kinds


async def test_muted_ignores_everything():
    kinds, _, _ = await run(audio((1.0, 1.6), (1.7, 2.6)), wake_at=1.6, before=lambda l: l.set_muted(True))
    assert kinds == []


async def test_stt_missing_still_wakes():
    kinds, _, _ = await run(audio((1.0, 1.6), (1.7, 2.6)), wake_at=1.6, stt=FakeSTT(missing=True))
    assert kinds == [K.WAKE, K.SPEECH_START]


async def test_slow_kws_does_not_block_loop():
    """A 30 ms KWS step runs on kws_pool: loop time per frame stays < 20 ms and other tasks keep running."""
    import time

    class SlowWake(FakeWake):
        def accept(self, frame, sample_index):
            time.sleep(0.03)
            return super().accept(frame, sample_index)

    gaps = []

    async def ticker():
        last = time.perf_counter()
        while True:
            await asyncio.sleep(0.005)
            now = time.perf_counter()
            gaps.append(now - last)
            last = now

    t = asyncio.create_task(ticker())
    src = FileInput(audio((1.0, 1.6), (1.7, 2.6), total_s=3.0), trailing_silence_s=0.0)
    lis = ListenerImpl(src, SlowWake(int(1.6 * SAMPLE_RATE)), FakeVAD(), FakeSTT(), Config())
    events = []

    async def collect():
        async for ev in lis.events():
            events.append(ev.kind)

    await asyncio.wait_for(asyncio.gather(lis.run(), collect()), 30)
    t.cancel()
    assert events[:2] == [K.WAKE, K.SPEECH_START]
    assert lis.frame_ms_max < 20
    assert lis.kws_ms_max >= 30
    # The property is "the loop is not blocked by the KWS step" (30 ms on every frame). Were it
    # blocked, nearly every tick would be late; a loaded CI runner can stall any single tick, so
    # the bound is on the 95th percentile, not the worst gap.
    gaps.sort()
    assert gaps[int(len(gaps) * 0.95)] < 0.03, gaps[-10:]


async def test_max_utterance_cap():
    cfg = Config()
    cfg.listen.max_utterance_s = 2.0
    kinds, events, _ = await run(audio((1.0, 1.6), (1.7, 5.0)), wake_at=1.6, cfg=cfg, trailing=1.0)
    # the cap finalizes the first part AND the rest keeps being captured as a continuation
    assert kinds == [K.WAKE, K.SPEECH_START, K.UTTERANCE, K.UTTERANCE]
    first, rest = events[2].utterance, events[3].utterance
    assert first.woke and not first.continuation and first.audio_s < 4.0  # 1.6 s look-back + 2 s cap
    assert rest.continuation and not rest.woke and rest.audio_s > 1.0


async def test_stop_word_while_speaking():
    """While Mac talks, short VAD segments are transcribed; only bare stop words come out."""
    speaking = lambda l: l.set_speaking(True, "Here is a long answer about the weather")
    kinds, events, _ = await run(audio((1.0, 1.5)), wake_at=None, stt=FakeSTT("okay stop"), before=speaking)
    assert kinds == [K.UTTERANCE]
    u = events[0].utterance
    assert u.window is WindowReason.SPEAKING and not u.woke and u.text == "okay stop"
    # not a stop word -> dropped silently
    kinds, _, _ = await run(audio((1.0, 1.5)), wake_at=None, stt=FakeSTT("what about tomorrow"), before=speaking)
    assert kinds == []
    # AEC off and Mac itself just said "stop" -> self-trigger, dropped
    kinds, _, _ = await run(audio((1.0, 1.5)), wake_at=None, stt=FakeSTT("stop"), before=lambda l: l.set_speaking(True, "I will stop here"))
    assert kinds == []
    # long speech while speaking is capped at 2 s per segment and still only yields stop words
    kinds, _, lis = await run(audio((1.0, 4.0)), wake_at=None, stt=FakeSTT("blah blah"), before=speaking)
    assert kinds == [] and lis.metrics.counters.get("stop_words", 0) == 0
    # playback ended -> no more stop-word captures (window closed by set_speaking(False))
    kinds, _, _ = await run(audio((1.0, 1.5)), wake_at=None, stt=FakeSTT("stop"), before=lambda l: (l.set_speaking(True, "x"), l.set_speaking(False)))
    assert kinds == []


class EventSink:
    def __init__(self):
        self.events = []

    def emit(self, type_, **fields):
        self.events.append((type_, fields))
        return True


async def test_wake_instrumentation_events_and_counters():
    sink = EventSink()
    src = FileInput(audio((1.0, 1.6), (1.7, 2.6)), trailing_silence_s=0.5)
    lis = ListenerImpl(src, FakeWake(int(1.6 * SAMPLE_RATE)), FakeVAD(), FakeSTT(), Config(), eventlog=sink)
    kinds = []

    async def collect():
        async for ev in lis.events():
            kinds.append(ev.kind)

    await asyncio.wait_for(asyncio.gather(lis.run(), collect()), 10)
    types = [t for t, _ in sink.events]
    assert types[:2] == ["kws_hit", "wake_verify"]
    assert sink.events[0][1]["phrase"] == "hey mac" and sink.events[1][1] == {
        "source": "file", "accepted": True, "transcript": "hey mac what time is it", "reason": "phrase_found"}
    c = lis.metrics.counters
    assert c["kws_hits"] == 1 and c["wakes"] == 1 and c.get("false_wakes", 0) == 0
    assert lis.metrics.latency("kws_hit_to_wake")["n"] == 1
    # rejected + suppressed hits are logged with reasons
    sink = EventSink()
    src = FileInput(audio((1.0, 1.6), (1.7, 2.6)), trailing_silence_s=0.5)
    lis = ListenerImpl(src, FakeWake(int(1.6 * SAMPLE_RATE)), FakeVAD(), FakeSTT("big mac"), Config(), eventlog=sink)
    lis.set_speaking(True, "I'm Mac")
    await asyncio.wait_for(asyncio.gather(lis.run(), collect()), 10)
    # (the last one: the stop-word capture of Mac's own voice, no echo cancellation)
    assert [t for t, _ in sink.events] == ["kws_hit", "wake_suppressed", "barge_capture"]
    assert sink.events[1][1]["reason"] == "self_trigger"
    assert sink.events[2][1]["aec"] is False and sink.events[2][1]["paused"] is False and sink.events[2][1]["probes"] == 0
    assert lis.metrics.counters["wakes_suppressed"] == 1


async def test_wake_right_after_mac_stops_speaking_and_during_followup():
    """The tail guard only blocks VAD captures; a wake word 0.5 s after playback ends is caught,
    and a wake hit inside an open FOLLOWUP window still takes the woke path."""
    def before(l):
        l.set_speaking(True, "Here is the answer")
        l.set_speaking(False)  # playback just ended at t=0: guard active for tail_guard_ms
        l.open_window(6.0, WindowReason.FOLLOWUP)

    kinds, events, lis = await run(audio((0.5, 1.1), (1.2, 2.1)), wake_at=1.1, before=before)
    # the follow-up window's VAD capture starts on "hey mac" itself (SPEECH_START), then the KWS
    # hit takes over: exactly one utterance, woke, not attributed to the window
    assert [k for k in kinds if k is not K.SPEECH_START] == [K.WAKE, K.UTTERANCE]
    assert events[-1].utterance.woke and events[-1].utterance.window is None
    assert lis._window_end is None  # the wake closed the follow-up window


def _speaking_aec(text="Here is a long answer about the weather"):
    return lambda l: (setattr(l, "_aec", True), l.set_speaking(True, text))


def _pause_times(lis):
    """Records the stream time (s) of every BARGE_IN as it is put on the listener's queue."""
    times, put = [], lis._queue.put_nowait

    def record(ev):
        if ev is not None and ev.kind is K.BARGE_IN:
            times.append(lis._pos / SAMPLE_RATE)
        put(ev)

    lis._queue.put_nowait = record
    return times


class GateSTT(FakeSTT):
    """The first ``hold`` calls wait for ``release``; ``texts`` are returned in call order (then
    the last one). ``started`` is set once the first call is waiting."""

    def __init__(self, *texts: str, hold: int = 1) -> None:
        super().__init__()
        self.texts, self.hold, self.n = list(texts), hold, 0
        self.release, self.started = asyncio.Event(), asyncio.Event()

    async def transcribe(self, pcm):
        i, self.n = self.n, self.n + 1
        self.calls.append(len(pcm) / SAMPLE_RATE)
        if i < self.hold:
            self.started.set()
            await self.release.wait()
        else:
            await asyncio.sleep(0)
        return self.texts[min(i, len(self.texts) - 1)]


async def test_barge_in_with_aec_pauses_on_words_then_passes_any_words():
    """With AEC, speech over Mac emits BARGE_IN once a probe hears words in it (it starts after
    barge_in_onset_ms of speech), then the full transcript (the orchestrator decides what it
    means); an empty transcript is a NO_SPEECH verdict."""
    times, stt = [], FakeSTT("what about tomorrow")
    kinds, events, _ = await run(audio((1.0, 2.0)), wake_at=None, stt=stt,
                                   before=lambda l: (_speaking_aec()(l), times.append(_pause_times(l))))
    assert kinds == [K.BARGE_IN, K.UTTERANCE]
    assert events[0].window is WindowReason.SPEAKING
    u = events[1].utterance
    assert u.window is WindowReason.SPEAKING and u.text == "what about tomorrow" and not u.woke
    # the probe starts at 1.3 s (300 ms of speech) and reads the capture so far: 0.3 s lead-in + 0.3 s
    assert len(stt.calls) == 2 and stt.calls[0] == pytest.approx(0.6)
    assert times[0] == [pytest.approx(1.3 + 0.06)]  # its answer is applied three frames later
    # onset: 300 ms of speech, so a 200 ms click is never even probed
    stt = FakeSTT("")
    kinds, _, _ = await run(audio((1.0, 1.2)), wake_at=None, stt=stt, before=_speaking_aec())
    assert K.BARGE_IN not in kinds and len(stt.calls) == 1
    # no STT model: the old rule (pause on the sound), and a NO_SPEECH verdict (never stuck paused)
    kinds, _, _ = await run(audio((1.0, 1.5)), wake_at=None, stt=FakeSTT(missing=True), before=_speaking_aec())
    assert kinds == [K.BARGE_IN, K.NO_SPEECH]


async def test_a_sound_without_words_over_mac_never_pauses_it():
    """Live 14 to 18 Nov: most pauses were false alarms that resumed with "no words", each a gap
    of 0.6 to 2 s. A cough, a door, a fan: probed, nothing heard, no BARGE_IN; the verdict is
    still NO_SPEECH. Probes thin out (400 ms of sound after the first, then 800, then 1600)."""
    stt = FakeSTT("")
    kinds, events, _ = await run(audio((1.0, 1.5)), wake_at=None, stt=stt, before=_speaking_aec())
    assert kinds == [K.NO_SPEECH] and events[0].window is WindowReason.SPEAKING
    stt = FakeSTT("")
    kinds, _, _ = await run(audio((1.0, 6.0), total_s=8.0), wake_at=None, stt=stt, before=_speaking_aec())
    assert kinds == [K.NO_SPEECH]
    # probes after 0.3, 0.7, 1.5, 3.1 and 4.7 s of sound, each on at most the last 4 s; then the final pass
    assert stt.calls[:-1] == pytest.approx([0.6, 1.0, 1.8, 3.4, 4.0])
    assert stt.calls[-1] > 5.0


async def test_a_backchannel_over_mac_does_not_pause_it_but_words_after_it_do():
    """ "Yeah" while Mac talks is following along: no pause (the transcript still goes out and the
    orchestrator drops it). "Yeah, but wait" is an interruption once the words are there."""
    kinds, events, _ = await run(audio((1.0, 1.6)), wake_at=None, stt=FakeSTT("mm-hmm"), before=_speaking_aec())
    assert kinds == [K.UTTERANCE] and events[0].utterance.window is WindowReason.SPEAKING
    stt = SeqSTT("yeah", "yeah but wait", "yeah but wait that is wrong")
    kinds, events, _ = await run(audio((1.0, 3.0)), wake_at=None, stt=stt, before=_speaking_aec())
    assert kinds == [K.BARGE_IN, K.UTTERANCE] and events[1].utterance.text == "yeah but wait that is wrong"


async def test_one_or_two_words_do_not_pause_mac_until_more_follow():
    """Half a second of noise or leaked echo often transcribes as "So" or "You're": too little to
    stop Mac for. The start of a sentence is heard again 200 ms later; "stop" and "wait" need no more."""
    times, stt = [], SeqSTT("can you", "can you repeat", "can you repeat that")
    kinds, events, _ = await run(audio((1.0, 3.0)), wake_at=None, stt=stt,
                                   before=lambda l: (_speaking_aec()(l), times.append(_pause_times(l))))
    assert kinds == [K.BARGE_IN, K.UTTERANCE] and events[1].utterance.text == "can you repeat that"
    assert times[0] == [pytest.approx(1.5 + 0.06)]  # the probe after 200 ms more speech
    kinds, events, _ = await run(audio((1.0, 3.0)), wake_at=None, stt=FakeSTT("you're"), before=_speaking_aec())
    assert kinds == [K.UTTERANCE]  # never paused; the orchestrator drops it as too short
    # Mac's own words with one stray one are still its echo, cut off mid-word
    kinds, _, _ = await run(audio((1.0, 3.0)), wake_at=None, stt=FakeSTT("here it is"),
                            before=_speaking_aec("Here is what I found about the weather."))
    assert kinds == [K.UTTERANCE]
    for word in ("stop", "wait", "okay stop please", "hold on"):
        times = []
        kinds, _, _ = await run(audio((1.0, 1.6)), wake_at=None, stt=FakeSTT(word),
                                before=lambda l, times=times: (_speaking_aec()(l), times.append(_pause_times(l))))
        assert kinds == [K.BARGE_IN, K.UTTERANCE] and times[0] == [pytest.approx(1.3 + 0.06)], word


async def test_words_after_an_empty_probe_pause_mac_once():
    """ "Uh" (nothing heard) and then words: the second probe hears them. One BARGE_IN only."""
    times, stt = [], SeqSTT("", "wait that is", "wait that is wrong")
    kinds, events, _ = await run(audio((1.0, 3.0)), wake_at=None, stt=stt,
                                   before=lambda l: (_speaking_aec()(l), times.append(_pause_times(l))))
    assert kinds == [K.BARGE_IN, K.UTTERANCE] and events[1].utterance.text == "wait that is wrong"
    assert times[0] == [pytest.approx(1.7 + 0.06)]  # the probe after 400 ms more speech
    assert len(stt.calls) == 3  # nothing is probed once Mac has paused


async def test_macs_own_words_leaking_through_do_not_pause_it():
    """Echo that the canceller let through transcribes as what Mac is saying: not an interruption.
    The last two items count (an echo trails the audio), and real words later still pause."""
    said = "Here is a long answer about the weather"
    kinds, _, _ = await run(audio((1.0, 2.0)), wake_at=None, stt=FakeSTT("a long answer about the weather"),
                            before=_speaking_aec(said))
    assert kinds == [K.UTTERANCE]  # the orchestrator's own echo rule handles the transcript

    def two_items(l):
        _speaking_aec(said)(l)
        l.set_speaking(True, "and tomorrow looks dry")  # the echo of the first sentence is still arriving

    kinds, _, _ = await run(audio((1.0, 2.0)), wake_at=None, stt=FakeSTT("answer about the weather"), before=two_items)
    assert kinds == [K.UTTERANCE]
    stt = SeqSTT("long answer", "long answer no stop that", "long answer no stop that is wrong")
    kinds, _, _ = await run(audio((1.0, 3.0)), wake_at=None, stt=stt, before=_speaking_aec(said))
    assert kinds == [K.BARGE_IN, K.UTTERANCE]


async def test_a_capture_that_ends_while_its_probe_is_running_does_not_pause_late():
    """Live audio does not wait for a probe. If the capture is finalised first, the late answer is
    dropped: the transcript alone goes to the orchestrator (which handles it without a pause)."""
    stt = GateSTT("stop that", "stop that")

    def live(l):
        _speaking_aec()(l)
        l._probe_wait = False  # as with the microphone
        orig = l._on_frame

        async def frame(f):
            await orig(f)
            if l._capture is None and stt.started.is_set():
                stt.release.set()  # the probe answers only after the capture is over

        l._on_frame = frame

    kinds, events, lis = await run(audio((1.0, 1.5)), wake_at=None, stt=stt, before=live)
    assert kinds == [K.UTTERANCE] and events[0].utterance.window is WindowReason.SPEAKING
    assert stt.n == 2 and not lis._jobs  # the probe did finish; its answer was not used


async def test_a_probe_answer_is_dropped_when_mac_stopped_speaking_or_the_listener_was_muted():
    for change in (lambda l: l.set_speaking(False), lambda l: l.set_muted(True)):
        stt = GateSTT("stop that", "stop that")

        def live(l, change=change, stt=stt):
            _speaking_aec()(l)
            l._probe_wait = False
            orig = l._on_frame

            async def frame(f):
                await orig(f)
                if stt.started.is_set() and not stt.release.is_set():
                    change(l)
                    stt.release.set()

            l._on_frame = frame

        kinds, _, _ = await run(audio((1.0, 2.0)), wake_at=None, stt=stt, before=live)
        assert K.BARGE_IN not in kinds, kinds


async def test_live_audio_pauses_when_the_probe_answers_without_holding_up_frames():
    """With a paced source the frame path never waits for STT: frames keep flowing while the probe
    runs, and BARGE_IN goes out on the first frame after its answer."""
    stt = GateSTT("stop that", "stop that right now")
    times = []

    def live(l):
        _speaking_aec()(l)
        l._probe_wait = False
        times.append(_pause_times(l))
        orig = l._on_frame

        async def frame(f):
            await orig(f)
            if l._pos == int(1.8 * SAMPLE_RATE):
                stt.release.set()  # a slow probe: half a second

        l._on_frame = frame

    kinds, _, lis = await run(audio((1.0, 3.0)), wake_at=None, stt=stt, before=live)
    assert kinds == [K.BARGE_IN, K.UTTERANCE]
    assert 1.8 < times[0][0] <= 1.86 and len(times[0]) == 1
    assert stt.n == 2  # one probe in flight at a time: none was started while the first one ran
    assert lis.frame_ms_max < 20


def _at(l, *steps):
    """Runs each (time_s, fn) once, right after the frame that ends at that stream time."""
    todo, orig = list(steps), l._on_frame

    async def frame(f):
        await orig(f)
        while todo and l._pos >= round(todo[0][0] * SAMPLE_RATE):
            todo.pop(0)[1](l)

    l._on_frame = frame


async def test_mac_starting_an_item_into_an_open_capture_asks_stt_at_once():
    """An item that starts while a capture with speech is open (the user paused to think) is
    probed right then, not after another 300 ms of speech: words pause Mac within a few frames
    of its first sound; a capture that holds only noise does not pause it at all."""
    def before(times, sink):
        return lambda l: (setattr(l, "_aec", True), setattr(l, "eventlog", sink), l.open_window(3600, WindowReason.TURN),
                          times.append(_pause_times(l)),
                          _at(l, (2.4, lambda l: l.set_speaking(True, "The best people to connect with"))))

    times, sink = [], EventSink()
    kinds, events, _ = await run(audio((1.0, 2.0), (2.6, 3.4)), wake_at=None, turn=FakeTurn(0.05, 0.9),
                                   stt=FakeSTT("you can get my profile from the resume folder"), before=before(times, sink))
    assert kinds == [K.SPEECH_START, K.BARGE_IN, K.UTTERANCE] and events[-1].utterance.window is WindowReason.SPEAKING
    assert 2.4 < times[0][0] <= 2.48 and len(times[0]) == 1  # before the user's next word at 2.6 s
    (ev,) = [f for t, f in sink.events if t == "barge_capture"]
    assert ev["began"] == "item_start" and ev["paused"] and ev["probes"] == 1 and 40 <= ev["pause_after_ms"] <= 80

    times, stt = [], FakeSTT("")
    kinds, _, _ = await run(audio((1.0, 2.0), (2.6, 3.4)), wake_at=None, turn=FakeTurn(0.05, 0.9), stt=stt,
                              before=before(times, EventSink()))
    assert kinds == [K.SPEECH_START, K.NO_SPEECH] and times[0] == []
    assert stt.calls[0] == pytest.approx(1.7, abs=0.05)  # probed at 2.4 s: the capture since 0.7 s


async def test_an_unconfirmed_barge_capture_is_probed_again_when_the_next_item_starts():
    """Mac's sentence ends while the first probe is still running (its answer is dropped: nothing
    to pause), the user pauses, and the next sentence starts into that pause."""
    stt, times = GateSTT("so", "so what about", "so what about tomorrow"), []

    def before(l):
        _speaking_aec()(l)
        l._probe_wait = False
        times.append(_pause_times(l))
        _at(l, (1.4, lambda l: l.set_speaking(False)), (1.44, lambda l: stt.release.set()),
            (2.0, lambda l: l.set_speaking(True, "It will rain in the morning")))

    kinds, _, _ = await run(audio((1.0, 1.5), (2.6, 3.0)), wake_at=None, stt=stt, turn=FakeTurn(0.05, 0.9), before=before)
    assert kinds == [K.BARGE_IN, K.UTTERANCE]
    assert 2.0 < times[0][0] <= 2.1 and len(times[0]) == 1


async def test_every_speaking_capture_logs_its_numbers_and_never_any_text():
    sink = EventSink()
    stt = SeqSTT("", "wait that is", "wait that is wrong")
    await run(audio((1.0, 3.0)), wake_at=None, stt=stt, before=lambda l: (_speaking_aec()(l), setattr(l, "eventlog", sink)))
    (ev,) = [f for t, f in sink.events if t == "barge_capture"]
    assert {k for k, v in ev.items() if isinstance(v, str)} == {"source", "began"}  # the rest are numbers, flags
    assert set(ev["probe_heard"]) <= {"none", "echo", "backchannel", "few", "words"}  # and categories, never the words
    assert ev.pop("probe_ms") >= 0
    assert ev == {
        "source": "file", "aec": True, "barge": True, "paused": True, "began": "onset",
        "capture_ms": 3000,  # 0.3 s lead-in, 2 s of speech, 0.7 s of silence
        "voiced_ms": 2000, "vad_max": 1.0, "vad_mean": pytest.approx(0.73, abs=0.02),
        "peak_dbfs": -6.0, "rms_dbfs": pytest.approx(-7.2, abs=0.1),
        "probes": 2, "probe_heard": ["none", "words"],
        "pause_after_ms": 760,  # second probe after 0.7 s of speech, applied 60 ms later
        "words": 4,
    }
    # a sound with no words, and a capture without echo cancellation: still one event each
    sink = EventSink()
    await run(audio((1.0, 1.5)), wake_at=None, stt=FakeSTT(""), before=lambda l: (_speaking_aec()(l), setattr(l, "eventlog", sink)))
    (ev,) = [f for t, f in sink.events if t == "barge_capture"]
    assert not ev["paused"] and ev["pause_after_ms"] is None and ev["words"] == 0 and ev["probes"] == 1
    sink = EventSink()
    await run(np.zeros(SAMPLE_RATE * 2, np.float32), wake_at=None, before=lambda l: (_speaking_aec()(l), setattr(l, "eventlog", sink)))
    assert sink.events == []  # no capture, no event


async def test_barge_in_confirm_off_pauses_on_the_onset_as_before():
    cfg = Config()
    cfg.listen.barge_in_confirm = False
    times, stt = [], FakeSTT("")
    kinds, events, _ = await run(audio((1.0, 1.5)), wake_at=None, stt=stt, cfg=cfg,
                                   before=lambda l: (_speaking_aec()(l), times.append(_pause_times(l))))
    assert kinds == [K.BARGE_IN, K.NO_SPEECH] and events[1].window is WindowReason.SPEAKING
    assert times[0] == [pytest.approx(1.3)] and len(stt.calls) == 1  # at 300 ms of sound, nothing probed
    kinds, _, _ = await run(audio((1.0, 1.2)), wake_at=None, stt=FakeSTT(""), cfg=cfg, before=_speaking_aec())
    assert K.BARGE_IN not in kinds


async def test_barge_in_capture_is_a_full_utterance_and_survives_playback_end():
    # 3 s of talking over Mac is one utterance (not cut at the 2 s stop-word cap)
    kinds, events, _ = await run(audio((1.0, 4.0)), wake_at=None, stt=FakeSTT("tell me something else"), before=_speaking_aec())
    assert kinds == [K.BARGE_IN, K.UTTERANCE] and events[1].utterance.audio_s > 3.0

    def stop_mid(l):
        _speaking_aec()(l)
        orig = l._on_frame

        async def frame(f):
            await orig(f)
            if l._pos == int(1.5 * SAMPLE_RATE):
                l.set_speaking(False)  # Mac's sentence ended while the user was still talking

        l._on_frame = frame

    kinds, events, _ = await run(audio((1.0, 2.0)), wake_at=None, stt=FakeSTT("and another thing"), before=stop_mid)
    assert kinds == [K.BARGE_IN, K.UTTERANCE] and events[1].utterance.text == "and another thing"


async def test_barge_in_disabled_keeps_stop_word_only_capture():
    cfg = Config()
    cfg.listen.barge_in = False
    kinds, _, _ = await run(audio((1.0, 1.5)), wake_at=None, stt=FakeSTT("what about tomorrow"), cfg=cfg,
                            before=_speaking_aec())
    assert kinds == []


async def test_turn_window_stays_open_across_utterances():
    kinds, events, _ = await run(audio((1.0, 1.5), (3.0, 3.5)), wake_at=None, stt=FakeSTT("and on Friday"),
                                 before=lambda l: l.open_window(3600, WindowReason.TURN))
    assert kinds == [K.SPEECH_START, K.UTTERANCE, K.SPEECH_START, K.UTTERANCE]
    assert all(e.utterance.window is WindowReason.TURN for e in events if e.utterance)


# ----------------------------------------------------------------------------- Smart Turn (semantic end of turn)


class FakeTurn:
    """Returns the queued probabilities in order (then 1.0); records the audio it was shown."""

    def __init__(self, *probs: float, fail: bool = False) -> None:
        self.probs, self.fail, self.seen = list(probs), fail, []

    def complete_prob(self, pcm):
        self.seen.append(len(pcm) / SAMPLE_RATE)
        if self.fail:
            raise RuntimeError("onnx blew up")
        return self.probs.pop(0) if self.probs else 1.0


def _window(l):
    l.open_window(3600, WindowReason.TURN)


async def test_smart_turn_not_finished_waits_through_the_pause():
    """"give me a fun fact about ... octopuses": the model says the first part is unfinished, so
    the 1.2 s pause (longer than silence_ms) does not end the turn: one utterance, both parts."""
    turn, stt = FakeTurn(0.05, 0.9), FakeSTT("give me a fun fact about octopuses")
    kinds, _, _ = await run(audio((1.0, 2.0), (3.2, 3.8)), wake_at=None, stt=stt, turn=turn, before=_window)
    assert kinds == [K.SPEECH_START, K.UTTERANCE]
    assert len(turn.seen) == 2 and stt.calls == [pytest.approx(3.8 + 0.3 - 0.7, abs=0.05)]  # one STT pass over both


async def test_smart_turn_finished_ends_the_turn_before_silence_ms():
    """A complete sentence ends after turn_check_ms (+3 frames), not silence_ms: a 0.5 s gap that
    the 700 ms silence rule would have merged is two turns."""
    kinds, _, _ = await run(audio((1.0, 1.5), (2.0, 2.5)), wake_at=None, stt=FakeSTT("yes"), turn=FakeTurn(0.9, 0.9), before=_window)
    assert kinds == [K.SPEECH_START, K.UTTERANCE, K.SPEECH_START, K.UTTERANCE]
    kinds, _, _ = await run(audio((1.0, 1.5), (2.0, 2.5)), wake_at=None, stt=FakeSTT("yes"), before=_window)
    assert kinds == [K.SPEECH_START, K.UTTERANCE]  # no model: silence_ms decides


async def test_smart_turn_not_finished_but_nothing_follows_ends_at_max_silence():
    turn = FakeTurn(0.1)
    kinds, _, _ = await run(audio((1.0, 2.0), total_s=5.0), wake_at=None, stt=FakeSTT("tell me about"), turn=turn, before=_window)
    assert kinds == [K.SPEECH_START, K.UTTERANCE] and len(turn.seen) == 1
    cfg = Config()
    cfg.listen.turn_max_silence_ms = 5000  # longer than the input: only end-of-input finalises it
    _, events, _ = await run(audio((1.0, 2.0), total_s=3.5), wake_at=None, stt=FakeSTT("tell me about"), cfg=cfg,
                             turn=FakeTurn(0.1), before=_window, trailing=0.0)
    assert [e.kind for e in events][-1] is K.UTTERANCE


async def test_smart_turn_disabled_or_failing_falls_back_to_silence():
    cfg = Config()
    cfg.listen.smart_turn = False
    turn = FakeTurn(0.0)
    kinds, _, _ = await run(audio((1.0, 1.5), (2.0, 2.5)), wake_at=None, stt=FakeSTT("yes"), cfg=cfg, turn=turn, before=_window)
    assert kinds == [K.SPEECH_START, K.UTTERANCE] and turn.seen == []
    kinds, _, _ = await run(audio((1.0, 2.0)), wake_at=None, stt=FakeSTT("yes"), turn=FakeTurn(fail=True), before=_window)
    assert kinds == [K.SPEECH_START, K.UTTERANCE]


async def test_smart_turn_skips_no_aec_stop_word_captures():
    turn = FakeTurn(0.0)
    kinds, _, _ = await run(audio((1.0, 1.4), total_s=4.0), wake_at=None, stt=FakeSTT("stop"), turn=turn,
                                 before=lambda l: l.set_speaking(True, "a long answer"))
    assert K.UTTERANCE in kinds and turn.seen == []



async def test_mac_starting_to_speak_mid_utterance_turns_it_into_a_barge_in():
    """Live 2026-11-11: the user was still talking (Smart Turn held a thinking pause open) when
    the reply started; the open capture was a TURN capture, so Mac never paused and talked over
    them. Now the capture becomes a barge-in: BARGE_IN on the next voiced frames, window SPEAKING."""
    lis_box = []

    class SpeakAt(FakeVAD):
        spoke = False

        def accept(self, frame):
            if not SpeakAt.spoke and lis_box[0]._pos >= 2.4 * SAMPLE_RATE:  # Mac's reply starts in the pause
                SpeakAt.spoke = True
                lis_box[0].set_speaking(True, "The best people to connect with")
            return super().accept(frame)

    cfg = Config()
    src = FileInput(audio((1.0, 2.0), (2.6, 3.4), total_s=6.0), trailing_silence_s=0.5)
    lis = ListenerImpl(src, None, SpeakAt(), FakeSTT("you can get my profile from the resume folder"), cfg,
                       aec=True, turn=FakeTurn(0.05, 0.9))
    lis_box.append(lis)
    lis.open_window(3600, WindowReason.TURN)
    events = []

    async def collect():
        async for ev in lis.events():
            events.append(ev)

    await asyncio.wait_for(asyncio.gather(lis.run(), collect()), 10)
    kinds = [e.kind for e in events]
    assert kinds == [K.SPEECH_START, K.BARGE_IN, K.UTTERANCE], kinds
    (utt,) = [e.utterance for e in events if e.utterance]
    assert utt.window is WindowReason.SPEAKING and "resume folder" in utt.text


async def test_mid_utterance_speech_start_without_aec_keeps_the_turn_capture():
    lis_box = []

    class SpeakAt(FakeVAD):
        spoke = False

        def accept(self, frame):
            if not SpeakAt.spoke and lis_box[0]._pos >= 1.5 * SAMPLE_RATE:
                SpeakAt.spoke = True
                lis_box[0].set_speaking(True, "x")
            return super().accept(frame)

    src = FileInput(audio((1.0, 2.0), total_s=4.0), trailing_silence_s=0.5)
    lis = ListenerImpl(src, None, SpeakAt(), FakeSTT("hello there"), Config(), aec=False)
    lis_box.append(lis)
    lis.open_window(3600, WindowReason.TURN)
    events = []

    async def collect():
        async for ev in lis.events():
            events.append(ev)

    await asyncio.wait_for(asyncio.gather(lis.run(), collect()), 10)
    assert K.BARGE_IN not in [e.kind for e in events]
    assert [e.utterance.window for e in events if e.utterance] == [WindowReason.TURN]


async def test_mac_speaking_over_a_follow_on_after_the_wake_command_is_a_barge_in():
    """Short question, fast answer: the wake capture is still open (5 s no-speech deadline) when
    the user goes on and Mac's reply starts. The follow-on becomes a barge-in, not a talk-over."""
    lis_box = []
    resumed = asyncio.Event()  # the user's follow-on has begun (3.0 s): the first STT returns after it

    class SpeakAt(FakeVAD):
        spoke = False

        def accept(self, frame):
            t = lis_box[0]._pos / SAMPLE_RATE
            if t >= 3.2:
                resumed.set()
            if not SpeakAt.spoke and t >= 3.5:
                SpeakAt.spoke = True
                lis_box[0].set_speaking(True, "The capital of France is Paris.")
            return super().accept(frame)

    stt = FakeSTT("hey mac what is the capital of france")
    orig = stt.transcribe

    async def transcribe(pcm):
        n = len(stt.calls)  # call 0 verifies the wake phrase, call 1 is the command, call 2 the follow-on
        if n == 1:
            await resumed.wait()  # Parakeet is still on the command when the user goes on
        if n >= 2:
            return "and also tell me about germany"
        return await orig(pcm)

    stt.transcribe = transcribe  # type: ignore[method-assign]
    src = FileInput(audio((1.0, 2.0), (3.0, 4.2), total_s=7.0), trailing_silence_s=0.5)
    lis = ListenerImpl(src, FakeWake(int(1.5 * SAMPLE_RATE)), SpeakAt(), stt, Config(), aec=True, turn=FakeTurn(0.9, 0.9))
    lis_box.append(lis)
    events = []

    async def collect():
        async for ev in lis.events():
            events.append(ev)

    await asyncio.wait_for(asyncio.gather(lis.run(), collect()), 10)
    kinds = [e.kind for e in events]
    assert kinds[:3] == [K.WAKE, K.SPEECH_START, K.UTTERANCE] and kinds[-1] is K.UTTERANCE, kinds
    assert sorted(k.value for k in kinds[3:-1]) == ["barge_in", "speech_start"], kinds  # BARGE_IN bypasses the STT chain
    first, second = [e.utterance for e in events if e.utterance]
    assert first.woke and first.window is None
    assert not second.woke and second.window is WindowReason.SPEAKING and "germany" in second.text


async def test_barge_in_repeats_at_each_item_start_while_the_user_talks_on():
    """Live G2: the first BARGE_IN landed on a thinking earcon (system speech, not paused by the
    orchestrator); the answer that started a few seconds later then played over the user. The
    converted capture re-sends BARGE_IN at every item start while it is open."""
    lis_box = []

    class SpeakAt(FakeVAD):
        starts = 0

        def accept(self, frame):
            t = lis_box[0]._pos / SAMPLE_RATE
            if (SpeakAt.starts == 0 and t >= 1.5) or (SpeakAt.starts == 1 and t >= 2.2):  # earcon, then the answer
                SpeakAt.starts += 1
                lis_box[0].set_speaking(True, "")
            return super().accept(frame)

    src = FileInput(audio((1.0, 3.0), total_s=5.0), trailing_silence_s=0.5)
    lis = ListenerImpl(src, None, SpeakAt(), FakeSTT("and the number for the second one"), Config(), aec=True, turn=FakeTurn(0.9))
    lis_box.append(lis)
    lis.open_window(3600, WindowReason.TURN)
    events = []

    async def collect():
        async for ev in lis.events():
            events.append(ev)

    await asyncio.wait_for(asyncio.gather(lis.run(), collect()), 10)
    kinds = [e.kind for e in events]
    assert kinds == [K.SPEECH_START, K.BARGE_IN, K.BARGE_IN, K.UTTERANCE], kinds
    assert events[-1].utterance.window is WindowReason.SPEAKING


# ----------------------------------------------------------------------------- empty captures in a window


class SeqSTT(FakeSTT):
    """Returns the queued transcripts in order (then the last one); ``on_call`` runs before each."""

    def __init__(self, *texts: str, on_call=None) -> None:
        super().__init__()
        self.texts, self.on_call = list(texts), on_call

    async def transcribe(self, pcm):
        await asyncio.sleep(0)
        self.calls.append(len(pcm) / SAMPLE_RATE)
        if self.on_call:
            self.on_call(len(self.calls))
        return self.texts.pop(0) if len(self.texts) > 1 else self.texts[0]


async def test_a_cough_in_the_followup_window_does_not_end_it():
    """A capture that transcribes empty (cough, click, echo tail) used to close the follow-up
    window, so the real answer a moment later needed "Hey Mac" again."""
    kinds, events, _ = await run(audio((0.5, 0.8), (2.5, 3.5)), wake_at=None, stt=SeqSTT("", "yes please"),
                                 before=lambda l: l.open_window(4.0, WindowReason.FOLLOWUP))
    assert kinds == [K.SPEECH_START, K.SPEECH_START, K.UTTERANCE]
    assert events[-1].utterance.text == "yes please" and events[-1].utterance.window is WindowReason.FOLLOWUP


async def test_a_cough_in_the_confirm_window_does_not_answer_it():
    kinds, events, _ = await run(audio((0.5, 0.8), (2.0, 2.6)), wake_at=None, stt=SeqSTT("", "yes"),
                                 before=lambda l: l.open_window(4.0, WindowReason.CONFIRM))
    assert kinds == [K.SPEECH_START, K.SPEECH_START, K.UTTERANCE]
    assert events[-1].utterance.window is WindowReason.CONFIRM


async def test_a_reopened_window_still_expires_on_time():
    kinds, events, lis = await run(audio((0.5, 0.8), total_s=4.0), wake_at=None, stt=SeqSTT(""),
                                   before=lambda l: l.open_window(2.0, WindowReason.FOLLOWUP))
    assert kinds == [K.SPEECH_START, K.NO_SPEECH] and events[-1].window is WindowReason.FOLLOWUP
    assert lis._window_end is None


async def test_a_window_closed_meanwhile_is_not_reopened():
    holder = {}

    def close(_n):
        holder["lis"].close_window()  # the orchestrator moved on while the cough was transcribed

    def before(l):
        holder["lis"] = l
        l.open_window(4.0, WindowReason.FOLLOWUP)

    kinds, _, _ = await run(audio((0.5, 0.8), (2.5, 3.5)), wake_at=None, stt=SeqSTT("", "yes please", on_call=close), before=before)
    assert kinds == [K.SPEECH_START, K.NO_SPEECH]


async def test_an_empty_push_to_talk_capture_still_ends_the_window():
    kinds, _, _ = await run(audio((0.5, 0.8), (2.5, 3.5)), wake_at=None, stt=SeqSTT("", "hello"),
                            before=lambda l: l.open_window(4.0, WindowReason.PTT))
    assert kinds == [K.SPEECH_START, K.NO_SPEECH]
