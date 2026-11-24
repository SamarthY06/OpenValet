# Changelog

## Unreleased

- Scheduled tasks and reminders: ask by voice or in a WhatsApp group ("every weekday at 8 am, find 20 companies that fit my resume and save them"; "remind me at 6 to call the bank") and the background service runs it at that time. Results go back where you asked: the group, or a Mac notification (ask "what did it find" for the full report). A task created in a group belongs to that group (one task per group) and shares its memory with what you tell it there. Runs are unattended: nothing that sends, posts, pays or deletes happens without you; such steps are listed in the report as waiting for your OK. If the Claude usage limit stops a run, you get one message with the reset time; "continue <task>" picks up where it left off, "stop <task>" stops it. Settings > Tasks lists, edits, runs, pauses and deletes tasks with their history.
- A model and an effort level per task: set them in the task's editor in Settings > Tasks or by asking ("run the job scan on Opus at high effort"). Left unset, a task uses your default model and Claude Code's default effort; a change applies from the next run.
- A scheduled run keeps a plugged-in Mac awake until it ends. On battery the Mac is left to sleep and the run carries on at wake.
- Fixed: in a scheduled run, the helper agents Claude starts could not hand their results back (each one was refused as a send).
- Fixed: a scheduled run that was cut by a Mac sleep, recovered and finished was still marked as failed and run a second time.
- Fixed: Mac's speech kept cutting out, with words missing, whenever the microphone picked up a sound while it talked (a noise in the room, its own voice coming back). It now pauses only once it has made out an interruption: "stop", "wait" or "hold on" at once, anything else after three words. A noise, a "yeah" or its own echo no longer pauses it. An interruption is noticed a little later than before (about half a second to a second after you start, instead of 0.3 s). `listen.barge_in_confirm = false` restores the old behaviour.
- Fixed: a request said with a pause in the middle was answered as two requests. The first half started an answer, and the rest, arriving once that answer was playing, replaced it as a new request on its own ("book a meeting with Sam", then "on Friday at three"). If you carry on before Mac's answer starts playing, your words are now the rest of the same request (unless Mac had already started acting on it): Mac drops the answer it had begun and answers the whole thing.
- Fixed: a reply you began over Mac's last words and finished after it stopped was dropped, so Mac sat silent. It is now taken as your follow-up.
- Fixed: Mac's own voice leaking back through the microphone, or someone talking in another room, could still be heard as a sentence and cut the answer off. A faint or distant sound no longer pauses or interrupts Mac, whatever words are made of it; "stop" still works. `listen.barge_in_min_confidence` (default 0.95; 0 = off) sets how clearly a sound must register as a voice; lower it (try 0.8) if you speak softly or sit far from the microphone and Mac ignores you.
- Fixed: the first words of an answer could be lost when echo cancelling was switched on late. The switch no longer waits for the app's interface, and speech already queued when the audio engine restarts (echo cancelling switching, another app taking the microphone) is carried over instead of dropped.
- The app notes in its log when its interface stops responding and saves a stack sample to `~/Library/Logs/Mac/stalls/`. One `barge_capture` line per sound heard over Mac's speech goes to the event log (numbers only, never the words), to show what set it off.
- Fixed: the model download now checks every file against its published SHA-256 before it is installed or extracted. A download cut short behind a normal response, an archive that does not extract or lacks a file, or a file that is not the published one now moves on to the next mirror (before, only timeouts and HTTP errors did). Mirrors are pinned to a commit, and a resumed download never continues another mirror's partial file. Thanks to @capitaharlock for the report.
- Fixed: "Hey Mac, stop the other task" did nothing when the background services were installed (the voice daemon had no token for the backend).
- A restart no longer loses the request in flight: the voice daemon and the chat backend pick a turn cut by a restart up again in the same session ("Back. Picking up where I left off."), if it was running within the last 30 minutes; older ones still ask.
- Claude Code's own permission classifier runs on every tool call (`brain.permission_mode = "auto"`, the default); what it escalates is put to the user as a confirmation and passes once they say yes. `"bypassPermissions"` restores the previous behaviour.
- `brain.confirm_outward` (default true): set it to false and the model decides when to ask before sending, posting or paying; the hook then blocks only dangerous actions (destructive commands, secrets, Mac's own files, the never-list).
- "Stop everything" (chat and voice): ends the running task, its background agents and the session at once.
- Voice: short progress lines every twenty to thirty seconds on long tasks, and mid-task questions answered at once.

## 0.1.0 - 2026-11-11

First public release, under the name OpenMac (renamed OpenValet the next day). The assistant is still called Mac.

### Voice

- Wake word ("Hey Mac", "OK Mac"), local speech recognition (Parakeet) and local voices
  (Kokoro, 54 voices in 8 languages; Kyutai Pocket TTS for cloning your own voice).
- Barge-in: talking over Mac pauses it within 300 ms. Backchannels and echoes let it resume;
  real words stop it, become the next request or a steer, and Claude is told what was and was
  not heard. "stop" always stops.
- Semantic end of turn with Pipecat Smart Turn v3.2: a pause mid-sentence does not end the
  turn; a finished sentence ends it in about 360 ms. Falls back to silence timing when the
  model is off or missing.
- Speech during a task reaches Claude without the wake word: merged into the request if
  nothing has been produced yet, otherwise as a marked steer.
- Follow-up window after each answer, push-to-talk (⌥Space), screen off and screen locked
  operation, memory in a plain `memory.md`, tutor mode, phone access over LiveKit.

### WhatsApp

- Linked device through a Go bridge (whatsmeow). `scripts/mac whatsapp link`, background
  services with `scripts/services.sh`, frozen per-commit snapshots with rollback.
- Owner-only triggers: messages starting with "Mac", or replies to Mac's messages. Other
  members' messages are context (last 50 messages, 8,000 characters), never instructions.
- Per-chat profiles: persona, folder, capabilities (`scripts/mac chats allow|deny|on|off|remove`).
- A never-list of chats Mac will never read or write (`never_chats`), checked before every
  message.
- Background tasks survive a new message during a report-back; the message joins the turn.

### Safety

- Rule-based gate in a `PreToolUse` hook: irreversible actions (send, post, pay, delete,
  upload) need a yes to a summary of that exact action; the token is bound to the action and
  expires in 120 seconds.
- Never-readable set (`~/.ssh`, keychains, browser profiles, `~/.aws` and similar). Tools can
  write only Mac's own workspace and memory under `~/.mac-agent/`; settings, tokens and the
  WhatsApp link are never readable or writable.
- Scripts are re-analysed as command lines only when they look like one, after the gate
  blocked harmless HTML-to-text helpers as credential reads.
- Security audit of 2026-11-10 folded into tests.

### Setup

- One-line install: clones into `~/OpenValet`, installs uv, Python, the speech models, builds
  and opens `OpenValet.app`. The Meet Mac window handles permissions and Chrome pairing.
- Go is optional: without it the WhatsApp bridge is downloaded prebuilt from the release.
- `./install.sh --uninstall`, `scripts/services.sh uninstall`.

### Known limits

- Apple Silicon only, macOS 14 or newer, English speech recognition.
- The gate is a filter, not a sandbox.
- WhatsApp is text and files only: no voice notes, polls or buttons.
