import Foundation

/// Notices when the main thread stops answering and records what it was doing.
///
/// Everything the menu bar shows runs on the main thread, and a hang there leaves no trace of
/// its own. This asks the main thread for a reply once a second from its own queue; no reply for
/// `limit` seconds (noticed at the next tick, so 2 to 3 s in) is a stall: one line in app.log
/// when it is noticed, one when it ends (with its length), and a stack sample of this process in
/// `<logs>/stalls/` (function names only, never content).
final class MainThreadWatchdog: @unchecked Sendable {
    private let queue = DispatchQueue(label: "mac.watchdog", qos: .utility)
    private let limit: TimeInterval = 2
    private let sampleEvery: TimeInterval = 300  // a sample pauses the process briefly: not often
    private let keepSamples = 8
    // All below: on `queue`.
    private var timer: DispatchSourceTimer?
    private var asked: DispatchTime?  // the question the main thread has not answered yet
    private var reported = false
    private var lastSample: DispatchTime?

    func start() {
        queue.sync {
            let t = DispatchSource.makeTimerSource(queue: queue)
            t.schedule(deadline: .now() + 1, repeating: 1, leeway: .milliseconds(100))
            t.setEventHandler { [weak self] in self?.tick() }
            t.resume()
            timer = t
        }
    }

    /// At quit: what the main thread does from here on is no longer watched.
    func stop() {
        queue.sync { timer?.cancel(); timer = nil }
    }

    private static func seconds(since t: DispatchTime) -> TimeInterval {
        // Uptime clock: it stands still while the computer sleeps, so sleep is never a stall.
        Double(DispatchTime.now().uptimeNanoseconds - t.uptimeNanoseconds) / 1e9
    }

    private func tick() {
        guard let since = asked else {
            let at = DispatchTime.now()
            asked = at
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.queue.async { self.answered(at) }
            }
            return
        }
        guard !reported, Self.seconds(since: since) >= limit else { return }
        reported = true
        Log.warn("main thread is not responding", "watchdog")
        if lastSample.map({ Self.seconds(since: $0) >= sampleEvery }) ?? true {
            lastSample = .now()
            sample()
        }
    }

    private func answered(_ at: DispatchTime) {
        asked = nil
        guard reported else { return }
        reported = false
        Log.warn(String(format: "main thread is back after %.1f s", Self.seconds(since: at)), "watchdog")
    }

    /// `/usr/bin/sample <pid> 1 50`: one second, a stack every 50 ms, written by `sample` itself.
    private func sample() {
        let dir = Paths.logDir.appendingPathComponent("stalls")
        let fm = FileManager.default
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        let old = ((try? fm.contentsOfDirectory(atPath: dir.path)) ?? []).filter { $0.hasPrefix("stall-") }.sorted()
        for name in old.dropLast(keepSamples - 1) { try? fm.removeItem(at: dir.appendingPathComponent(name)) }
        let stamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "")
        let out = dir.appendingPathComponent("stall-\(stamp).txt")
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/sample")
        p.arguments = ["\(ProcessInfo.processInfo.processIdentifier)", "1", "50", "-file", out.path]
        p.standardOutput = FileHandle.nullDevice
        p.standardError = FileHandle.nullDevice
        p.terminationHandler = { p in
            Log.info(p.terminationStatus == 0 ? "stack sample saved: stalls/\(out.lastPathComponent)"
                                              : "stack sample failed (status \(p.terminationStatus))", "watchdog")
        }
        do { try p.run() } catch { Log.warn("stack sample not started: \(error.localizedDescription)", "watchdog") }
    }
}
