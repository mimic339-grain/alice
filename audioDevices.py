"""Resolve audio devices by name instead of by index.

PyAudio's device indices are not stable. A reboot on this machine moved the DirectSound mic
array from index 6 to index 7, and nothing failed loudly: STT opened index 6 -- by then the
"primary sound capture driver" -- printed "STT Ready", and then transcribed nothing forever.
Matching on the device name survives that reshuffle.

Host API matters as much as the index. All of these are the same physical array mic, but only
some of them can actually deliver what RealtimeSTT asks for (16 kHz mono):

    MME          rms 13677   works
    DirectSound  rms   242   works, but ~50x quieter -- too quiet for Silero VAD to trigger
    WASAPI                   refuses 16000 Hz outright
    WDM-KS                   refuses to open at all

So the preference order below is deliberate, not arbitrary.
"""

import sys

import pyaudio


def _safe(text):
    """Make text printable on a cp932 console.

    Device names here contain characters the Windows console codepage cannot encode (the
    registered-trademark sign in "インテル®", for one). Printing one raw kills the STT thread
    at startup with a UnicodeEncodeError, so anything headed for stdout goes through this.
    """
    enc = sys.stdout.encoding or "utf-8"
    return text.encode(enc, errors="replace").decode(enc, errors="replace")


# Ordered best-first. Names as PyAudio reports them.
_HOST_API_PREFERENCE = ["MME", "Windows DirectSound", "Windows WASAPI", "Windows WDM-KS"]


def _host_api_rank(name):
    try:
        return _HOST_API_PREFERENCE.index(name)
    except ValueError:
        return len(_HOST_API_PREFERENCE)


def list_input_devices(pa):
    """Every device that can record, with its host API resolved to a name."""
    out = []
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            api = pa.get_host_api_info_by_index(info["hostApi"])["name"]
            out.append((i, api, info["name"]))
    return out


def resolve_input_device(name_preferences, explicit_index=None):
    """Return the index of the input device to record from.

    name_preferences is an ordered list of name fragments, best first, so a USB mic can be
    picked up automatically the moment it is plugged in without editing any code. The first
    fragment that matches a present device wins; if none match, PyAudio's system default is
    used, which beats guessing at an index that may now belong to something else.

    An explicit index overrides all of it, for the case where a machine really does need one
    specific device pinned.
    """
    if explicit_index is not None:
        return explicit_index

    if isinstance(name_preferences, str):
        name_preferences = [name_preferences]

    pa = pyaudio.PyAudio()
    try:
        devices = list_input_devices(pa)
        for fragment in name_preferences or []:
            needle = fragment.casefold()
            matches = [d for d in devices if needle in d[2].casefold()]
            if not matches:
                continue
            matches.sort(key=lambda d: (_host_api_rank(d[1]), d[0]))
            index, api, full_name = matches[0]
            print(_safe(f"AUDIO: input device -> [{index}] {api} / {full_name}"))
            return index

        print(_safe(
            f"AUDIO: none of {name_preferences} matched an input device. Falling back to the "
            f"system default. Available: {[(i, a, n[:30]) for i, a, n in devices]}"))
        return None
    finally:
        pa.terminate()


def check_input_health(device_index, seconds=1.0, rate=16000):
    """Warn at startup if the chosen mic is not delivering anything speech-shaped.

    This machine's internal array failed by delivering a full-scale sub-300 Hz rumble instead of
    audio. Every layer reported success -- the device was ACTIVE, unmuted, at 87% level, the
    driver was fine, PyAudio opened it, and frames arrived on schedule -- so STT printed
    "Ready" and then silently transcribed nothing, which reads as a code bug for hours.

    Speech puts most of its energy between 300 Hz and 3 kHz. One second at startup is enough to
    tell "quiet room" (low level, normal spectrum) from "broken capture path" (high level, all
    of it below 300 Hz), so say which out loud instead of failing silently.
    """
    try:
        import numpy as np
    except ImportError:
        return

    pa = pyaudio.PyAudio()
    try:
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True,
                         input_device_index=device_index, frames_per_buffer=1024)
        frames = b''
        for _ in range(int(rate / 1024 * seconds)):
            frames += stream.read(1024, exception_on_overflow=False)
        stream.stop_stream()
        stream.close()
    except Exception as e:
        print(f"AUDIO: could not health-check the input device: {type(e).__name__}: {e}")
        pa.terminate()
        return
    pa.terminate()

    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return
    rms = float(np.sqrt((samples ** 2).mean()))

    spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
    freqs = np.fft.rfftfreq(len(samples), 1 / rate)
    total = spectrum.sum()
    if total <= 0:
        return
    speech_pct = float(spectrum[(freqs >= 300) & (freqs < 3000)].sum() / total * 100)
    rumble_pct = float(spectrum[freqs < 300].sum() / total * 100)

    # A loud signal that is nearly all sub-300 Hz is not a quiet room, it is a broken capture
    # path. A genuinely quiet room is quiet: low RMS, and no single band dominating this hard.
    # Digital silence, not a quiet room: a live microphone always picks up some noise floor.
    # Usually means the mic is muted -- the QuadCast mutes from a tap on its top plate, which
    # is easy to trigger by accident. This check previously reported "health OK (rms=0)",
    # because it only looked for the broken-array signature and never for nothing at all.
    if rms < 1.0 or int(np.abs(samples).max()) <= 2:
        print(f"AUDIO: WARNING -- this input is returning silence (rms={rms:.1f}). A live mic "
              f"always has some noise floor, so it is almost certainly MUTED or at zero gain. "
              f"On the QuadCast, tap the top plate to unmute (the LED is off when muted).")
    elif rms > 1000 and rumble_pct > 90:
        print(f"AUDIO: WARNING -- this input looks BROKEN, not quiet. rms={rms:.0f}, "
              f"{rumble_pct:.0f}% of its energy is below 300 Hz and only {speech_pct:.1f}% is "
              f"in the speech band. Silero VAD will never fire on this, so nothing will be "
              f"transcribed. Check for a different microphone.")
    else:
        print(f"AUDIO: input health OK (rms={rms:.0f}, speech band {speech_pct:.1f}%)")
