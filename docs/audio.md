# Audio

How titles make sound and how it reaches the Mac. `crates/hle/src/audio` (libSceAudioOut and the host side) and `crates/hle/src/ngs2` (libSceNgs2).

## The two paths

A PS5 title hands samples to the system in one of two ways:

- **AudioOut directly.** The title mixes its own audio, opens a port with `sceAudioOutOpen` and pushes one buffer at a time with `sceAudioOutOutput`, which blocks for about one buffer period. Titles use that blocking as their audio clock.
- **NGS2, then AudioOut.** The title builds a voice graph in libSceNgs2: sampler voices play waveform blocks, their output ports patch into submixer or mastering voices, and `sceNgs2SystemRender` mixes one grain (256 frames by default) into a buffer the title then passes to `sceAudioOutOutput`.

Balatro (LÖVE) uses the second path: one stereo S16 port of 256 frames, one NGS2 system, a sampler rack (id `0x1000`, up to 128 voices, one per sound) and a mastering rack (id `0x3000`, one stereo voice). Every sound is a 16-bit stereo or mono WAVE file at 48 kHz in `resources/sounds`. Short effects are queued as one block covering the whole file; the five music tracks are streamed in 7 to 16 KiB blocks with the "continue" flag. All of this comes from tracing Balatro's calls (`PS5SLOP_NGS2_TRACE=1`), not from Sony's code.

## What is implemented

**AudioOut** (`audio/mod.rs`): `Init`, `Open`, `Output`, `Outputs`, `Close`, `SetVolume`. Formats 0 to 7: S16 and float, mono, stereo and 8 channels (8-channel buffers are folded to stereo with the centre and surround channels at -3 dB). `SetVolume` takes one volume per channel bit, 32768 being unity. `Output` returns the port's frame count, as SharpEmu-mac does. Every port's buffer is converted to stereo float at 48 kHz and goes to:

- the host's default output device, through an AudioToolbox output queue (`audio/coreaudio.rs`, our own small FFI, no new dependency): three 512-frame float buffers, filled on the queue's own thread from one queue per port, summed. A port's writer waits while more than 2048 frames (43 ms) are queued for it, so the device's clock paces the title. If the device stops taking samples for 250 ms, the excess is dropped and the writer carries on;
- without a device, `Output` sleeps until the previous buffer would have finished playing (the console's pace by the host's clock), and the samples are dropped;
- in both cases, a WAVE dump when `PS5SLOP_AUDIO_WAV=path` is set: 16-bit stereo, the first port to `path`, further ports to `<stem>-<port>.wav` beside it. The header is rewritten after every buffer, so the file is valid however the process ends.

The device opens with the first port when the window is shown. `--headless` runs keep it closed. `PS5SLOP_AUDIO=1` opens it anyway and `PS5SLOP_AUDIO=0` keeps it closed.

**NGS2** (`ngs2/`), ported from SharpEmu-mac's NGS2 library:

- Systems (grain size and rate from the option struct, `SetGrainSamples`, `SetSampleRate`), racks, voices. Handles are small numbers from one counter. `RackGetVoiceHandle` gives the same voice for the same index.
- Voice parameter lists (`VoiceControl`) and 16-byte command records (`VoiceRunCommands`): waveform format, block queue (append, replace, join), pitch, port volume, level matrices, port patches (to another voice, a rack's voice 0, or a render buffer), events (play, stop, kill, pause, resume), callbacks, mastering channels, gain and output buffer. Effects parameters are accepted and ignored.
- `SystemRender`: each playing sampler voice reads its blocks from guest memory (PCM 8, 16, 24, 32-bit and float, either byte order), is resampled to the system rate by linear interpolation at `rate / system rate * pitch`, and follows its patches (submixers pass the signal on unchanged) to a mastering voice, which adds it times its gain to its render buffer (S16 or float, 1 to 8 channels). Block callbacks are called as guest functions after the render, with SharpEmu-mac's 64-byte info block.
- `VoiceGetState`, `VoiceGetStateFlags`: in use, playing, paused, stopped, error, out of blocks; decoded sample and byte counts; the current block.
- `ParseWaveformData`, `ParseWaveformFile`: RIFF WAVE images (PCM and float, including `WAVE_FORMAT_EXTENSIBLE`) to `SceNgs2WaveformInfo` with one block covering the data.

Two choices differ from SharpEmu-mac, both forced by Balatro:

- A voice ends when it plays out its last block, not whenever its queue is empty. Balatro starts its music voices before queueing their first block, and SharpEmu-mac's rule stopped them at the first render. A voice fed with streamed blocks (flag 1) waits for more instead of ending.
- A block's sample count is clamped to the data it carries. Balatro's streamed blocks give the whole track's length (for example 8,332,853 frames for a 16 KiB block of `music1.wav`), which SharpEmu-mac refuses for PCM.

Placeholders (`STUBS`, they print `STUB!`): `sceNgs2SystemGetInfo`.

**AudioOut2** (`audio/out2.rs`), the PS5's output library, which FMOD uses: a context owns ports; the title points each port at one grain of samples with `PortSetAttributes` (attribute 0, PCM: the value points at the grain's address; attribute 1, gain: a float) and plays the grain with `ContextAdvance` or `ContextPush`. Each grain is mixed to stereo (float or S16 ports, 1 to 16 channels; layouts other than mono, stereo and 8 channels keep their first two channels) and goes to the same host mixer and WAVE dump as AudioOut. A context has a clock of grains with the queue depth it asked for (default 4 grains of 256 frames); `Advance` plays a grain without waiting, `Push` plays one unless `Advance` already did and, if blocking, waits for a free slot (the device's pace when the device runs), and `GetQueueLevel` reports grains queued and slots left (32 bits each). `ResetParam` writes SharpEmu-mac's defaults and `QueryMemory` asks for its size; the context memory itself is not used. Structure layouts, attribute ids and error codes come from SharpEmu-mac's `AudioOut2Exports.cs`; the pacing is ours. Also `UserCreate`, `UserDestroy`, `PortCreate`, `PortDestroy`, `Initialize`.

**Ajm** (`ajm.rs`), placeholders: no codec is decoded. Batch jobs take their space in the title's batch buffer as the library would (`{ buffer, offset, size, last job, last job return address }`, job sizes 48, 64 and 32 bytes), a decode consumes all its input and writes silence, `GetCodecInfo` writes a zero result, and batches finish when they start. Layouts, sizes and error codes follow SharpEmu-mac's `AjmExports.cs`, which does the same for codecs it cannot decode. FMOD registers codec 3 (unknown to us) and so plays its compressed sounds as silence.

## Verified, and how

- **Balatro, headless, 60 s, WAVE dump.** 58 of 60 seconds carry sound (per-second RMS 170 to 1,940 of 32,768, peaks up to 8,543, nothing clipped); the first two seconds are silent while the title loads. Cross-correlating 2-second windows of the dump at 30 s and 45 s against the game's own `music1.wav` resampled by 0.7 gives correlation 1.0000, gain 0.300 and a residual of 0.1 to 0.2 % of the signal, and the two windows sit exactly 15 s apart in the track. So the music plays continuously, in real time, at the pitch and volume Balatro asks for. The pitch is Balatro's: `engine/sound_manager.lua` plays music with `per = 0.7`, and the volume 0.3 is the port volume it sets. Before the menu music, the dump holds the start-up sounds and a song that fades out by the game's own volume commands.
- **Device path, 20 s** with `PS5SLOP_AUDIO=1`: the queue opens, and 19.7 s of audio pass in 20 s of wall time, so the device's clock paces the title. Nobody listened to it for this check.
- Unit tests: WAVE parsing and the info layout, sample scaling, voice event states, block looping and ending, channel mapping, the host mixer, the WAVE writer.

## Assumed, unknown

- All NGS2 structure layouts, parameter and command ids, error codes and the waveform info layout come from SharpEmu-mac. Balatro's traced calls fit them (formats, block descriptors, patches and volumes decode to sensible values that reproduce the game's music exactly), but nothing was checked on a PS5. Only the parts Balatro uses are exercised.
- What a real NGS2 voice does with an empty queue, and what the `samples` field of a streamed block means, is unknown; our rules are above.
- No effects (reverb, EQ, filters, envelopes) and no compressed waveforms: VAG and ATRAC9 images are refused (`0x804A8431`), and Balatro has none. ATRAC9 would need a decoder (SharpEmu-mac ports one, LibAtrac9).
- AudioOut ports are assumed to run at 48 kHz; other rates are written to the dump as declared but sent to the device unconverted. Port types (main, BGM, voice, pad speaker) are not told apart.
- AudioOut2: only Superliminal's FMOD has used it, and nobody has listened to it yet; its compressed sounds are silent because Ajm decodes nothing. What codec 3 is, and real ATRAC9 or Opus decoding, are open (SharpEmu-mac decodes Opus and MP3 through FFmpeg).
- The `sceAudioOutOutput` return value (the frame count) is SharpEmu-mac's; on a console it is unverified.
