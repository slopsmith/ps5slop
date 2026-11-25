# Audio

How titles make sound and how it reaches the Mac. Code: `crates/hle/src/audio` (AudioOut, AudioOut2, AudioIn and the host side), `crates/hle/src/ngs2`, `ajm.rs`, `acm.rs`.

## Paths through the system

A title hands samples to the system in one of three ways:

- **AudioOut.** The title mixes its own audio, opens a port with `sceAudioOutOpen` and pushes one buffer at a time with `sceAudioOutOutput`, which blocks for about one buffer period. Titles use that blocking as their audio clock.
- **NGS2, then AudioOut.** The title builds a voice graph in libSceNgs2 (sampler voices, submixers, mastering voices), and `sceNgs2SystemRender` mixes one grain into a buffer the title then passes to AudioOut.
- **AudioOut2**, the PS5's output library, which FMOD and RAGE use. A context owns ports; the title points each port at one grain of samples with `PortSetAttributes` and plays the grain with `ContextAdvance` or `ContextPush`.

Compressed audio is decoded by the title itself (FMOD decodes its own banks) or through **Ajm**.

All three output paths end in one host mixer: every port's samples are converted to stereo float at 48 kHz and summed.

## The host side

- **Device.** The mixer feeds the default output device through an AudioToolbox output queue (`audio/coreaudio.rs`, our own small FFI instead of a crate such as cpal, to avoid a dependency tree): three 512-frame float buffers, filled on the queue's own thread. A port's writer waits while more than 2048 frames (43 ms) are queued for it, so **the device's clock paces the title** and the title's audio clock never drifts from the speakers. If the device stops taking samples for 250 ms, the excess is dropped and the writer carries on.
- **No device.** Writers sleep until the previous buffer would have finished playing (the host clock paces the title), and the samples are dropped.
- The device opens with the first port when the window is shown; `--headless` keeps it closed. `PS5SLOP_AUDIO` overrides both ways ([environment.md](environment.md)).
- **WAVE dump.** `PS5SLOP_AUDIO_WAV=path` records what each port plays as 16-bit stereo: the first port to `path`, further ports to `<stem>-<port>.wav`. The header is rewritten after every buffer, so the file is valid however the process ends. This is how output is checked without listening.
- **Late buffers.** A buffer or grain submitted more than one buffer after the previous one should have finished is counted in the problem report, per port or context. On a device each one is a dropout.

## AudioOut

Formats 0 to 7: S16 and float, mono, stereo and 8 channels. 8-channel buffers fold to stereo with the centre and surround channels at -3 dB. `SetVolume` takes one volume per channel bit, 32768 being unity. `Output` returns the port's frame count, as SharpEmu-mac does (unverified on a console).

Assumed: ports run at 48 kHz (other rates go to the dump as declared but to the device unconverted). Port types (main, BGM, voice, pad speaker) are not told apart.

## AudioOut2

- Port attributes: 0 is the PCM pointer for the next grain, 1 the gain (a float), 8 the port's ambisonics channel (`u32::MAX` for none, ACN numbering from 64). Other attributes are accepted.
- Ports are float or S16 with 1 to 16 channels. Mono and stereo play as they are, 8 channels fold as in AudioOut, other layouts keep their first two channels.
- A context has a clock of grains with the queue depth it asked for (default 4 grains of 256 frames). `Advance` plays a grain without waiting; `Push` plays one unless `Advance` already did and, if blocking, waits for a free slot. `GetQueueLevel` reports grains queued and slots left.
- The context memory the title allocates is not used. Speaker arrays are created and destroyed, but nothing is mixed through them.
- Structure layouts, attribute ids and error codes come from SharpEmu-mac's `AudioOut2Exports.cs`; the pacing is ours.

### Ambisonics

A mono port with an ambisonics channel is one channel of a soundfield, not a signal to play. Mixing every channel at full gain makes positioned sounds much too loud, with a level that jumps as sources or the camera move. So a soundfield is decoded to two virtual cardioid microphones pointing left and right from its first two ACN channels:

- left = (W + Y) / 2, right = (W - Y) / 2
- Left plus right is the omnidirectional W from every direction, so the decode never exceeds the source's level.
- Higher orders are dropped: they add spatial detail two speakers cannot show. An HRTF or speaker-array decode can replace this if a title needs real spatial audio.

Assumed, not verified: the normalisation (SN3D or N3D changes only how wide Y pans), that Y's positive direction is left as in the ACN convention, and how loud a PS5 decoding to stereo makes W relative to the title's plain ports. A unit test checks the decode against a synthetic soundfield.

## NGS2

Ported from SharpEmu-mac's NGS2 (GPL-2.0-or-later, see [references.md](references.md)), PCM only:

- Systems, racks and voices. Handles are small numbers from one counter.
- Voice parameter lists (`VoiceControl`) and 16-byte command records (`VoiceRunCommands`): waveform format, block queue, pitch, port volume, level matrices, patches, events, callbacks, mastering channels and gain. Effect parameters are accepted and ignored.
- `SystemRender`: each playing sampler voice reads its blocks from guest memory (PCM 8 to 32 bit and float, either byte order), is resampled by linear interpolation at `rate / system rate * pitch`, and follows its patches to a mastering voice, which adds it times its gain into its render buffer. Block callbacks run as guest calls after the render.
- `ParseWaveformData`, `ParseWaveformFile`: RIFF WAVE images only. `CalcWaveformBlock` handles PCM only.

Two rules differ from SharpEmu-mac:

- A voice ends when it plays out its last block, not whenever its queue is empty: titles start voices before queueing their first block. A voice fed with streamed blocks waits for more instead of ending.
- A block's sample count is clamped to the data it carries: streamed blocks may carry the whole track's length.

Not done: effects (reverb, EQ, filters, envelopes) and compressed waveforms (VAG and ATRAC9 are refused; ATRAC9 would need a decoder, which SharpEmu-mac ports from LibAtrac9). `ParseWaveformFile` reads host paths, so it cannot read files inside a package yet.

## Ajm

- Batch jobs take their space in the title's batch buffer as the library would, and batches finish when they start.
- **MP3** (codec 0) is decoded with FFmpeg's libavcodec through `ffmpeg-next` (a slim LGPL build from the flake, bundled with the app; FFmpeg can also serve video later). A decode job takes the whole MPEG audio frames in its input plus what the previous job of the instance left over, decodes them with one decoder per instance (so the bit reservoir carries across jobs) while the output has room, and reports input consumed, bytes written and the running sample count. An incomplete last frame waits for the next job.
- **Sizes come from the frame headers**, not from FFmpeg: each frame writes exactly its header's sample count, padded with silence when FFmpeg gives fewer. So the results a title reads never depend on the decoder.
- Output is S16 unless the instance flags ask for S32 or float.
- Other codecs consume all input, zero the output buffer and report nothing written.
- Instance ids, the result layout and the flag bits follow SharpEmu-mac's `AjmExports.cs`.
- Instances are not kept in save states: after a load, a stream's partial frame, decoder state and sample count start empty, so the first frames may be silent.

## Acm, AudioIn, AudioPropagation

- **libSceAcm**: placeholders that return 0 and write nothing. What the library computes is unknown (no public description, and no emulator we checked implements it). FMOD gives up all output when the context create fails. If FMOD starts batches in game and needs their results, that will show as missing effects, not silence.
- **libSceAudioIn**: a silent microphone. `Input` waits one grain and writes zeros, since capture threads read it in a loop and would spin otherwise.
- **libSceAudioPropagation**: placeholders; no rooms, portals or sound paths are modelled.

## Verified, and how

- Music in Balatro (NGS2 path) and Minecraft (FMOD through AudioOut2) was checked from WAVE dumps by cross-correlating windows against the game's own track files: correlation 1.0, right pitch and the gain the game asks for, in real time. vic listened to a Minecraft dump.
- The device path paces the title by the device's clock (seconds of audio per second of wall time).
- Unit tests: WAVE parsing, sample scaling, voice states, block looping and ending, channel mapping, the mixer, the WAVE writer, the ambisonics decode, MP3 decoding of our own sine (pitch, level, S16 and float).
- Not checked: GTA V's sound in game since the Ajm and ambisonics changes; nobody has listened.

## Unknown

- Every NGS2 layout, parameter id and error code is SharpEmu-mac's; only the parts our titles use are exercised.
- What a real NGS2 voice does with an empty queue, and what a streamed block's sample count means.
- What Ajm codec 3 is; ATRAC9 and Opus decoding are not done.
