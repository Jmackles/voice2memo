"""Transcribe: faster-whisper wrapper, Segment dataclass, JSON in/out.

Segment contract (one per Whisper segment, serialized beside the WAV):
    index    int    position in the memo
    start    float  seconds
    end      float  seconds
    text     str    raw transcript for this span
    cleaned  str    rewritten text, empty until cleanup runs
    words    list   Whisper's word timestamps, kept for the clipper

FakeWhisper shares the interface and returns ``segments.json`` from a named
fixture so GUI and serializer tests run without a model.
"""
