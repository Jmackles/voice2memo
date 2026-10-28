One folder per memo: `memo.wav`, `segments.json`, `cleaned.json`, `entities.json`,
`tags.json`, `expected.md`. Promote folders here by hand from `tests/fixtures-inbox/`,
which is gitignored, so a personal memo never gets committed by accident.

`sample/` is synthetic: a one-second tone and hand-typed JSON, there so the fake
backends and `bench.py` have something to read before any real memo is recorded.
Replace it once real fixtures exist; nothing depends on its contents by name.

The fakes read the files as the stages would produce them:

- `segments.json`: the segment list as Whisper produced it, `cleaned` blank.
- `cleaned.json`: the same list with `cleaned` filled; the fake rewriter matches on `text`.
- `entities.json`: a list of terms, as the sanitizer leaves them.
- `tags.json`: `{"proposed": [...], "rejected": [...]}`; the fake returns the proposed ones.
- `expected.md`: the note the serializer should produce (step 9).
