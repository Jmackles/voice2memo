"""Note: serializer (segments + lists -> markdown), vault paths, obsidian.json check.

Note file layout, written here and nowhere else::

    ---
    date: 2026-10-27 09:34
    audio: 20261027-0934.wav
    tags: [idea, database]
    ---
    <cleaned text, link spans wrapped in [[ ]]>

    > [!quote]- Original
    > <raw transcript, one callout line per segment>

    ![[20261027-0934.wav]]

If the vault path is under a sync folder, write to a temp file beside the
target and ``os.replace`` into place so the sync client never sees a half file.
"""
