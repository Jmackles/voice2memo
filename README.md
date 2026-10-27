# voice2memos (aka Coppermind)

Honestly this is just a personal project. I enjoy the functionality of ObsidianMD and like the idea of being able to quickly record and transcribe something on the fly. Being able to locally search my stream of consciousness and ideas sounded like such a cool concept. So this application is a slapped together `tkinter` interface that can either capture via push-to-talk/recording (utilizing `pynput`, `sounddevice` and `soundfile`) or loads an audio file, transcribes it locally via `faster-whisper`, displays the transcription for the user in a text pane, then passes that transcript to the `ollama` model gemma4 (possibly others like i said personal project here) to clean up garbage words like "um", and "so", and identifying important words/phrases/concepts and applying appropriate wikilink markdown to the text. The cleaned up note to be saved to the vault is then displayed on the right hand text pane. The user can edit either field and save the file to their obsidian vault. 

## Status

lol (pre-alpha)

## What it does

You hold a key and talk. A nonintrusive visual popup is introduced to clue you in that you're recording if you are not directly focused on the app window.

1. The audio is saved as a WAV.
2. Whisper (faster-whisper, running locally) transcribes it into timestamped segments.
3. A local model (gemma4:e2b through Ollama) rewrites each segment into clean written sentences without adding, removing, or summarizing anything.
4. A second call picks out the subjects the memo is about and proposes a couple of tags from a list you control.
5. The app shows you the original on the left, locked, and the cleaned version on the right, with proposed wikilinks highlighted. You click words to link or unlink them, strike any tag you don't want, and save.
6. A note lands in your Obsidian vault with the cleaned text, the original transcript in a collapsed quote, and the audio embedded.

Every memo is its own note, so clicking a linked term in Obsidian shows every time you've talked about it, and how you said it. Nothing leaves your machine unless you deliberately point the cleanup step at an API provider in Settings.
 
There is also a push-to-note mode: a second hotkey opens the window with an empty pane and the cursor ready, for memos you'd rather type.

## Install
 
Planned: one script that checks what you already have, installs what's missing, and says which was which. Until that exists, the manual version on Windows is:
 
```
winget install Python.Python.3.11
winget install Ollama.Ollama
ollama pull gemma4:e2b
 
git clone https://github.com/<you>/voice2memos
cd voice2memos
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m voice2memos
```
 
On first launch the Whisper model downloads (a few hundred MB); the status line says so.
 
NVIDIA GPU users add `pip install -r requirements-cuda.txt`. CPU is *probably* fine for memos under a minute. I honestly don't know yet. I'm kindof fleshing out my readme under the assumption that I will find out.
 
macOS and Linux are designed for but not yet tested. macOS needs Accessibility and Microphone permissions, which the app will prompt for. Wayland uses the GlobalShortcuts portal for the hotkey where the desktop supports it.

## Configuration

There is a `voice2memos.cfg` file editable by hand or through the settings dialog: hotkeys, Whisper model size, ollama model, model parameter settings (ie temperature or repeat penalty), the rewrite prompt itself, the tag pool, the vault path, and an example/primer section for giving mock data at the start for the model. This is absolutely essential so you can lay out how you want it to behave based off your own style. I recommend organically finding some things you comment on throughout your day, write that comment down, and manually clean it up, add wikilinks to it and boom you're good. It's easier to do that than to try to come up with mock data on the spot. you want something you'll actually want to note. I guess if you want you can sit down and think of stuff. I'm not your boss.

## Roadmap
 
In build order. Each step is meant to be runnable on its own before the next starts.
 
- [ ] Installer that checks and installs dependencies idempotently (I had to learn this word so you do too. )
- [ ] Window shell with every control present and clickable
- [ ] Config file, loader, and defaults
- [ ] Settings dialog, applied without restart
- [ ] Testing workbench: bench.py and fake backends
- [ ] Buttons wired and tested against fakes
- [ ] Whisper wrapper, tested from the bench
- [ ] Transcript shown in the left pane via a worker thread
- [ ] Editable and savable panes; notes land in the vault
- [ ] Mock data: hand-written transcript and cleanup pairs as fixtures and prompt examples
- [ ] Cleanup through Ollama, tuned against fixtures
- [ ] Entities, tags, paint mode
- [ ] Error model, logging, log viewer
- [ ] Push-to-note
- [ ] Per-segment audio clips embedded in the note
- [ ] Bootstrap scripts, launchers, packaged build
Out of scope, on purpose: long recordings, multiple speakers, editing existing notes, mobile, any server component, and organizing your vault for you. Multiple speakers is something I'm interested in later for other reasons but not right now.

## Development
 
```
pip install -r requirements-dev.txt
pytest
python bench.py transcribe tests/fixtures/<name>/memo.wav
python bench.py clean --all
```
 
`bench.py` runs any single stage against a saved fixture and diffs the result against the expected output. `clean --all` is the loop for tuning the rewrite prompt. Setting `provider = fake` in the config runs the whole app end to end on fixtures with no models loaded.
 
## Why
 
This sounds really ridiculous but I can't take notes but I also need to take notes. I don't want someone to write notes for me, as that defeats the entire purpose of notating something because the act of doing so is in itself helping retain information. Personally amongst a plethora of other compounding factors I possess ADHD. Severe executive dysfunction is my life, and a huge driving factor in that is working memory. Accessing information in one's brain on command. 

So I can have a ton of thoughts, but acting upon them or retrieving them later is much more difficult. But it's extremely easy to speak them aloud. Only when I have certain ideas I'm not always in a position or desire to open up an app and note stuff down as each one of these things is in itself its own tedious task. 

Idk I could ramble about this for forever. The hardest part is getting the thoughts somewhere searchable without the friction of writing, and this is that bridge, which is built to keep the original words alongside the tidied ones.
 
## License
 
MIT
