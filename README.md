# 🎲 ChronicleWeave - From D&D Sessions to Living Stories

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

> Capture the magic of your tabletop adventures — from chaotic voices to polished, readable narratives.

---

## 🧙‍♂️ What is ChronicleWeave?

**ChronicleWeave** is a modular Python pipeline designed to transform **multi-speaker audio recordings** (like D&D sessions) into structured, readable scripts.

It automates the process from individual audio tracks per speaker to a final text document, handling:

1.  **Audio Chunking:** Intelligently splits long recordings based on silence, optimized for speaker-specific tracks.
2.  **Transcription:** Uses the powerful **[WhisperX](https://github.com/m-bain/whisperX)** model via Docker for accurate speech-to-text.
3.  **Correction & Formatting:** Cleans and corrects transcription timestamp anomalies for better accuracy.
4.  **Dialogue Assembly:** Merges transcriptions from different chunks and speakers into a coherent, timestamped SRT file.
5.  **Script Generation:** Creates a final plain text script suitable for reading or further processing.

### 🚀 **Incoming:** Module for LLM-based summarization or narrative generation.

Read the full story behind ChronicleWeave in the **🌱 Context and Origin** section.

---

## 📦 Key Features

- 🎧 **Silence-based Audio Chunking** optimized for multi-track recordings.
- 🐳 **Dockerized WhisperX Integration** for robust, GPU-accelerated transcription.
- 🛠️ **Advanced Timestamp Correction** using statistical analysis.
- ⏱️ **SRT Subtitle Generation** with speaker labels.
- 💬 **Speaker Turn Merging** for cleaner SRT output.
- 📝 **Plain Text Script Generation**.
- 📚 **Batch mode** over many `session<N>` folders, skipping finished ones.
- 🗂️ **Organized Folder Structure** per session.
- 🔧 **Selectable step range** (one contiguous range within 1-9).

---

## ⚙️ System Requirements

1.  **Python:** >= 3.10
2.  **Docker:** Required for running WhisperX transcription. Docker Engine needs to be installed and running. GPU access configured within Docker is highly recommended for performance. [Install Docker](https://docs.docker.com/engine/install/)
3.  **FFmpeg:** Required to merge the tracks and cut them into chunks (called directly).
    - Install via your system's package manager (e.g., `sudo apt install ffmpeg`, `brew install ffmpeg`) or download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to your system's PATH.
    - Verify with: `ffmpeg -version`

---

## 🐳 WhisperX Transcription via Docker

ChronicleWeave leverages the powerful **[WhisperX](https://github.com/m-bain/whisperX)** library for transcription and timestamp alignment. To ensure a consistent and reliable environment, especially regarding specific CUDA and cuDNN dependencies that can sometimes cause issues when installed directly, ChronicleWeave runs WhisperX within a **Docker container**.

**Why Docker?**

- **Dependency Management:** Encapsulates the specific versions of WhisperX, PyTorch, CUDA, cuDNN, and other dependencies known to work together reliably. This avoids common environment setup problems reported by users online.
- **Reproducibility:** Ensures the transcription step runs the same way regardless of the host machine's specific Python or CUDA setup.
- **Ease of Use:** The pipeline handles running the container, mounting volumes, and passing arguments automatically unless `--no-whisperx` is given.

**Setup:**

1.  **Install Docker Engine:** Follow instructions at [https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/).
2.  **Configure GPU Access (Recommended):** For significantly faster transcription, ensure Docker can access your NVIDIA GPU. See [NVIDIA Container Toolkit Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
3.  **Build the Image:** From the project root directory (containing the `Dockerfile`), run:
    ```bash
    docker build -t chronicleweave-whisperx .
    ```
    _(This uses the included Dockerfile based on official NVIDIA CUDA images and installs the necessary WhisperX version)._

Once built, the `run_pipeline` function will use the `chronicleweave-whisperx` image automatically for Step 2.

---

## 🚀 Quickstart & Development Setup

### 1. Clone and install

```bash
git clone https://github.com/bclement14/chronicleweave.git
cd chronicleweave
uv venv ~/.venvs/chronicleweave
uv pip install --python ~/.venvs/chronicleweave/bin/python -e ".[dev]"
```

This installs the `chronicleweave` and `chronicleweave-batch` commands in `~/.venvs/chronicleweave/bin/`.

### 2. Build the WhisperX Docker image

```bash
docker build -t chronicleweave-whisperx .
```

### 3. Configure `.env`

Copy `.env.example` to `.env` (gitignored) and edit it:

- `CW_SPEAKERS`: comma-separated `craig_username:Tag` pairs. Tags are the speaker names used in the script.
- `CW_IGNORE_TRACKS`: tracks whose username contains one of these words (case-insensitive) are dropped, e.g. `Spoticord`.

Steps 1-8 need no API key. The `.env` is looked up in the current folder and its parents, then the repository root; `--env-file PATH` uses a specific file, and values already set in the shell win.

### 4. Prepare a session folder

Put the Craig archive (`craig-*.flac.zip`) in the session folder, or the speaker `.flac` files in a `tracks/` subfolder:

```
session26/
└── craig-XXXX.flac.zip
```

### 5. Run

One session:

```bash
chronicleweave -b /path/to/session26 [--steps 1-8]
```

Every `session<N>` folder under a root, one after another (sessions already done are skipped; a failure does not stop the batch):

```bash
chronicleweave-batch /path/to/Sessions [--only 13,15-19] [--force]
```

The batch prints a summary (OK / FAILED / SKIPPED, duration, log) and exits with 1 if any session failed. Run the same command again to resume.

Both commands accept `--steps`, `--log-level`, `--verbose`, `--env-file`, `--no-whisperx`, `--diarize`, `--whisperx-model` (name or local CTranslate2 folder, default `large-v3`) and `--whisperx-language` (default `fr`). Exit code is 0 on success, 1 on failure.

---

## 🧹 Pipeline Steps Overview

`--steps` takes one step or one contiguous range within 1-9 (e.g. `5` or `2-7`). The default is **1-8**. Step 9 (LLM processing through a paid API) is optional and off by default.

| Step | Action                                                                                          | Output (default)                              |
| :--- | :---------------------------------------------------------------------------------------------- | :-------------------------------------------- |
| 0    | Prepare tracks: extract the Craig archive, rename tracks to speaker Tags, drop ignored tracks   | `tracks/`                                     |
| 1    | ffmpeg merges all tracks (16 kHz mono), finds silence cut points, ffmpeg slices each track      | `chunked_tracks/`, `final_outputs/cut_points.txt` |
| 2    | WhisperX `large-v3` in the existing Docker image (model cache `~/.cache/chronicleweave`)        | `wx_output/`                                  |
| 3    | Correct JSON timestamps                                                                         | `json_files/`                                 |
| 4    | Convert JSON to SRT                                                                             | `srt_files/`                                  |
| 5    | Merge chunk SRTs using the cut-point offsets                                                    | `final_outputs/merged_transcript.srt`         |
| 6    | Merge consecutive entries of the same speaker                                                   | `final_outputs/cleaned_transcript.srt`        |
| 7    | Create the plain text script                                                                    | `final_outputs/final_script.txt`              |
| 8    | Split the script into chunks                                                                    | `final_outputs/script_chunks/`                |
| 9    | Optional: LLM processing through an API (needs a key, off by default)                           | configured LLM output                         |

Step 0 runs automatically before step 1 unless `--no-prepare-tracks` is given.

**Outputs.** Each run writes `pipeline.log` in the session folder. A successful run writes the completion marker `final_outputs/.chronicleweave_done.json`; the batch treats a session as done when this marker exists, or, for sessions processed before the marker existed, when `final_script.txt` exists and no `final_outputs/.chronicleweave_running` marker is left behind by a crashed run.

**Re-running.** A run starting at step k first removes the outputs of steps k to 8, then regenerates them. `tracks/` and the Craig archives are never deleted.

---

## 🐳 Manual Docker Command (Advanced)

If you need to run the WhisperX transcription step manually outside the pipeline:

1.  Ensure your chunked audio files (e.g., `.flac`) are in a subdirectory (e.g., `chunked_tracks`).
2.  Build the image: `docker build -t chronicleweave-whisperx .`
3.  Run the container:

    ```bash
    # Example using 'find' (safer for many files/special chars) on Linux/macOS:
    docker run --gpus all -it --rm -v "$(pwd):/app" --ipc=host \
      chronicleweave-whisperx \
      whisperx $(find chunked_tracks -name '*.flac' -printf '%p ') \
      --model large-v3 --language fr \
      --output_dir wx_output --output_format json \
      # Add --diarize --hf_token YOUR_TOKEN if needed
    ```

    **OR**

    ```bash
    # Example using shell wildcard (CAUTION: May fail with many files or special characters):
    docker run --gpus all -it --rm -v "$(pwd):/app" --ipc=host \
      chronicleweave-whisperx \
      whisperx chunked_tracks/*.flac \
      --model large-v3 --language fr \
      --output_dir wx_output --output_format json \
      # Add --diarize --hf_token YOUR_TOKEN if needed
    ```

Place the resulting JSON files from `wx_output/` into the expected directory for the pipeline's Step 3 (default: same `wx_output/` name relative to your `base_path`).

---

## 📂 Example Final Folder Structure

```
your_session_name/
├── tracks/                     # Input speaker audio files
├── chunked_tracks/             # Output from Step 1
├── wx_output/                  # Output from Step 2 (WhisperX JSON)
├── json_files/                 # Output from Step 3 (Corrected JSON)
├── srt_files/                  # Output from Step 4 (Chunked SRTs)
└── final_outputs/              # Final aggregated outputs
    ├── merged_audio.flac       # Merged audio (from Step 1)
    ├── cut_points.txt          # Silence cut points (from Step 1)
    ├── merged_transcript.srt   # Output from Step 5
    ├── cleaned_transcript.srt  # Output from Step 6
    └── final_script.txt        # Output from Step 7
```

---

## 🧪 Testing

To run the unit tests:

1.  Ensure development dependencies are installed: `uv pip install --python ~/.venvs/chronicleweave/bin/python -e ".[dev]"`
2.  Run pytest from the project root directory:
    ```bash
    ~/.venvs/chronicleweave/bin/python -m pytest -q
    ```

---

## 🌱 Context and Origin

ChronicleWeave started as a personal quest with a perhaps ambitious goal: I wanted to automatically transcribe my D&D group's recorded sessions and use LLMs to generate summaries or session-based stories, dreaming of potentially weaving those adventures into a novel someday.

Like many D&D groups we used Discord for our sessions and we used the **[Craig bot](https://craig.chat/)** 🐻 to record the sessions. It provides separate audio tracks for each speaker (multi-track recording) – a feature that proved crucial later on. My first attempts, however, involved manually chunking the audio using tools like Audacity and trying the original [Whisper](https://github.com/openai/whisper) transcription model. The results weren't ideal, especially with the long silences present in individual speaker tracks (often, these silences, where other speakers were active, constituted more time than actual speech within a given chunk).

Checking for better models, I discovered **[WhisperX](https://github.com/m-bain/whisperX)**. My initial experiments focused on simplifying the input by using a _merged_ audio track and leveraging WhisperX's built-in diarization (`--diarize`) to separate speakers. Unfortunately, even after several rounds of parameter fine-tuning, it often failed to identify speakers correctly, which was a crucial requirement for the automated pipeline I envisioned.

This led me back to using the individual speaker tracks. Processing these separately gave much clearer transcriptions, as WhisperX appeared to handle the extensive silences within these tracks more effectively than the original Whisper model had in my earlier tests. However, a new challenge arose: due to these same long silences, WhisperX's word-level timestamp alignment was sometimes inaccurate. This, in turn, led to poor temporal synchronization when attempting to merge the dialogues from different speaker tracks into a cohesive script, which led to a misleading understanding of the temporal continuum by the LLM and poor restitution of the actual events.

To address this, I developed a custom script to process WhisperX's JSON output, detect timestamp inconsistencies (like words with improbable durations or gaps), and correct them using a home-made heuristic. This correction step became a cornerstone. My workflow evolved into a pipeline of several independent Python scripts: one to analyze the combined audio for silence and then chunk the individual speaker tracks using common cut points (ensuring temporal consistency), another for the WhisperX JSON correction, one more to reassemble the transcribed chunks into a full script, and finally, manually prompting Large Language Models (LLMs) to generate summaries or even pre-novelized versions of the sessions.

While functional for my own use, this multi-script, manual process was cumbersome. I wanted a fully automated pipeline that I could run right after each session, delivering the LLM outputs automatically.

Realizing others might have similar goals – whether for creating campaign summaries, session recaps for players, or other creative uses – the idea formed to transform these scripts not only into a personal automated pipeline but into a more robust, automated, and shareable library.

That's where this version of ChronicleWeave comes from. It has been significantly refactored and improved from those initial scripts into a structured, installable Python library (`chronicleweave`) with considerable assistance from Large Language Models (such as Gemini 2.5 Pro, GPT-4o, and Claude 3.5 Sonnet\*). The goal of this LLM-guided refactoring was to consolidate the steps, enhance robustness, add proper error handling, improve maintainability through modern Python practices, incorporate unit testing, increase usability, and prepare the code for open-sourcing on GitHub. The development involved an iterative process of code generation, detailed review, and refinement, often guided by the LLMs.

---

## 💡 Future Directions

### Next building block

- [ ] **LLM Integration:** Summarization, narrative generation, character dialogue extraction through LLM API.
- [ ] **Pipeline Running Example:** Find audio sample to allow pipeline running for testing.

### Possible evolutions

- [ ] **Direct Discord Bot Integration:** Streamline the process from recording to script.
- [ ] **Alternative ASR Models:** Integration options beyond WhisperX.
- [ ] **Configuration Files:** Allow pipeline configuration via YAML/TOML files.

---

## 🧑‍💻 License

This project is open-source under the **GNU GENERAL PUBLIC LICENSE v3.0**. See the `LICENSE` file for details.

<!-- ---

# 📣 Join the Adventure!

Contributions, feedback, and feature requests are welcome! Help make ChronicleWeave the ultimate tool for chronicling tabletop stories. Create an issue or submit a pull request. ⚔️📜 -->
