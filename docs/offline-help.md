# Local support guide

This guide describes setup and data interpretation. It does not diagnose a recording automatically.

## Permission error after moving the project

A Python virtual environment is tied to the directory where it was created.
Activation scripts and command launchers may still reference the previous location.
An error mentioning a different project directory is evidence of a stale path.
Recreate the environment in the current project directory; preserve the old one
until the new environment works. Avoid copying activation scripts between environments.
Use `python -m streamlit` to launch with the interpreter you selected.

## Dashboard setup and launch

From the project directory, create an environment if one does not exist:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m streamlit run dashboard/app.py
```

Installation requires access to the package registry unless dependencies are cached.
Once dependencies are installed, the help search runs offline.
Select **Offline help** in the sidebar to read this guide without starting an experiment.
If the browser does not open, visit http://localhost:8501. Press Control-C in the
terminal to stop Streamlit. If that port is occupied, add `--server.port 8502`.

## Missing module or import error

Check which interpreter is active with `python -c "import sys; print(sys.executable)"`.
It should point inside this project's `.venv`. Install this project with
`python -m pip install -e '.[dev]'` using that interpreter. A missing package
does not by itself indicate a defect in the application. Keep the complete
traceback when reporting an error; redact personal paths if sharing publicly.

## IQ recording format and metadata

The existing file reader expects complex64 binary IQ: interleaved 32-bit floating
point I and Q values. Each complex sample occupies eight bytes. A file whose
size is not divisible by eight cannot contain only complete samples of this format.
Do not assume an arbitrary WAV, CSV or unsigned integer recording uses this format.

A JSON sidecar supplies `sample_rate` in samples per second,
`center_frequency` in Hz, and optionally `start_time` in seconds. For example:

```json
{"sample_rate": 20000000, "center_frequency": 500000000, "start_time": 0}
```

Verify these values against the recording instrument. Incorrect metadata changes
the displayed axes; it cannot change which frequencies were physically recorded.
The current replay adapter does not establish valid arbitrary tuning outside a
recording. Its multi-band replay output should not be treated as measured coverage.

## Units, power and calibration

Hz describes frequency, seconds describe time, and samples per second describes
sampling rate. MHz equals one million Hz. Sample count divided by sample rate
gives recording duration.

Relative digital power is not automatically calibrated input power in dBm.
Absolute power requires instrument calibration and a documented conversion.
A power spectral density has power-per-frequency units; it is not interchangeable
with total integrated power. Document the measurement settings when comparing plots.

## Missing annotations and unavailable measurements

Without independent annotations, a recording cannot establish detection accuracy
or a false-alarm rate. Unknown activity is not confirmed inactivity. An apparent
zero may mean there were no comparable observations, rather than perfect performance.
The current recording evaluator substitutes inactive truth; its truth-based
metrics must not be used as validation of an unannotated recording.

## Reproducibility and reports

Keep the original recording, its metadata, the software revision and exact analysis
settings together. Record whether data are synthetic or measured. A random seed
alone does not establish reproducibility across different software versions or
unseeded processing steps. Separate repeatability from physical measurement accuracy.

## Offline help search and privacy

Help search matches words against this bundled guide and ranks section headings
above body text. When the first search has incomplete word coverage, it recognizes
common support aliases and conservatively corrects spelling before a second search.
The refinement is bounded to two passes and a fixed vocabulary. An expandable
trace shows which terms were used; unmatched terms remain visible. Search relevance
is not diagnostic confidence. You can export the retrieved answer with its sources.
It returns up to four excerpts with source line numbers. It does
not call an LLM, upload recordings, execute questions as code, or recursively
modify the application. No matching excerpt means the guide does not contain an
answer; it is not evidence that the requested capability exists.
