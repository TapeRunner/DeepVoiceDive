"""Tkinter desktop GUI for DeepVoiceDive.

A single window with one tab per command (analyse, compare, matrix, profile,
screen). It sits directly on top of the library API, so the GUI stays a thin
front-end: every heavy call (loading audio, openSMILE, neural model, plotting)
runs on a background thread and reports back to the Tk main loop through a
queue, keeping the window responsive.

Tkinter is part of the Python standard library but ships as a separate system
package (e.g. ``python3-tk``) on some Linux distributions. Importing this module
never requires Tkinter; it is imported lazily inside :func:`main` so the rest of
the package keeps working in headless environments.

Launch with::

    deepvoicedive-gui
"""
from __future__ import annotations

import queue
import threading
import traceback
from pathlib import Path

# Only one plotting task may run at a time: matplotlib's pyplot state machine
# (used by deepvoicedive.visualize) is not thread-safe.
_AUDIO_FILETYPES = [("WAV-Audio", "*.wav"), ("Alle Dateien", "*.*")]
_JSON_FILETYPES = [("JSON-Profil", "*.json"), ("Alle Dateien", "*.*")]


def _load_scaled_photo(tk, path, max_w: int, max_h: int):
    """Load a PNG into a ``tk.PhotoImage``, integer-downscaled to fit a box.

    Tk 8.6's PhotoImage reads PNG natively, so no Pillow dependency is needed.
    Scaling is limited to integer ``subsample`` factors, which is plenty for
    previewing report figures inside the window.
    """
    img = tk.PhotoImage(file=str(path))
    factor = max(
        1,
        -(-img.width() // max_w),   # ceil division
        -(-img.height() // max_h),
    )
    if factor > 1:
        img = img.subsample(factor)
    return img


def build_app():
    """Construct and return the main application window.

    Imported symbols stay local so that importing this module never pulls in
    Tkinter (which may be absent in headless setups).
    """
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    class FileListPanel(ttk.Frame):
        """A labelled list of file paths with add / remove buttons."""

        def __init__(self, master, label: str, filetypes):
            super().__init__(master)
            self._filetypes = filetypes
            ttk.Label(self, text=label).grid(
                row=0, column=0, columnspan=2, sticky="w", pady=(0, 2)
            )

            self.listbox = tk.Listbox(self, height=5, selectmode=tk.EXTENDED)
            self.listbox.grid(row=1, column=0, sticky="nsew")
            scroll = ttk.Scrollbar(
                self, orient="vertical", command=self.listbox.yview
            )
            scroll.grid(row=1, column=1, sticky="ns")
            self.listbox.configure(yscrollcommand=scroll.set)

            btns = ttk.Frame(self)
            btns.grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))
            ttk.Button(btns, text="Hinzufügen…", command=self._add).pack(
                side="left"
            )
            ttk.Button(btns, text="Entfernen", command=self._remove).pack(
                side="left", padx=(6, 0)
            )
            ttk.Button(btns, text="Leeren", command=self._clear).pack(
                side="left", padx=(6, 0)
            )

            self.columnconfigure(0, weight=1)
            self.rowconfigure(1, weight=1)

        def _add(self):
            paths = filedialog.askopenfilenames(filetypes=self._filetypes)
            for p in paths:
                self.listbox.insert(tk.END, p)

        def _remove(self):
            for index in reversed(self.listbox.curselection()):
                self.listbox.delete(index)

        def _clear(self):
            self.listbox.delete(0, tk.END)

        def paths(self) -> list:
            return list(self.listbox.get(0, tk.END))

    class PathField(ttk.Frame):
        """A single-path entry with a Browse button (open or save)."""

        def __init__(self, master, label, *, mode="open",
                     filetypes=None, default=""):
            super().__init__(master)
            self._mode = mode
            self._filetypes = filetypes or [("Alle Dateien", "*.*")]
            ttk.Label(self, text=label, width=14).pack(side="left")
            self.var = tk.StringVar(value=default)
            ttk.Entry(self, textvariable=self.var).pack(
                side="left", fill="x", expand=True
            )
            ttk.Button(self, text="…", width=3, command=self._browse).pack(
                side="left", padx=(4, 0)
            )

        def _browse(self):
            if self._mode == "open":
                path = filedialog.askopenfilename(filetypes=self._filetypes)
            elif self._mode == "save":
                path = filedialog.asksaveasfilename(filetypes=self._filetypes)
            else:  # directory
                path = filedialog.askdirectory()
            if path:
                self.var.set(path)

        def get(self) -> str:
            return self.var.get().strip()

    class DeepVoiceDiveApp(tk.Tk):
        """Main window: a notebook with one tab per command."""

        def __init__(self):
            super().__init__()
            from . import __version__

            self.title(f"DeepVoiceDive {__version__}")
            self.geometry("980x720")
            self.minsize(820, 600)

            self._queue: "queue.Queue" = queue.Queue()
            self._busy = False
            self._image_refs: list = []  # keep PhotoImages alive

            notebook = ttk.Notebook(self)
            notebook.pack(fill="both", expand=True, padx=10, pady=(10, 4))
            self._build_analyze_tab(notebook)
            self._build_compare_tab(notebook)
            self._build_matrix_tab(notebook)
            self._build_profile_tab(notebook)
            self._build_screen_tab(notebook)

            status = ttk.Frame(self)
            status.pack(fill="x", side="bottom")
            self._status = tk.StringVar(value="Bereit.")
            ttk.Label(
                status, textvariable=self._status, anchor="w", relief="sunken"
            ).pack(side="left", fill="x", expand=True, padx=(10, 4), pady=2)
            self._progress = ttk.Progressbar(status, mode="indeterminate",
                                              length=140)
            self._progress.pack(side="right", padx=(0, 10), pady=2)

            self.after(100, self._poll_queue)

        # -- background execution ------------------------------------------

        def _poll_queue(self):
            """Drain callbacks posted by worker threads, on the main thread."""
            try:
                while True:
                    self._queue.get_nowait()()
            except queue.Empty:
                pass
            self.after(100, self._poll_queue)

        def _run_async(self, work, on_success, *, busy_msg, buttons):
            """Run ``work()`` off-thread; marshal the result back to Tk.

            ``buttons`` are disabled for the duration so a second plotting task
            cannot start concurrently (pyplot is not thread-safe).
            """
            if self._busy:
                messagebox.showinfo(
                    "Bitte warten",
                    "Es läuft bereits eine Berechnung. Bitte abwarten.",
                )
                return
            self._busy = True
            for b in buttons:
                b.configure(state="disabled")
            self._status.set(busy_msg)
            self._progress.start(12)

            def worker():
                try:
                    result = work()
                    self._queue.put(lambda: _finish(result, None))
                except Exception as exc:  # noqa: BLE001
                    tb = traceback.format_exc()
                    self._queue.put(lambda: _finish(None, (exc, tb)))

            def _finish(result, error):
                self._busy = False
                self._progress.stop()
                for b in buttons:
                    b.configure(state="normal")
                if error is not None:
                    exc, tb = error
                    self._status.set(f"Fehler: {exc}")
                    messagebox.showerror("Fehler", f"{exc}\n\n{tb}")
                    return
                self._status.set("Fertig.")
                on_success(result)

            threading.Thread(target=worker, daemon=True).start()

        def _show_image(self, container, path, *, max_w=900, max_h=440):
            """Display a PNG inside ``container`` (a Label), scaled to fit."""
            try:
                photo = _load_scaled_photo(tk, path, max_w, max_h)
            except Exception as exc:  # noqa: BLE001
                container.configure(image="", text=f"(Bild nicht ladbar: {exc})")
                return
            self._image_refs.append(photo)
            container.configure(image=photo, text="")

        @staticmethod
        def _require(paths, message):
            if not paths:
                raise ValueError(message)

        # -- tab: analyze --------------------------------------------------

        def _build_analyze_tab(self, notebook):
            tab = ttk.Frame(notebook, padding=10)
            notebook.add(tab, text="Analyse")

            self._an_input = PathField(
                tab, "WAV-Aufnahme:", mode="open", filetypes=_AUDIO_FILETYPES
            )
            self._an_input.pack(fill="x")
            self._an_output = PathField(
                tab, "Ausgabeordner:", mode="dir", default="results"
            )
            self._an_output.pack(fill="x", pady=(6, 0))

            self._an_no_egemaps = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                tab, text="eGeMAPS-Report überspringen (schneller, ohne openSMILE)",
                variable=self._an_no_egemaps,
            ).pack(anchor="w", pady=(6, 0))

            run = ttk.Button(tab, text="Analysieren", command=self._do_analyze)
            run.pack(anchor="w", pady=(8, 6))
            self._an_run = run

            self._an_summary = tk.Text(tab, height=6, wrap="word")
            self._an_summary.pack(fill="x")
            self._an_summary.configure(state="disabled")
            self._an_image = ttk.Label(tab, anchor="center")
            self._an_image.pack(fill="both", expand=True, pady=(8, 0))

        def _do_analyze(self):
            from .report import analyze_file

            inp = self._an_input.get()
            out = self._an_output.get() or "results"
            self._require(inp, "Bitte eine WAV-Aufnahme auswählen.")
            with_egemaps = not self._an_no_egemaps.get()

            def work():
                return analyze_file(inp, out, with_egemaps=with_egemaps)

            def done(result):
                lines = [
                    "Analyse abgeschlossen.",
                    f"Voice Embedding : {len(result['embedding'])}-dim",
                    f"  {result['embedding_json']}",
                    f"  {result['embedding_npy']}",
                    f"Visualisierung  : {result['report_png']}",
                ]
                if "egemaps_csv" in result:
                    lines.append(
                        f"eGeMAPS-Report  : {result['egemaps_csv']} (88 Parameter)"
                    )
                self._set_text(self._an_summary, "\n".join(lines))
                self._show_image(self._an_image, result["report_png"])

            self._run_async(
                work, done,
                busy_msg="Analysiere Aufnahme …", buttons=[self._an_run],
            )

        # -- tab: compare --------------------------------------------------

        def _build_compare_tab(self, notebook):
            tab = ttk.Frame(notebook, padding=10)
            notebook.add(tab, text="Vergleich")

            self._cmp_a = PathField(
                tab, "Aufnahme A:", mode="open", filetypes=_AUDIO_FILETYPES
            )
            self._cmp_a.pack(fill="x")
            self._cmp_b = PathField(
                tab, "Aufnahme B:", mode="open", filetypes=_AUDIO_FILETYPES
            )
            self._cmp_b.pack(fill="x", pady=(6, 0))

            run = ttk.Button(tab, text="Vergleichen", command=self._do_compare)
            run.pack(anchor="w", pady=(8, 6))
            self._cmp_run = run

            self._cmp_match = ttk.Label(tab, text="", font=("TkDefaultFont", 22))
            self._cmp_match.pack(anchor="w", pady=(10, 0))
            self._cmp_detail = ttk.Label(tab, text="", font=("TkDefaultFont", 12))
            self._cmp_detail.pack(anchor="w")

        def _do_compare(self):
            from .compare import compare_files

            a, b = self._cmp_a.get(), self._cmp_b.get()
            self._require(a and b, "Bitte zwei WAV-Aufnahmen auswählen.")

            def work():
                return compare_files(a, b)

            def done(result):
                dist, sim, match = result
                self._cmp_match.configure(text=f"Voice Match: {match:.2f} %")
                self._cmp_detail.configure(
                    text=f"Kosinus-Distanz: {dist:.4f}    "
                         f"Ähnlichkeit: {sim:.4f}"
                )

            self._run_async(
                work, done,
                busy_msg="Vergleiche Aufnahmen …", buttons=[self._cmp_run],
            )

        # -- tab: matrix ---------------------------------------------------

        def _build_matrix_tab(self, notebook):
            tab = ttk.Frame(notebook, padding=10)
            notebook.add(tab, text="Matrix")

            self._mx_files = FileListPanel(
                tab, "WAV-Aufnahmen (mind. 2):", _AUDIO_FILETYPES
            )
            self._mx_files.pack(fill="x")
            self._mx_output = PathField(
                tab, "Ausgabeordner:", mode="dir", default="results"
            )
            self._mx_output.pack(fill="x", pady=(6, 0))

            run = ttk.Button(tab, text="Matrix berechnen",
                             command=self._do_matrix)
            run.pack(anchor="w", pady=(8, 6))
            self._mx_run = run

            self._mx_summary = ttk.Label(tab, text="", anchor="w")
            self._mx_summary.pack(fill="x")
            self._mx_image = ttk.Label(tab, anchor="center")
            self._mx_image.pack(fill="both", expand=True, pady=(8, 0))

        def _do_matrix(self):
            from .batch import write_similarity_report

            files = self._mx_files.paths()
            out = self._mx_output.get() or "results"
            self._require(
                len(files) >= 2,
                "Bitte mindestens zwei Aufnahmen hinzufügen.",
            )

            def work():
                return write_similarity_report(files, out)

            def done(result):
                self._mx_summary.configure(
                    text=f"Matrix gespeichert: {result['csv']}"
                )
                self._show_image(self._mx_image, result["png"])

            self._run_async(
                work, done,
                busy_msg="Berechne Voice-Match-Matrix …", buttons=[self._mx_run],
            )

        # -- tab: profile --------------------------------------------------

        def _build_profile_tab(self, notebook):
            tab = ttk.Frame(notebook, padding=10)
            notebook.add(tab, text="Profil")

            self._pf_clips = FileListPanel(
                tab, "Saubere Clips deiner eigenen Stimme:", _AUDIO_FILETYPES
            )
            self._pf_clips.pack(fill="x")
            self._pf_output = PathField(
                tab, "Profil speichern als:", mode="save",
                filetypes=_JSON_FILETYPES, default="voice_profile.json",
            )
            self._pf_output.pack(fill="x", pady=(6, 0))

            self._pf_method = tk.StringVar(value="neural")
            self._method_radios(tab).pack(anchor="w", pady=(6, 0))

            run = ttk.Button(tab, text="Profil erstellen",
                             command=self._do_profile)
            run.pack(anchor="w", pady=(8, 6))
            self._pf_run = run

            self._pf_summary = tk.Text(tab, height=8, wrap="word")
            self._pf_summary.pack(fill="both", expand=True)
            self._pf_summary.configure(state="disabled")

        def _method_radios(self, master, var=None):
            var = var or self._pf_method
            frame = ttk.Frame(master)
            ttk.Label(frame, text="Methode:").pack(side="left")
            ttk.Radiobutton(
                frame, text="neural (genauer, Fallback → mfcc)",
                value="neural", variable=var,
            ).pack(side="left", padx=(6, 0))
            ttk.Radiobutton(
                frame, text="mfcc (offline)", value="mfcc", variable=var,
            ).pack(side="left", padx=(6, 0))
            return frame

        def _do_profile(self):
            from .profile import build_profile, save_profile

            clips = self._pf_clips.paths()
            out = self._pf_output.get() or "voice_profile.json"
            requested = self._pf_method.get()
            self._require(clips, "Bitte mindestens einen Clip hinzufügen.")

            def work():
                method = _resolve_method(requested)
                profile = build_profile(clips, method=method)
                save_profile(profile, out)
                return method, profile, out

            def done(result):
                method, profile, path = result
                lines = [
                    f"Stimmprofil erstellt ({method}, "
                    f"{len(profile['embedding'])}-dim) "
                    f"aus {len(clips)} Clip(s).",
                ]
                if profile["f0_median"] is not None:
                    lines.append(
                        f"Tonlage: {profile['f0_low']:.0f}-"
                        f"{profile['f0_high']:.0f} Hz "
                        f"(Median {profile['f0_median']:.0f} Hz)"
                    )
                lines.append(f"Gespeichert: {path}")
                self._set_text(self._pf_summary, "\n".join(lines))

            self._run_async(
                work, done,
                busy_msg="Erstelle Stimmprofil …", buttons=[self._pf_run],
            )

        # -- tab: screen ---------------------------------------------------

        def _build_screen_tab(self, notebook):
            tab = ttk.Frame(notebook, padding=10)
            notebook.add(tab, text="Screening")

            self._sc_profile = PathField(
                tab, "Stimmprofil:", mode="open", filetypes=_JSON_FILETYPES
            )
            self._sc_profile.pack(fill="x")
            ttk.Label(
                tab,
                text="… oder stattdessen Referenz-Clips angeben (wenn kein "
                     "Profil gesetzt ist):",
                foreground="#555",
            ).pack(anchor="w", pady=(4, 2))
            self._sc_refs = FileListPanel(
                tab, "Referenz-Clips deiner Stimme:", _AUDIO_FILETYPES
            )
            self._sc_refs.pack(fill="x")

            self._sc_stems = FileListPanel(
                tab, "Kandidaten-Stems zum Prüfen:", _AUDIO_FILETYPES
            )
            self._sc_stems.pack(fill="x", pady=(6, 0))

            opts = ttk.Frame(tab)
            opts.pack(fill="x", pady=(6, 0))
            ttk.Label(opts, text="Schwelle (%):").pack(side="left")
            self._sc_threshold = tk.DoubleVar(value=75.0)
            ttk.Spinbox(
                opts, from_=0, to=100, increment=1, width=6,
                textvariable=self._sc_threshold,
            ).pack(side="left", padx=(4, 12))
            self._sc_method = tk.StringVar(value="neural")
            self._method_radios(opts, var=self._sc_method).pack(side="left")

            self._sc_output = PathField(
                tab, "Ausgabeordner:", mode="dir", default="results"
            )
            self._sc_output.pack(fill="x", pady=(6, 0))

            run = ttk.Button(tab, text="Screening starten",
                             command=self._do_screen)
            run.pack(anchor="w", pady=(8, 6))
            self._sc_run = run

            columns = ("match", "pitch", "suitability", "verdict")
            self._sc_table = ttk.Treeview(
                tab, columns=columns, show="headings", height=6
            )
            for col, head in zip(
                columns, ("Match %", "Tonlage %", "Eignung %", "Eignung")
            ):
                self._sc_table.heading(col, text=head)
                self._sc_table.column(col, width=110, anchor="center")
            self._sc_table.column("#0", width=0)
            self._sc_table.pack(fill="x")
            self._sc_image = ttk.Label(tab, anchor="center")
            self._sc_image.pack(fill="both", expand=True, pady=(8, 0))

        def _do_screen(self):
            from .profile import build_profile, load_profile
            from .screen import write_screening_report

            profile_path = self._sc_profile.get()
            refs = self._sc_refs.paths()
            stems = self._sc_stems.paths()
            threshold = float(self._sc_threshold.get())
            out = self._sc_output.get() or "results"
            requested = self._sc_method.get()
            self._require(stems, "Bitte mindestens einen Stem hinzufügen.")
            if not profile_path and not refs:
                raise ValueError(
                    "Bitte ein Profil ODER Referenz-Clips angeben."
                )

            def work():
                if profile_path:
                    profile = load_profile(profile_path)
                    method = profile.get("method", "mfcc")
                    if method == "neural" and _resolve_method("neural") != "neural":
                        raise ValueError(
                            "Profil wurde neuronal erstellt, das Modell ist hier "
                            "nicht verfügbar. Bitte '.[neural]' installieren oder "
                            "ein MFCC-Profil verwenden."
                        )
                else:
                    method = _resolve_method(requested)
                    profile = build_profile(refs, method=method)
                return write_screening_report(
                    profile, stems, out, method=method, threshold=threshold
                )

            def done(result):
                self._sc_table.delete(*self._sc_table.get_children())
                for r in result["results"]:
                    mark = "✅" if r["verdict"] else "❌"
                    self._sc_table.insert(
                        "", "end",
                        values=(
                            f"{r['match']:.0f}", f"{r['pitch']:.0f}",
                            f"{r['suitability']:.0f}", mark,
                        ),
                        text=r["name"],
                    )
                self._show_image(self._sc_image, result["png"])

            self._run_async(
                work, done,
                busy_msg="Screene Kandidaten …", buttons=[self._sc_run],
            )

        # -- helpers -------------------------------------------------------

        @staticmethod
        def _set_text(widget, text):
            widget.configure(state="normal")
            widget.delete("1.0", "end")
            widget.insert("1.0", text)
            widget.configure(state="disabled")

    return DeepVoiceDiveApp()


def _resolve_method(method: str) -> str:
    """Return ``method``, falling back from 'neural' to 'mfcc' if unavailable.

    Mirrors :func:`deepvoicedive.cli._resolve_method` so the GUI behaves like the
    command line when the neural backend is missing.
    """
    if method != "neural":
        return method
    try:
        from .neural import _get_encoder

        _get_encoder()
        return "neural"
    except Exception:  # noqa: BLE001
        return "mfcc"


def main(argv=None) -> int:
    """Entry point for the ``deepvoicedive-gui`` console script."""
    try:
        import tkinter  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        import sys

        print(
            "Fehler: Tkinter ist nicht verfügbar. Auf Linux ggf. das Paket "
            f"'python3-tk' installieren. ({exc})",
            file=sys.stderr,
        )
        return 1

    app = build_app()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
