"""Rekordbox Organizer - pick how you want your library organized, press Run."""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import organizer

APP_NAME = "Rekordbox Organizer"

IMPORT_STEPS = (
    "To bring the playlists into rekordbox:\n\n"
    "1. Preferences > Advanced > Database > rekordbox xml:\n"
    "   set 'Imported Library' to the new file.\n"
    "2. Preferences > View > Layout: enable 'rekordbox xml'.\n"
    "3. In the sidebar, open 'rekordbox xml', right-click the\n"
    "   '{folder}' folder and choose 'Import Playlist'."
)


class App(ttk.Frame):
    def __init__(self, master: tk.Tk):
        super().__init__(master, padding=16)
        self.master = master
        self.queue: queue.Queue = queue.Queue()
        self.output_path = ""

        self.xml_path = tk.StringVar()
        self.folder_name = tk.StringVar(value="Organized")
        defaults = organizer.Options()
        self.vars = {name: tk.BooleanVar(value=getattr(defaults, name)) for name in (
            "genre", "bpm", "key", "artist", "label", "year", "rating", "colour",
            "date_added", "duplicates", "missing_files", "missing_info", "never_played")}
        self.bpm_step = tk.IntVar(value=defaults.bpm_step)
        self.min_tracks = tk.IntVar(value=defaults.min_tracks)

        self._build()
        self.grid(sticky="nsew")
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)

    # ------------------------------------------------------------------ UI --

    def _build(self):
        self.columnconfigure(0, weight=1)

        ttk.Label(self, text=APP_NAME, font=("", 16, "bold")).grid(sticky="w")

        # Step 1 - file
        step1 = ttk.LabelFrame(self, text=" 1. Your rekordbox library ", padding=10)
        step1.grid(sticky="ew", pady=(12, 0))
        step1.columnconfigure(0, weight=1)
        ttk.Entry(step1, textvariable=self.xml_path).grid(row=0, column=0, sticky="ew")
        ttk.Button(step1, text="Browse...", command=self.browse).grid(row=0, column=1, padx=(8, 0))
        ttk.Label(step1, foreground="gray",
                  text="In rekordbox: File > Export Collection in xml format, then pick that file."
                  ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

        # Step 2 - options
        step2 = ttk.LabelFrame(self, text=" 2. Organize by ", padding=10)
        step2.grid(sticky="ew", pady=(12, 0))
        for c in range(2):
            step2.columnconfigure(c, weight=1)

        def check(parent, key, text, row, col):
            ttk.Checkbutton(parent, text=text, variable=self.vars[key]).grid(
                row=row, column=col, sticky="w", pady=2)

        check(step2, "genre", "Genre", 0, 0)
        bpm_row = ttk.Frame(step2)
        bpm_row.grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(bpm_row, text="BPM range, every", variable=self.vars["bpm"]).pack(side="left")
        ttk.Spinbox(bpm_row, from_=1, to=20, width=3, textvariable=self.bpm_step).pack(side="left", padx=4)
        ttk.Label(bpm_row, text="BPM").pack(side="left")
        check(step2, "key", "Key (Camelot)", 1, 0)
        check(step2, "year", "Release year", 1, 1)
        check(step2, "artist", "Artist", 2, 0)
        check(step2, "label", "Record label", 2, 1)
        check(step2, "rating", "Star rating", 3, 0)
        check(step2, "colour", "Track color", 3, 1)
        check(step2, "date_added", "Month added", 4, 0)
        min_row = ttk.Frame(step2)
        min_row.grid(row=4, column=1, sticky="w")
        ttk.Label(min_row, text="Artist/label needs at least").pack(side="left")
        ttk.Spinbox(min_row, from_=1, to=50, width=3, textvariable=self.min_tracks).pack(side="left", padx=4)
        ttk.Label(min_row, text="tracks").pack(side="left")

        # Cleanup
        step3 = ttk.LabelFrame(self, text=" 3. Cleanup playlists ", padding=10)
        step3.grid(sticky="ew", pady=(12, 0))
        for c in range(2):
            step3.columnconfigure(c, weight=1)
        check(step3, "duplicates", "Possible duplicates", 0, 0)
        check(step3, "missing_files", "Missing files", 0, 1)
        check(step3, "missing_info", "Missing genre / key / BPM", 1, 0)
        check(step3, "never_played", "Never played", 1, 1)

        # Run row
        run_row = ttk.Frame(self)
        run_row.grid(sticky="ew", pady=(14, 0))
        run_row.columnconfigure(2, weight=1)
        ttk.Label(run_row, text="Playlist folder name:").grid(row=0, column=0)
        ttk.Entry(run_row, textvariable=self.folder_name, width=18).grid(row=0, column=1, padx=(6, 0))
        self.run_btn = ttk.Button(run_row, text="Run", command=self.start, width=12)
        self.run_btn.grid(row=0, column=3, sticky="e")

        self.progress = ttk.Progressbar(self, mode="indeterminate")
        self.progress.grid(sticky="ew", pady=(10, 0))

        # Log
        self.rowconfigure(6, weight=1)
        self.log_box = tk.Text(self, height=9, wrap="word", state="disabled",
                               relief="flat", borderwidth=1, highlightthickness=1)
        self.log_box.grid(row=6, sticky="nsew", pady=(10, 0))

        self.open_btn = ttk.Button(self, text="Show output file", command=self.reveal_output)

    # ------------------------------------------------------------- actions --

    def browse(self):
        path = filedialog.askopenfilename(
            title="Choose your rekordbox XML export",
            filetypes=[("rekordbox XML", "*.xml"), ("All files", "*.*")])
        if path:
            self.xml_path.set(path)

    def log(self, msg: str):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def collect_options(self) -> organizer.Options:
        opt = organizer.Options(**{k: v.get() for k, v in self.vars.items()})
        try:
            opt.bpm_step = max(1, int(self.bpm_step.get()))
        except (tk.TclError, ValueError):
            opt.bpm_step = 5
        try:
            opt.min_tracks = max(1, int(self.min_tracks.get()))
        except (tk.TclError, ValueError):
            opt.min_tracks = 3
        opt.folder_name = self.folder_name.get().strip() or "Organized"
        return opt

    def start(self):
        path = self.xml_path.get().strip()
        if not path or not os.path.isfile(path):
            messagebox.showwarning(APP_NAME, "Please choose your rekordbox XML file first.")
            return
        opt = self.collect_options()

        self.run_btn.configure(state="disabled")
        self.open_btn.grid_forget()
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")
        self.progress.start(12)

        def work():
            try:
                out = organizer.run(path, opt, log=lambda m: self.queue.put(("log", m)))
                self.queue.put(("done", (out, opt.folder_name)))
            except Exception as exc:  # report anything to the user rather than crash
                self.queue.put(("error", str(exc)))

        threading.Thread(target=work, daemon=True).start()
        self.after(50, self.poll)

    def poll(self):
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "log":
                    self.log(payload)
                elif kind == "done":
                    self.finish()
                    self.output_path, folder = payload
                    self.open_btn.grid(row=7, sticky="w", pady=(8, 0))
                    self.log("")
                    self.log(IMPORT_STEPS.format(folder=folder))
                    messagebox.showinfo(APP_NAME, "Done!\n\n" + IMPORT_STEPS.format(folder=folder))
                    return
                elif kind == "error":
                    self.finish()
                    self.log("Error: " + payload)
                    messagebox.showerror(APP_NAME, payload)
                    return
        except queue.Empty:
            pass
        self.after(50, self.poll)

    def finish(self):
        self.progress.stop()
        self.run_btn.configure(state="normal")

    def reveal_output(self):
        if not self.output_path:
            return
        if sys.platform == "win32":
            subprocess.run(["explorer", "/select,", os.path.normpath(self.output_path)])
        elif sys.platform == "darwin":
            subprocess.run(["open", "-R", self.output_path])
        else:
            subprocess.run(["xdg-open", os.path.dirname(self.output_path)])


def main():
    if sys.platform == "win32":
        try:  # crisp text on high-DPI screens
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = tk.Tk()
    root.title(APP_NAME)
    root.minsize(560, 640)
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
