"""Rekordbox Organizer - pick how you want your library organized, press Run."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import organizer

APP_NAME = "Rekordbox Organizer"
MAX_PLAYLISTS = 30
SETTINGS_VERSION = 2

DIRECT_DONE = (
    "Your library has been updated.\n\n"
    "Open rekordbox to find your playlists in the '{folder}' folder in the sidebar."
)

XML_DONE = (
    "To bring the playlists into rekordbox:\n\n"
    "1. Preferences > Advanced > Database > rekordbox xml:\n"
    "   set 'Imported Library' to the new file.\n"
    "2. Preferences > View > Layout: enable 'rekordbox xml'.\n"
    "3. In the sidebar, open 'rekordbox xml', right-click the\n"
    "   '{folder}' folder and choose 'Import Playlist'."
)

EXAMPLE_PLAYLISTS = [
    organizer.CustomPlaylist("House", "house", "118-130", ""),
    organizer.CustomPlaylist("Techno", "techno", "130-145", ""),
    organizer.CustomPlaylist("Hip Hop", "hip hop", "", ""),
]


def settings_path() -> str:
    return os.path.join(organizer.app_data_dir(), "settings.json")


class ScrollFrame(ttk.Frame):
    """A frame with a vertical scrollbar; put widgets in .inner."""

    def __init__(self, master, height: int):
        super().__init__(master)
        self.canvas = tk.Canvas(self, height=height, highlightthickness=0, borderwidth=0)
        bar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>",
                        lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(window, width=e.width))
        self.canvas.configure(yscrollcommand=bar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        self.canvas.bind_all("<MouseWheel>", self._wheel, add="+")

    def _wheel(self, event):
        # Only scroll when the pointer is over this list (or a box inside it).
        widget = self.winfo_containing(event.x_root, event.y_root)
        while widget is not None and widget is not self:
            widget = widget.master
        if widget is None:
            return
        step = -1 * (event.delta // 120 if sys.platform == "win32" else event.delta)
        self.canvas.yview_scroll(step, "units")

    def sync_background(self, style: ttk.Style):
        self.canvas.configure(background=style.lookup("TFrame", "background") or None)


class PlaylistRow:
    def __init__(self, parent, index: int, data: organizer.CustomPlaylist):
        self.name = tk.StringVar(value=data.name)
        self.genres = tk.StringVar(value=data.genres)
        self.bpm = tk.StringVar(value=data.bpm)
        self.energy = tk.StringVar(value=data.energy)
        self.widgets = [
            ttk.Label(parent, text=f"{index}.", width=3, anchor="e"),
            ttk.Entry(parent, textvariable=self.name, width=18),
            ttk.Entry(parent, textvariable=self.genres, width=18),
            ttk.Entry(parent, textvariable=self.bpm, width=9),
            ttk.Entry(parent, textvariable=self.energy, width=7),
        ]
        for col, w in enumerate(self.widgets):
            w.grid(row=index, column=col, sticky="ew", padx=2, pady=2)

    def get(self) -> organizer.CustomPlaylist:
        return organizer.CustomPlaylist(self.name.get(), self.genres.get(),
                                        self.bpm.get(), self.energy.get())

    def destroy(self):
        for w in self.widgets:
            w.destroy()


class App(ttk.Frame):
    def __init__(self, master: tk.Tk):
        super().__init__(master, padding=16)
        self.master = master
        self.queue: queue.Queue = queue.Queue()
        self.output_path = ""

        self.source = tk.StringVar(value="direct")
        self.xml_path = tk.StringVar()
        self.folder_name = tk.StringVar(value="Organized")
        defaults = organizer.Options(custom_enabled=True)
        self.vars = {name: tk.BooleanVar(value=getattr(defaults, name)) for name in (
            "custom_enabled", "custom_leftovers", "energy", "energy_ratings", "keep_ratings",
            "genre", "bpm", "key", "artist", "label", "year", "rating", "colour",
            "date_added", "duplicates", "missing_files", "missing_info", "never_played")}
        self.bpm_step = tk.IntVar(value=defaults.bpm_step)
        self.min_tracks = tk.IntVar(value=defaults.min_tracks)
        self.playlist_count = tk.IntVar(value=len(EXAMPLE_PLAYLISTS))
        self.rows: list[PlaylistRow] = []
        # Remembers rows removed by lowering the count, so raising it brings them back.
        self.row_memory: list[organizer.CustomPlaylist] = list(EXAMPLE_PLAYLISTS)

        self.load_settings()
        self._build()
        self.grid(sticky="nsew")
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)
        for var in (self.source, self.vars["energy"], self.vars["energy_ratings"],
                    self.vars["custom_enabled"]):
            var.trace_add("write", lambda *_: self._sync_states())
        self.playlist_count.trace_add("write", lambda *_: self._resize_rows())
        self._sync_states()
        master.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------------------------------------------------------ UI --

    def _build(self):
        self.columnconfigure(0, weight=1)

        ttk.Label(self, text=APP_NAME, font=("", 16, "bold")).grid(sticky="w")

        # Step 1 - library
        step1 = ttk.LabelFrame(self, text=" 1. Your rekordbox library ", padding=10)
        step1.grid(sticky="ew", pady=(12, 0))
        step1.columnconfigure(0, weight=1)
        ttk.Radiobutton(step1, text="Straight from rekordbox (close rekordbox first)",
                        variable=self.source, value="direct").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Radiobutton(step1, text="From an XML export file",
                        variable=self.source, value="xml").grid(row=1, column=0, columnspan=2, sticky="w")
        self.xml_entry = ttk.Entry(step1, textvariable=self.xml_path)
        self.xml_entry.grid(row=2, column=0, sticky="ew", padx=(22, 0), pady=(4, 0))
        self.browse_btn = ttk.Button(step1, text="Browse...", command=self.browse)
        self.browse_btn.grid(row=2, column=1, padx=(8, 0), pady=(4, 0))
        self.source_hint = ttk.Label(step1, foreground="gray")
        self.source_hint.grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))

        def check(parent, key, text, row, col, **kw):
            cb = ttk.Checkbutton(parent, text=text, variable=self.vars[key])
            cb.grid(row=row, column=col, sticky="w", pady=2, **kw)
            return cb

        # Step 2 - energy
        step2 = ttk.LabelFrame(self, text=" 2. Energy ", padding=10)
        step2.grid(sticky="ew", pady=(12, 0))
        check(step2, "energy", "Listen to my music and rate its energy 1-5", 0, 0)
        self.energy_ratings_cb = check(step2, "energy_ratings",
                                       "Set each track's star rating to its energy level",
                                       1, 0, padx=(22, 0))
        self.keep_ratings_cb = check(step2, "keep_ratings",
                                     "Don't change tracks I've already rated",
                                     2, 0, padx=(44, 0))
        ttk.Label(step2, foreground="gray",
                  text="1 = calmest 20% of your library, 5 = most energetic 20%. Straight from "
                       "rekordbox it uses rekordbox's own track analysis, so it's quick.",
                  wraplength=540, justify="left").grid(row=3, column=0, sticky="w", pady=(4, 0))

        # Step 3 - playlists (tabs)
        ttk.Label(self, text="3. Playlists", font=("", 10, "bold")).grid(sticky="w", pady=(12, 2))
        tabs = ttk.Notebook(self)
        tabs.grid(sticky="ew")
        tabs.add(self._build_custom_tab(tabs, check), text="  My playlists  ")
        tabs.add(self._build_auto_tab(tabs, check), text="  Extra playlists (optional)  ")

        # Run row
        run_row = ttk.Frame(self)
        run_row.grid(sticky="ew", pady=(14, 0))
        run_row.columnconfigure(2, weight=1)
        ttk.Label(run_row, text="Main folder name:").grid(row=0, column=0)
        ttk.Entry(run_row, textvariable=self.folder_name, width=18).grid(row=0, column=1, padx=(6, 0))
        self.run_btn = ttk.Button(run_row, text="Run", command=self.start, width=12)
        self.run_btn.grid(row=0, column=3, sticky="e")

        self.progress = ttk.Progressbar(self, mode="determinate")
        self.progress.grid(sticky="ew", pady=(10, 0))

        # Log
        self.log_box = tk.Text(self, height=7, wrap="word", state="disabled",
                               relief="flat", borderwidth=1, highlightthickness=1)
        self.log_box.grid(sticky="nsew", pady=(10, 0))
        self.rowconfigure(self.log_box.grid_info()["row"], weight=1)

        self.open_btn = ttk.Button(self, text="Show output file", command=self.reveal_output)

    def _build_custom_tab(self, tabs, check) -> ttk.Frame:
        tab = ttk.Frame(tabs, padding=10)
        tab.columnconfigure(0, weight=1)

        top = ttk.Frame(tab)
        top.grid(sticky="ew")
        check(top, "custom_enabled", "Sort my library into these playlists", 0, 0)
        count_row = ttk.Frame(top)
        count_row.grid(row=0, column=1, sticky="e", padx=(16, 0))
        ttk.Label(count_row, text="How many:").pack(side="left")
        self.count_spin = ttk.Spinbox(count_row, from_=1, to=MAX_PLAYLISTS, width=4,
                                      textvariable=self.playlist_count)
        self.count_spin.pack(side="left", padx=(4, 0))

        self.rows_frame = ScrollFrame(tab, height=150)
        self.rows_frame.grid(sticky="ew", pady=(8, 0))
        self.rows_frame.sync_background(ttk.Style())
        inner = self.rows_frame.inner
        for col, text in enumerate(("", "Playlist name", "Genre has", "BPM", "Energy")):
            ttk.Label(inner, text=text, font=("", 9, "bold")).grid(row=0, column=col, sticky="w", padx=2)
        inner.columnconfigure(1, weight=2)
        inner.columnconfigure(2, weight=2)
        self._resize_rows()

        self.leftovers_cb = check(tab, "custom_leftovers",
                                  f"Also make an '{organizer.LEFTOVERS_NAME}' playlist for songs "
                                  "that fit nowhere", 2, 0)
        ttk.Label(tab, foreground="gray", wraplength=540, justify="left",
                  text="A song is added only if it matches every rule you fill in; songs that "
                       "don't fit are left alone. Genre is relaxed: 'house' also catches "
                       "Progressive House, Tech-House, Deep House... If a song fits several "
                       "playlists it goes to the best genre match. "
                       "BPM: 120-126, 128+ or 174. Energy (1-5): 4-5, 3+ or 2."
                  ).grid(row=3, column=0, sticky="w", pady=(4, 0))
        return tab

    def _build_auto_tab(self, tabs, check) -> ttk.Frame:
        tab = ttk.Frame(tabs, padding=10)
        for c in range(2):
            tab.columnconfigure(c, weight=1)
        check(tab, "genre", "Genre", 0, 0)
        bpm_row = ttk.Frame(tab)
        bpm_row.grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(bpm_row, text="BPM range, every", variable=self.vars["bpm"]).pack(side="left")
        ttk.Spinbox(bpm_row, from_=1, to=20, width=3, textvariable=self.bpm_step).pack(side="left", padx=4)
        ttk.Label(bpm_row, text="BPM").pack(side="left")
        check(tab, "key", "Key (Camelot)", 1, 0)
        check(tab, "year", "Release year", 1, 1)
        check(tab, "artist", "Artist", 2, 0)
        check(tab, "label", "Record label", 2, 1)
        check(tab, "rating", "Star rating", 3, 0)
        check(tab, "colour", "Track color", 3, 1)
        check(tab, "date_added", "Month added", 4, 0)
        min_row = ttk.Frame(tab)
        min_row.grid(row=4, column=1, sticky="w")
        ttk.Label(min_row, text="Artist/label needs at least").pack(side="left")
        ttk.Spinbox(min_row, from_=1, to=50, width=3, textvariable=self.min_tracks).pack(side="left", padx=4)
        ttk.Label(min_row, text="tracks").pack(side="left")
        ttk.Label(tab, text="Cleanup:", font=("", 9, "bold")).grid(row=5, column=0, sticky="w", pady=(8, 0))
        check(tab, "duplicates", "Possible duplicates", 6, 0)
        check(tab, "missing_files", "Missing files", 6, 1)
        check(tab, "missing_info", "Missing genre / key / BPM", 7, 0)
        check(tab, "never_played", "Never played", 7, 1)
        return tab

    def _resize_rows(self):
        try:
            want = max(1, min(MAX_PLAYLISTS, int(self.playlist_count.get())))
        except (tk.TclError, ValueError):
            return   # mid-typing; wait for a valid number
        self._remember_rows()
        while len(self.rows) > want:
            self.rows.pop().destroy()
        while len(self.rows) < want:
            i = len(self.rows)
            data = (self.row_memory[i] if i < len(self.row_memory)
                    else organizer.CustomPlaylist(f"Playlist {i + 1}"))
            self.rows.append(PlaylistRow(self.rows_frame.inner, i + 1, data))
        self._sync_states()

    def _remember_rows(self):
        for i, row in enumerate(self.rows):
            if i < len(self.row_memory):
                self.row_memory[i] = row.get()
            else:
                self.row_memory.append(row.get())

    def _sync_states(self):
        if not hasattr(self, "leftovers_cb"):   # still building the window
            return
        xml = self.source.get() == "xml"
        for w in (self.xml_entry, self.browse_btn):
            w.configure(state="normal" if xml else "disabled")
        self.source_hint.configure(text=(
            "In rekordbox: File > Export Collection in xml format, then pick that file."
            if xml else
            "Reads and updates your rekordbox library directly. A backup is made first."))
        energy = self.vars["energy"].get()
        self.energy_ratings_cb.configure(state="normal" if energy else "disabled")
        self.keep_ratings_cb.configure(
            state="normal" if energy and self.vars["energy_ratings"].get() else "disabled")
        custom = "normal" if self.vars["custom_enabled"].get() else "disabled"
        self.count_spin.configure(state=custom)
        self.leftovers_cb.configure(state=custom)
        for row in self.rows:
            for w in row.widgets[1:]:
                w.configure(state=custom)

    # ------------------------------------------------------------ settings --

    def load_settings(self):
        try:
            with open(settings_path(), encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return
        options = data.get("options", {})
        if data.get("version", 1) < SETTINGS_VERSION:
            # Older versions switched on extra playlists and rating changes by
            # default; start those fresh and keep only the user's own playlists.
            options = {k: v for k, v in options.items() if k == "custom_enabled"}
        for name, value in options.items():
            if name in self.vars and isinstance(value, bool):
                self.vars[name].set(value)
        for var, key in ((self.source, "source"), (self.xml_path, "xml_path"),
                         (self.folder_name, "folder_name")):
            if isinstance(data.get(key), str):
                var.set(data[key])
        for var, key in ((self.bpm_step, "bpm_step"), (self.min_tracks, "min_tracks")):
            if isinstance(data.get(key), int):
                var.set(data[key])
        playlists = data.get("playlists")
        if isinstance(playlists, list) and playlists:
            try:
                self.row_memory = [organizer.CustomPlaylist(**p) for p in playlists][:MAX_PLAYLISTS]
                self.playlist_count.set(len(self.row_memory))
            except TypeError:
                pass

    def save_settings(self):
        self._remember_rows()
        rows = [row.get() for row in self.rows]
        data = {
            "version": SETTINGS_VERSION,
            "options": {k: v.get() for k, v in self.vars.items()},
            "source": self.source.get(),
            "xml_path": self.xml_path.get(),
            "folder_name": self.folder_name.get(),
            "bpm_step": self._int(self.bpm_step, 5),
            "min_tracks": self._int(self.min_tracks, 3),
            "playlists": [vars(p) for p in rows],
        }
        try:
            os.makedirs(os.path.dirname(settings_path()), exist_ok=True)
            with open(settings_path(), "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError:
            pass   # settings are a convenience; never block the user over them

    def on_close(self):
        self.save_settings()
        self.master.destroy()

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

    @staticmethod
    def _int(var: tk.IntVar, fallback: int) -> int:
        try:
            return max(1, int(var.get()))
        except (tk.TclError, ValueError):
            return fallback

    def collect_options(self) -> organizer.Options:
        opt = organizer.Options(**{k: v.get() for k, v in self.vars.items()})
        opt.bpm_step = self._int(self.bpm_step, 5)
        opt.min_tracks = self._int(self.min_tracks, 3)
        opt.custom = [row.get() for row in self.rows]
        opt.folder_name = self.folder_name.get().strip() or "Organized"
        return opt

    def start(self):
        xml_path = None
        if self.source.get() == "xml":
            xml_path = self.xml_path.get().strip()
            if not xml_path or not os.path.isfile(xml_path):
                messagebox.showwarning(APP_NAME, "Please choose your rekordbox XML file first.")
                return
        opt = self.collect_options()
        try:
            organizer.compile_custom(opt)    # catch typos in the rules right away
        except ValueError as exc:
            messagebox.showwarning(APP_NAME, str(exc))
            return
        self.save_settings()
        if xml_path is None and not messagebox.askokcancel(
                APP_NAME, "This will update your rekordbox library directly.\n\n"
                          "Make sure rekordbox is closed. A backup of your library "
                          "is saved before anything is changed."):
            return

        self.run_btn.configure(state="disabled")
        self.open_btn.grid_forget()
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")
        self.progress.configure(mode="indeterminate", value=0)
        self.progress.start(12)

        def work():
            try:
                out = organizer.run(
                    opt, xml_path,
                    log=lambda m: self.queue.put(("log", m)),
                    progress=lambda done, total: self.queue.put(("progress", (done, total))))
                self.queue.put(("done", (out, opt.folder_name, xml_path is not None)))
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
                elif kind == "progress":
                    done, total = payload
                    if str(self.progress["mode"]) != "determinate":
                        self.progress.stop()
                        self.progress.configure(mode="determinate", maximum=total)
                    self.progress.configure(value=done)
                elif kind == "done":
                    self.finish()
                    self.output_path, folder, was_xml = payload
                    message = (XML_DONE if was_xml else DIRECT_DONE).format(folder=folder)
                    self.open_btn.configure(text="Show output file" if was_xml else "Show library folder")
                    self.open_btn.grid(sticky="w", pady=(8, 0))
                    self.log("")
                    self.log(message)
                    messagebox.showinfo(APP_NAME, "Done!\n\n" + message)
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
        self.progress.configure(mode="determinate", value=0)
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
    App(root)
    # Fit on small or scaled screens: the log box shrinks first.
    root.update_idletasks()
    height = min(root.winfo_reqheight(), root.winfo_screenheight() - 100)
    root.geometry(f"{root.winfo_reqwidth()}x{height}+40+20")
    root.minsize(600, min(560, height))
    root.mainloop()


if __name__ == "__main__":
    main()
