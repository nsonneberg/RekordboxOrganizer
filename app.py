"""Rekordbox Organizer - pick how you want your library organized, press Run."""

from __future__ import annotations

import datetime
import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import organizer
import updater
from version import VERSION

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
    TABS = ("Library", "My playlists", "Extra playlists", "Energy", "Cues", "Genres",
            "Set builder")

    def __init__(self, master: tk.Tk):
        super().__init__(master, padding=16)
        self.master = master
        self.queue: queue.Queue = queue.Queue()
        self.output_path = ""

        self.source = tk.StringVar(value="direct")
        self.xml_path = tk.StringVar()
        self.folder_name = tk.StringVar(value="Organized")
        self.date_mode = tk.StringVar(value="all")      # all / since_last / range
        self.date_from = tk.StringVar()
        self.date_to = tk.StringVar()
        self.last_run = ""                               # YYYY-MM-DD of the last successful run
        self.tab = 0
        defaults = organizer.Options(custom_enabled=True)
        self.vars = {name: tk.BooleanVar(value=getattr(defaults, name)) for name in (
            "custom_enabled", "custom_leftovers", "guess_genres", "auto_genres", "merge_existing",
            "move_mismatched",
            "energy", "energy_ratings", "keep_ratings", "energy_tags",
            "phrase_cues", "cue_colours", "memory_cues", "set_build",
            "genre", "bpm", "key", "artist", "label", "year", "rating", "colour",
            "date_added", "duplicates", "missing_files", "missing_info", "never_played")}
        # Other settings, saved as they are: {Options field: Tk variable}
        self.values = {
            "bpm_step": tk.IntVar(value=defaults.bpm_step),
            "min_tracks": tk.IntVar(value=defaults.min_tracks),
            "set_name": tk.StringVar(value=defaults.set_name),
            "set_start": tk.IntVar(value=defaults.set_start),
            "set_end": tk.IntVar(value=defaults.set_end),
            "set_length": tk.IntVar(value=defaults.set_length),
            "set_genres": tk.StringVar(value=defaults.set_genres),
            "set_bpm": tk.StringVar(value=defaults.set_bpm),
        }
        self.playlist_count = tk.IntVar(value=len(EXAMPLE_PLAYLISTS))
        self.rows: list[PlaylistRow] = []
        # Remembers rows removed by lowering the count, so raising it brings them back.
        self.row_memory: list[organizer.CustomPlaylist] = list(EXAMPLE_PLAYLISTS)

        self.load_settings()
        self._build()
        self.grid(sticky="nsew")
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)
        for var in (self.source, self.date_mode, *self.vars.values()):
            var.trace_add("write", lambda *_: self._sync_states())
        self.playlist_count.trace_add("write", lambda *_: self._resize_rows())
        self._sync_states()
        master.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------------------------------------------------------ UI --

    def _check(self, parent, key, text, row, col=0, **kw):
        cb = ttk.Checkbutton(parent, text=text, variable=self.vars[key])
        cb.grid(row=row, column=col, sticky="w", pady=2, **kw)
        return cb

    @staticmethod
    def _hint(parent, text, row, col=0, **kw):
        label = ttk.Label(parent, text=text, foreground="gray", wraplength=640, justify="left")
        label.grid(row=row, column=col, columnspan=kw.pop("columnspan", 1), sticky="w",
                   pady=kw.pop("pady", (2, 6)), **kw)
        return label

    def _build(self):
        self.columnconfigure(0, weight=1)

        title = ttk.Frame(self)
        title.grid(sticky="ew")
        title.columnconfigure(1, weight=1)
        ttk.Label(title, text=APP_NAME, font=("", 16, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(title, text="version " + VERSION if VERSION[0].isdigit() and "dev" not in VERSION
                  else "running from source", foreground="gray"
                  ).grid(row=0, column=2, sticky="e")

        # Shown when a newer version is out. It lives inside the title frame:
        # hidden in the main grid, its row would be taken by the next section.
        self.update_bar = ttk.Frame(title)
        self.update_label = ttk.Label(self.update_bar, foreground="#0a6")
        self.update_label.grid(row=0, column=0, sticky="w")
        self.update_btn = ttk.Button(self.update_bar, text="Update now", command=self.update_app)
        self.update_btn.grid(row=0, column=1, padx=(10, 0))
        self.update_bar.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        self.update_bar.grid_remove()

        self.tabs = ttk.Notebook(self)
        self.tabs.grid(sticky="nsew", pady=(12, 0))
        for name, build in zip(self.TABS, (
                self._tab_library, self._tab_custom, self._tab_auto, self._tab_energy,
                self._tab_cues, self._tab_genres, self._tab_set)):
            frame = ttk.Frame(self.tabs, padding=12)
            frame.columnconfigure(0, weight=1)
            build(frame)
            self.tabs.add(frame, text=f"  {name}  ")
        try:
            self.tabs.select(self.tab)
        except tk.TclError:
            pass

        # Run row
        run_row = ttk.Frame(self)
        run_row.grid(sticky="ew", pady=(12, 0))
        run_row.columnconfigure(0, weight=1)
        self.summary = ttk.Label(run_row, foreground="gray")
        self.summary.grid(row=0, column=0, sticky="w")
        buttons = ttk.Frame(run_row)
        buttons.grid(row=0, column=1, sticky="e")
        self.undo_btn = ttk.Button(buttons, text="Undo last run...", command=self.undo)
        self.undo_btn.pack(side="left")
        self.preview_btn = ttk.Button(buttons, text="Preview", width=10,
                                      command=lambda: self.start(preview=True))
        self.preview_btn.pack(side="left", padx=(8, 0))
        self.run_btn = ttk.Button(buttons, text="Run", command=self.start, width=12)
        self.run_btn.pack(side="left", padx=(8, 0))

        self.progress = ttk.Progressbar(self, mode="determinate")
        self.progress.grid(sticky="ew", pady=(10, 0))

        # Log
        self.log_box = tk.Text(self, height=7, wrap="word", state="disabled",
                               relief="flat", borderwidth=1, highlightthickness=1)
        self.log_box.grid(sticky="nsew", pady=(10, 0))
        self.rowconfigure(self.log_box.grid_info()["row"], weight=1)

        self.open_btn = ttk.Button(self, text="Show output file", command=self.reveal_output)

    def _tab_library(self, tab):
        ttk.Radiobutton(tab, text="Straight from rekordbox (close rekordbox before pressing Run)",
                        variable=self.source, value="direct").grid(row=0, column=0, columnspan=2,
                                                                   sticky="w")
        ttk.Radiobutton(tab, text="From an XML export file",
                        variable=self.source, value="xml").grid(row=1, column=0, columnspan=2,
                                                                sticky="w")
        self.xml_entry = ttk.Entry(tab, textvariable=self.xml_path)
        self.xml_entry.grid(row=2, column=0, sticky="ew", padx=(22, 0), pady=(4, 0))
        self.browse_btn = ttk.Button(tab, text="Browse...", command=self.browse)
        self.browse_btn.grid(row=2, column=1, padx=(8, 0), pady=(4, 0))
        self.source_hint = self._hint(tab, "", 3, columnspan=2)

        dates = ttk.Frame(tab)
        dates.grid(row=4, column=0, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Label(dates, text="Songs to organize:").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(dates, text="All", variable=self.date_mode,
                        value="all").grid(row=0, column=1, padx=(8, 0))
        self.since_rb = ttk.Radiobutton(dates, variable=self.date_mode, value="since_last")
        self.since_rb.grid(row=0, column=2, padx=(8, 0))
        ttk.Radiobutton(dates, text="Added from", variable=self.date_mode,
                        value="range").grid(row=0, column=3, padx=(8, 0))
        self.from_entry = ttk.Entry(dates, textvariable=self.date_from, width=11)
        self.from_entry.grid(row=0, column=4, padx=(4, 0))
        ttk.Label(dates, text="to").grid(row=0, column=5, padx=4)
        self.to_entry = ttk.Entry(dates, textvariable=self.date_to, width=11)
        self.to_entry.grid(row=0, column=6)
        self._hint(tab, "Dates as YYYY-MM-DD; either can be empty. The rest of the library is "
                        "still read for comparison.", 5, columnspan=2)

        folder = ttk.Frame(tab)
        folder.grid(row=6, column=0, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Label(folder, text="Main folder name:").pack(side="left")
        ttk.Entry(folder, textvariable=self.folder_name, width=20).pack(side="left", padx=(6, 0))
        self._check(tab, "merge_existing", "Add to existing folders / playlists with the same "
                                          "name (instead of replacing them)", 7)

    def _tab_custom(self, tab):
        top = ttk.Frame(tab)
        top.grid(sticky="ew")
        self._check(top, "custom_enabled", "Sort my library into these playlists", 0)
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

        self.leftovers_cb = self._check(tab, "custom_leftovers",
                                        f"Also make an '{organizer.LEFTOVERS_NAME}' playlist for "
                                        "songs that fit nowhere", 2)
        self.move_cb = self._check(tab, "move_mismatched",
                                   "When adding to existing playlists, move songs to the "
                                   "playlist they now fit best", 3)
        self._hint(tab, "A song is added only if it matches every rule you fill in. Genre is "
                        "relaxed: 'house' also catches Tech-House, Deep House... and, with "
                        "guessing on, untagged songs that sound like house. "
                        "BPM: 120-126, 128+ or 174. Energy (1-5): 4-5, 3+ or 2.", 4)

    def _tab_auto(self, tab):
        for c in range(2):
            tab.columnconfigure(c, weight=1)
        self._check(tab, "genre", "Genre", 0, 0)
        bpm_row = ttk.Frame(tab)
        bpm_row.grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(bpm_row, text="BPM range, every", variable=self.vars["bpm"]).pack(side="left")
        ttk.Spinbox(bpm_row, from_=1, to=20, width=3,
                    textvariable=self.values["bpm_step"]).pack(side="left", padx=4)
        ttk.Label(bpm_row, text="BPM").pack(side="left")
        self._check(tab, "key", "Key (Camelot)", 1, 0)
        self._check(tab, "year", "Release year", 1, 1)
        self._check(tab, "artist", "Artist", 2, 0)
        self._check(tab, "label", "Record label", 2, 1)
        self._check(tab, "rating", "Star rating", 3, 0)
        self._check(tab, "colour", "Track color", 3, 1)
        self._check(tab, "date_added", "Month added", 4, 0)
        min_row = ttk.Frame(tab)
        min_row.grid(row=4, column=1, sticky="w")
        ttk.Label(min_row, text="Artist/label needs at least").pack(side="left")
        ttk.Spinbox(min_row, from_=1, to=50, width=3,
                    textvariable=self.values["min_tracks"]).pack(side="left", padx=4)
        ttk.Label(min_row, text="tracks").pack(side="left")
        ttk.Label(tab, text="Cleanup:", font=("", 9, "bold")).grid(row=5, column=0, sticky="w",
                                                                  pady=(10, 0))
        self._check(tab, "duplicates", "Duplicates (and which copies to remove)", 6, 0)
        self._check(tab, "missing_files", "Missing files", 6, 1)
        self._check(tab, "missing_info", "Missing genre / key / BPM", 7, 0)
        self._check(tab, "never_played", "Never played", 7, 1)
        self._hint(tab, f"'{organizer.EXTRA_COPIES_NAME}' lists the copies you can remove: for "
                        "each song the playable copy with the most cues and plays is kept. In "
                        "rekordbox, select them and choose 'Remove from Collection'. The app "
                        "never deletes anything.", 8, columnspan=2)

    def _tab_energy(self, tab):
        self._check(tab, "energy", "Rate each song's energy 1-5", 0)
        self.energy_ratings_cb = self._check(tab, "energy_ratings",
                                             "Set its star rating to the energy level", 1,
                                             padx=(22, 0))
        self.keep_ratings_cb = self._check(tab, "keep_ratings", "Keep ratings I've already set",
                                           2, padx=(44, 0))
        self.energy_tags_cb = self._check(tab, "energy_tags",
                                          "Tag it 'Energy 1-5' in My Tag (your stars stay yours "
                                          "if you untick the star rating)", 3, padx=(22, 0))
        self._hint(tab, "Judged in each track's drops and choruses with rekordbox's own "
                        "analysis. 5 = your most energetic 20%, 1 = your calmest. A 'By Energy' "
                        "folder is made too. My Tags go in a column named 'Energy' (rekordbox's "
                        "unused 'Untitled Column' is renamed if needed) and need 'Straight from "
                        "rekordbox'.", 4)

    def _tab_cues(self, tab):
        self.cues_cb = self._check(tab, "phrase_cues",
                                   "Hot cues where sections start (Intro, Build-up, Drop, "
                                   "Breakdown, Chorus, Outro)", 0)
        self.colours_cb = self._check(tab, "cue_colours",
                                      "Colour them by section (drops red, build-ups orange, "
                                      "breakdowns blue, intro/outro green)", 1, padx=(22, 0))
        self.memory_cb = self._check(tab, "memory_cues",
                                     "Memory cues to mix in (where the intro ends) and mix out "
                                     "(where the outro starts)", 2)
        self._hint(tab, "Uses rekordbox's phrase analysis (Preferences > Analysis > Phrase). "
                        "Only empty pads (A-H) are used, your own cues are never moved, and "
                        "running again doesn't add the same cue twice. Works straight from "
                        "rekordbox only.", 3)

    def _tab_genres(self, tab):
        self._check(tab, "guess_genres", "Guess genres for untagged / vague songs", 0)
        self._check(tab, "auto_genres", "Auto genre detection (a folder per genre)", 1)
        self._hint(tab, "rekordbox doesn't detect genre, so guesses are estimates from BPM and "
                        "sound, learned from your tagged songs. Guessed songs go in "
                        "'(detected)' playlists so you can check them.", 2)
        row = ttk.Frame(tab)
        row.grid(row=3, column=0, sticky="w", pady=(10, 0))
        self.review_btn = ttk.Button(row, text="Review genres...", command=self.review_genres)
        self.review_btn.pack(side="left")
        ttk.Label(row, text="Check the guesses and save the right genres into rekordbox.",
                  foreground="gray").pack(side="left", padx=(10, 0))
        self._hint(tab, "Saved genres become real tags, so every later guess learns from them.", 4)

    def _tab_set(self, tab):
        self._check(tab, "set_build", "Build a set: one playlist in mixable order", 0)
        form = ttk.Frame(tab)
        form.grid(row=1, column=0, sticky="w", padx=(22, 0), pady=(4, 0))
        self.set_widgets = []

        def field(label, widget, row, col, hint=""):
            ttk.Label(form, text=label).grid(row=row, column=col, sticky="w", pady=3,
                                             padx=(0 if col == 0 else 18, 6))
            widget.grid(row=row, column=col + 1, sticky="w", pady=3)
            if hint:
                ttk.Label(form, text=hint, foreground="gray").grid(row=row, column=col + 2,
                                                                   sticky="w", padx=(6, 0))
            self.set_widgets.append(widget)

        v = self.values
        field("Name:", ttk.Entry(form, textvariable=v["set_name"], width=20), 0, 0)
        field("Songs:", ttk.Spinbox(form, from_=2, to=200, width=5,
                                    textvariable=v["set_length"]), 0, 2)
        field("Start energy:", ttk.Spinbox(form, from_=1, to=5, width=5,
                                           textvariable=v["set_start"]), 1, 0)
        field("End energy:", ttk.Spinbox(form, from_=1, to=5, width=5,
                                         textvariable=v["set_end"]), 1, 2)
        field("Genre has:", ttk.Entry(form, textvariable=v["set_genres"], width=20), 2, 0)
        field("BPM:", ttk.Entry(form, textvariable=v["set_bpm"], width=10), 2, 2, "e.g. 120-128")
        self._hint(tab, "Picks songs from your whole library so each one mixes into the next: "
                        "a compatible key (Camelot wheel), a BPM within a few percent, and "
                        "energy moving from the start level to the end level. The set goes in "
                        f"'{organizer.SETS_FOLDER}' and is rewritten each run. Leave Genre and "
                        "BPM empty for any.", 2, pady=(8, 0))

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
        if not hasattr(self, "set_widgets"):   # still building the window
            return
        on = lambda flag: "normal" if flag else "disabled"
        v = {k: var.get() for k, var in self.vars.items()}
        xml = self.source.get() == "xml"
        for w in (self.xml_entry, self.browse_btn):
            w.configure(state=on(xml))
        self.source_hint.configure(text=(
            "In rekordbox: File > Export Collection in xml format, then pick that file."
            if xml else
            "Reads and updates your rekordbox library directly. A backup is made first."))
        self.since_rb.configure(text=f"Added since last run ({self.last_run})"
                                if self.last_run else "Added since last run (none yet)")
        for w in (self.from_entry, self.to_entry):
            w.configure(state=on(self.date_mode.get() == "range"))
        self.cues_cb.configure(state=on(not xml))
        self.memory_cb.configure(state=on(not xml))
        self.colours_cb.configure(state=on(not xml and v["phrase_cues"]))
        self.undo_btn.configure(state=on(not xml))
        self.energy_ratings_cb.configure(state=on(v["energy"]))
        self.keep_ratings_cb.configure(state=on(v["energy"] and v["energy_ratings"]))
        self.energy_tags_cb.configure(state=on(v["energy"] and not xml))
        custom = on(v["custom_enabled"])
        self.count_spin.configure(state=custom)
        self.leftovers_cb.configure(state=custom)
        self.move_cb.configure(state=on(v["merge_existing"]))
        for row in self.rows:
            for w in row.widgets[1:]:
                w.configure(state=custom)
        for w in self.set_widgets:
            w.configure(state=on(v["set_build"]))

        # What Run will do, in a few words, so it shows from any tab.
        doing = []
        if v["custom_enabled"]:
            doing.append("my playlists")
        extras = ("genre", "bpm", "key", "artist", "label", "year", "rating", "colour",
                  "date_added", "duplicates", "missing_files", "missing_info", "never_played")
        if any(v[k] for k in extras) or v["auto_genres"]:
            doing.append("extra playlists")
        if v["energy"]:
            doing.append("energy" + (" + stars" if v["energy_ratings"] else "")
                         + (" + My Tag" if v["energy_tags"] and not xml else ""))
        if not xml and (v["phrase_cues"] or v["memory_cues"]):
            doing.append("cues")
        if v["set_build"]:
            doing.append("a set")
        self.summary.configure(text=("Run will do: " + ", ".join(doing)) if doing
                               else "Nothing ticked yet.")

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
                         (self.folder_name, "folder_name"), (self.date_mode, "date_mode"),
                         (self.date_from, "date_from"), (self.date_to, "date_to")):
            if isinstance(data.get(key), str):
                var.set(data[key])
        if isinstance(data.get("last_run"), str):
            self.last_run = data["last_run"]
        if isinstance(data.get("tab"), int):
            self.tab = data["tab"]
        values = dict(data.get("values", {}))
        for key in ("bpm_step", "min_tracks"):      # saved at the top level before 1.2
            values.setdefault(key, data.get(key))
        for key, var in self.values.items():
            value = values.get(key)
            if isinstance(var, tk.IntVar) and isinstance(value, int) and not isinstance(value, bool):
                var.set(value)
            elif isinstance(var, tk.StringVar) and isinstance(value, str):
                var.set(value)
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
            "date_mode": self.date_mode.get(),
            "date_from": self.date_from.get(),
            "date_to": self.date_to.get(),
            "last_run": self.last_run,
            "tab": self._current_tab(),
            "values": self._values(),
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

    def _current_tab(self) -> int:
        try:
            return self.tabs.index(self.tabs.select())
        except (AttributeError, tk.TclError):
            return self.tab

    def _values(self) -> dict:
        """The non-tickbox settings; a number box being typed in falls back to its default."""
        defaults = organizer.Options()
        out = {}
        for key, var in self.values.items():
            try:
                out[key] = var.get()
            except (tk.TclError, ValueError):
                out[key] = getattr(defaults, key)
        for key in ("bpm_step", "min_tracks"):
            out[key] = max(1, out[key])
        return out

    def collect_options(self) -> organizer.Options:
        opt = organizer.Options(**{k: v.get() for k, v in self.vars.items()}, **self._values())
        opt.custom = [row.get() for row in self.rows]
        opt.folder_name = self.folder_name.get().strip() or "Organized"
        mode = self.date_mode.get()
        if mode == "since_last":
            opt.date_from = self.last_run      # "" (never run) = all songs
        elif mode == "range":
            opt.date_from, opt.date_to = self.date_from.get(), self.date_to.get()
        return opt

    def start(self, preview: bool = False):
        xml_path = None
        if self.source.get() == "xml":
            xml_path = self.xml_path.get().strip()
            if not xml_path or not os.path.isfile(xml_path):
                messagebox.showwarning(APP_NAME, "Please choose your rekordbox XML file first.")
                return
        opt = self.collect_options()
        try:
            organizer.compile_custom(opt)    # catch typos in the rules right away
            organizer.select_tracks([], opt)  # ...and in the dates
            if opt.set_build:
                import setbuilder
                setbuilder.check(organizer.set_request(opt))
        except ValueError as exc:
            messagebox.showwarning(APP_NAME, str(exc))
            return
        self.save_settings()
        if not preview and xml_path is None and not messagebox.askokcancel(
                APP_NAME, "This will update your rekordbox library directly.\n\n"
                          "Make sure rekordbox is closed. A backup of your library "
                          "is saved before anything is changed."):
            return

        self._busy(True)
        self.open_btn.grid_forget()
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")
        self.progress.configure(mode="indeterminate", value=0)
        self.progress.start(12)

        if preview:
            self.log("Preview: working on a copy. Nothing in your library is changed.")

        def work():
            try:
                changes = organizer.Changes()
                out = organizer.run(
                    opt, xml_path,
                    log=lambda m: self.queue.put(("log", m)),
                    progress=lambda done, total: self.queue.put(("progress", (done, total))),
                    preview=preview, changes=changes)
                if preview:
                    self.queue.put(("preview", changes))
                else:
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
                elif kind == "preview":
                    self.finish()
                    self.show_preview(payload)
                    return
                elif kind == "done":
                    self.finish()
                    self.last_run = datetime.date.today().isoformat()
                    self._sync_states()
                    self.save_settings()
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

    # -------------------------------------------------------------- genres --

    def review_genres(self):
        xml_path = None
        if self.source.get() == "xml":
            xml_path = self.xml_path.get().strip()
            if not xml_path or not os.path.isfile(xml_path):
                messagebox.showwarning(APP_NAME, "Please choose your rekordbox XML file first.")
                return
        self._busy(True)
        self.review_btn.configure(state="disabled")
        self.log("Looking for songs with no genre or a vague one ...")
        self.progress.configure(mode="indeterminate")
        self.progress.start(12)
        found: queue.Queue = queue.Queue()

        def work():
            try:
                found.put(("ok", organizer.genre_review(
                    xml_path, log=lambda m: found.put(("log", m)))))
            except Exception as exc:
                found.put(("error", str(exc)))

        def wait():
            try:
                while True:
                    kind, payload = found.get_nowait()
                    if kind == "log":
                        self.log(payload)
                        continue
                    self.finish()
                    self.review_btn.configure(state="normal")
                    if kind == "error":
                        messagebox.showerror(APP_NAME, payload)
                    elif not payload[0]:
                        messagebox.showinfo(APP_NAME, "Every song already has a specific genre.")
                    else:
                        self.show_genre_review(*payload, can_save=xml_path is None)
                    return
            except queue.Empty:
                self.after(100, wait)

        threading.Thread(target=work, daemon=True).start()
        self.after(100, wait)

    def show_genre_review(self, rows, choices: list[str], can_save: bool):
        win = tk.Toplevel(self.master)
        win.title(f"{APP_NAME} - Review genres")
        win.transient(self.master)
        frame = ttk.Frame(win, padding=12)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)
        ttk.Label(frame, wraplength=760, justify="left",
                  text="These songs have no genre or a vague one. 'Save as' starts as the "
                       "app's guess: select songs and pick the right genre, or Clear the ones "
                       "you'd rather leave alone. Copies of a song change together."
                  ).grid(row=0, column=0, columnspan=2, sticky="w")

        cols = ("song", "artist", "bpm", "tag", "guess", "save")
        tree = ttk.Treeview(frame, columns=cols, show="headings", height=16)
        for col, text, width in (("song", "Song", 300), ("artist", "Artist", 140),
                                 ("bpm", "BPM", 50), ("tag", "Genre now", 110),
                                 ("guess", "Guess", 100), ("save", "Save as", 120)):
            tree.heading(col, text=text)
            tree.column(col, width=width, anchor="e" if col == "bpm" else "w",
                        stretch=col == "song")
        bar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=bar.set)
        tree.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        bar.grid(row=1, column=1, sticky="ns", pady=(8, 0))

        save_as = {r.id: r.guess for r in rows}
        by_id = {r.id: r for r in rows}
        copies: dict = {}
        for r in rows:
            key = organizer.song_key(organizer.Track(r.id, r.name, r.artist, "", "", r.bpm, "",
                                                     0, 0, "", "", 0, ""))
            copies.setdefault(key, []).append(r.id)
        copies_of = {i: ids for ids in copies.values() for i in ids}
        for r in rows:
            tree.insert("", "end", iid=r.id, values=(
                r.name, r.artist, f"{r.bpm:.0f}" if r.bpm else "", r.tag or "(none)",
                r.guess or "?", r.guess))

        controls = ttk.Frame(frame)
        controls.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        ttk.Label(controls, text="Genre for the selected songs:").pack(side="left")
        genre = tk.StringVar()
        ttk.Combobox(controls, textvariable=genre, values=choices, width=22).pack(side="left",
                                                                                  padx=(6, 0))
        count = ttk.Label(frame, foreground="gray")
        count.grid(row=3, column=0, sticky="w", pady=(8, 0))

        def refresh():
            for i, value in save_as.items():
                tree.set(i, "save", value)
            n = sum(1 for v in save_as.values() if v.strip())
            count.configure(text=f"{n} of {len(rows)} songs will be saved.")
            save_btn.configure(state="normal" if n and can_save else "disabled")

        def selected():
            return {c for i in tree.selection() for c in copies_of.get(i, [i])}

        def apply(value):
            for i in selected():
                save_as[i] = value
            refresh()

        ttk.Button(controls, text="Apply", command=lambda: apply(genre.get().strip())
                   ).pack(side="left", padx=(6, 0))
        ttk.Button(controls, text="Clear", command=lambda: apply("")).pack(side="left",
                                                                          padx=(6, 0))
        ttk.Button(controls, text="Use the guesses for all",
                   command=lambda: (save_as.update({r.id: r.guess for r in rows}), refresh())
                   ).pack(side="left", padx=(18, 0))

        def pick_up(_event=None):
            ids = list(tree.selection())
            if ids:
                genre.set(save_as.get(ids[0]) or by_id[ids[0]].guess)

        tree.bind("<<TreeviewSelect>>", pick_up)

        buttons = ttk.Frame(frame)
        buttons.grid(row=3, column=0, columnspan=2, sticky="e", pady=(8, 0))
        ttk.Button(buttons, text="Close", command=win.destroy).pack(side="left")

        def save():
            new = {i: v.strip() for i, v in save_as.items() if v.strip()}
            if not messagebox.askokcancel(
                    APP_NAME, f"Save the genre of {len(new)} songs into rekordbox?\n\n"
                              "Make sure rekordbox is closed. A backup of your library is "
                              "saved first, and Undo last run can put it back.", parent=win):
                return
            try:
                organizer.save_genres(new, log=self.log)
            except Exception as exc:
                messagebox.showerror(APP_NAME, str(exc), parent=win)
                return
            win.destroy()
            messagebox.showinfo(APP_NAME, "Saved. Open rekordbox to see the new genres.")

        save_btn = ttk.Button(buttons, text="Save to rekordbox", command=save)
        save_btn.pack(side="left", padx=(8, 0))
        if not can_save:
            ttk.Label(frame, foreground="gray", text="Saving needs 'Straight from rekordbox' "
                      "on the Library tab.").grid(row=4, column=0, sticky="w")
        refresh()

    # ------------------------------------------------------------- updates --

    def check_for_update(self):
        """Look for a newer release in the background (built apps only)."""
        updater.cleanup_old()
        if not updater.can_self_update():
            return
        found: queue.Queue = queue.Queue()

        def work():
            try:
                found.put(updater.check())
            except Exception as exc:   # offline or GitHub unreachable: try next time
                found.put(None)
                try:
                    with open(os.path.join(organizer.app_data_dir(), "update-check.log"),
                              "w", encoding="utf-8") as f:
                        f.write(f"{type(exc).__name__}: {exc}\n")
                except OSError:
                    pass

        def wait():
            try:
                self.new_release = found.get_nowait()
            except queue.Empty:
                self.after(300, wait)
                return
            if self.new_release:
                self.update_label.configure(
                    text=f"Version {self.new_release['version']} is available.")
                self.update_bar.grid()

        threading.Thread(target=work, daemon=True).start()
        self.after(300, wait)

    def update_app(self):
        info = getattr(self, "new_release", None)
        if not info:
            return
        if not info["url"]:
            import webbrowser
            webbrowser.open(info["page"])
            return
        notes = info["notes"].strip()
        if len(notes) > 600:
            notes = notes[:600].rstrip() + " ..."
        if not messagebox.askokcancel(
                APP_NAME, f"Install version {info['version']}? The app will restart."
                          + (f"\n\nWhat's new:\n{notes}" if notes else "")):
            return
        self.save_settings()
        self.update_btn.configure(state="disabled")
        self._busy(True)
        self.update_label.configure(text=f"Downloading version {info['version']} ...")
        self.progress.configure(mode="determinate", value=0, maximum=1)
        status: queue.Queue = queue.Queue()

        def work():
            try:
                updater.install(info["url"], lambda done, total: status.put(("progress", (done, total))))
                status.put(("done", None))
            except Exception as exc:
                status.put(("error", str(exc)))

        def poll():
            try:
                while True:
                    kind, payload = status.get_nowait()
                    if kind == "progress":
                        done, total = payload
                        self.progress.configure(maximum=max(total, done, 1), value=done)
                    elif kind == "done":
                        self.master.destroy()     # the new version is starting
                        return
                    else:
                        self.progress.configure(value=0)
                        self.update_btn.configure(state="normal")
                        self._busy(False)
                        self.update_label.configure(
                            text=f"Version {info['version']} is available.")
                        if messagebox.askyesno(
                                APP_NAME, f"The update didn't work:\n{payload}\n\n"
                                          "Open the download page instead?"):
                            import webbrowser
                            webbrowser.open(info["page"])
                        return
            except queue.Empty:
                pass
            self.after(100, poll)

        threading.Thread(target=work, daemon=True).start()
        self.after(100, poll)

    def _busy(self, busy: bool):
        for w in (self.run_btn, self.preview_btn, self.undo_btn):
            w.configure(state="disabled" if busy else "normal")
        if not busy:
            self._sync_states()

    def finish(self):
        self.progress.stop()
        self.progress.configure(mode="determinate", value=0)
        self._busy(False)

    # ------------------------------------------------------ preview / undo --

    def show_preview(self, changes: organizer.Changes):
        win = tk.Toplevel(self.master)
        win.title(f"{APP_NAME} - Preview")
        win.transient(self.master)
        frame = ttk.Frame(win, padding=12)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)
        ttk.Label(frame, text="This is what Run would change. Nothing has been changed yet.",
                  font=("", 10, "bold")).grid(row=0, column=0, columnspan=2, sticky="w")
        text = tk.Text(frame, width=100, height=28, wrap="none", relief="flat",
                       borderwidth=1, highlightthickness=1)
        bar = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=bar.set)
        text.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        bar.grid(row=1, column=1, sticky="ns", pady=(8, 0))
        text.insert("1.0", organizer.summarize_changes(changes))
        text.configure(state="disabled")
        buttons = ttk.Frame(frame)
        buttons.grid(row=2, column=0, columnspan=2, sticky="e", pady=(10, 0))
        ttk.Button(buttons, text="Close", command=win.destroy).pack(side="left")

        def run_now():
            win.destroy()
            self.start()

        if not changes.empty():
            ttk.Button(buttons, text="Run now", command=run_now).pack(side="left", padx=(8, 0))

    def undo(self):
        import rbdb
        try:
            db_path = rbdb.library_path()
        except Exception as exc:
            messagebox.showerror(APP_NAME, str(exc))
            return
        backups = rbdb.list_backups(os.path.dirname(db_path))
        if not backups:
            messagebox.showinfo(APP_NAME, "There's nothing to undo: no backups from this app "
                                          "were found.")
            return

        win = tk.Toplevel(self.master)
        win.title(f"{APP_NAME} - Undo")
        win.transient(self.master)
        frame = ttk.Frame(win, padding=12)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, wraplength=440, justify="left",
                  text="Put your library back to how it was just before a run. Everything "
                       "changed in rekordbox since then (by this app or by you) is undone too."
                  ).pack(anchor="w")
        ttk.Label(frame, text="Before the run on:", font=("", 9, "bold")).pack(anchor="w",
                                                                               pady=(10, 2))
        box = tk.Listbox(frame, height=min(len(backups), 6), exportselection=False)
        for path in backups:
            box.insert("end", rbdb.backup_time(path).strftime("%A %d %B %Y, %H:%M:%S"))
        box.selection_set(0)
        box.pack(fill="x")
        ttk.Label(frame, foreground="gray", wraplength=440, justify="left",
                  text="Your library as it is now is kept as a copy too, so this can be "
                       "reversed.").pack(anchor="w", pady=(6, 0))
        buttons = ttk.Frame(frame)
        buttons.pack(anchor="e", pady=(10, 0))
        ttk.Button(buttons, text="Cancel", command=win.destroy).pack(side="left")

        def restore():
            chosen = backups[box.curselection()[0]] if box.curselection() else backups[0]
            if rbdb.rekordbox_running():
                messagebox.showwarning(APP_NAME, "Please close rekordbox first.", parent=win)
                return
            try:
                keep = rbdb.restore_backup(chosen, db_path)
            except OSError as exc:
                messagebox.showerror(APP_NAME, f"Couldn't restore the backup:\n{exc}", parent=win)
                return
            win.destroy()
            self.log(f"Restored your library from {os.path.basename(chosen)}. "
                     f"The version from before the undo is saved as {os.path.basename(keep)}.")
            messagebox.showinfo(APP_NAME, "Done. Your library is back to how it was. "
                                          "Open rekordbox to check.")

        ttk.Button(buttons, text="Restore", command=restore).pack(side="left", padx=(8, 0))

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
    app = App(root)
    # Fit on small or scaled screens: the log box shrinks first.
    root.update_idletasks()
    height = min(root.winfo_reqheight(), root.winfo_screenheight() - 100)
    root.geometry(f"{root.winfo_reqwidth()}x{height}+40+20")
    root.minsize(600, min(560, height))
    app.check_for_update()
    root.mainloop()


if __name__ == "__main__":
    main()
