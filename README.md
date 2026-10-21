# Rekordbox Organizer

A small app for Windows and macOS. Tick the boxes for how you want your rekordbox
library organized, press **Run**, and you get a set of playlists back in rekordbox.

Your music files are never moved, renamed or edited. The app reads a rekordbox XML
export and writes a new XML file next to it (`<name>_organized.xml`).

## What it can make

**Organize by:** Genre · BPM range (step size you choose) · Key (Camelot, in wheel order) ·
Artist · Record label · Release year · Star rating · Track color · Month added

**Cleanup playlists:** Possible duplicates · Missing files · Missing genre / key / BPM · Never played

Everything goes into one playlist folder (named "Organized" by default). If you run
the app again, it replaces that folder.

## How to use

1. In rekordbox: **File → Export Collection in xml format**, and save the file.
2. Open Rekordbox Organizer, click **Browse...**, and pick that file.
3. Tick the options you want and press **Run**.
4. Back in rekordbox:
   - **Preferences → Advanced → Database → rekordbox xml**: set *Imported Library* to the new `_organized.xml` file.
   - **Preferences → View → Layout**: turn on **rekordbox xml**.
   - In the sidebar, open **rekordbox xml**, right-click the **Organized** folder, and choose **Import Playlist**.

## Running from source

You need Python 3.9 or newer. On Windows, the python.org installer includes Tkinter.
On macOS, use the python.org installer or `brew install python-tk`.

```
python app.py               # launch the app
python -m unittest -v       # run the tests
```

## Building the double-click apps

```
pip install pyinstaller
# Windows -> dist/Rekordbox Organizer.exe
pyinstaller --onefile --windowed --name "Rekordbox Organizer" app.py
# macOS   -> dist/Rekordbox Organizer.app
pyinstaller --windowed --name "Rekordbox Organizer" app.py
```

PyInstaller can only build for the OS it runs on. To get both builds from one place,
push this repo to GitHub: `.github/workflows/build.yml` builds both on every `v*` tag,
or you can start it by hand from the Actions tab.

The macOS build isn't signed. The first time you open it, right-click the app and
choose **Open**.
