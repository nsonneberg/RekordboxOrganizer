# Third-party software in the Rekordbox Organizer builds

The app itself is MIT-licensed (see LICENSE). The built apps (PyInstaller) also
contain the components below, each under its own licence. Licences were read
from each package's own metadata and licence files as installed for the build,
and the list was compared with the packaged files of the Windows build
(pyi-archive_viewer); check both again when dependency versions change. The
macOS build has not been compared file by file.

| Component | Licence | Notes |
|---|---|---|
| Python runtime and standard library | PSF License | includes Tcl/Tk (BSD-style), OpenSSL 3 (Apache-2.0), SQLite (public domain), zlib, libffi (MIT) |
| pyrekordbox | MIT | reads and writes the rekordbox database |
| sqlcipher3-wheels | zlib/libpng | includes SQLCipher (BSD-style licence by Zetetic) |
| SQLAlchemy | MIT | |
| numpy | BSD-3-Clause (and 0BSD, MIT, Zlib, CC0-1.0 parts) | bundles OpenBLAS (BSD-3-Clause) |
| soundfile | BSD-3-Clause | |
| libsndfile (bundled by soundfile) | **LGPL-2.1** | source: https://github.com/libsndfile/libsndfile. You may replace it: the app's full source is public, so it can be rebuilt with a modified libsndfile (see README, Running from source) |
| onnxruntime | MIT | runs the optional genre model |
| keyring (and jaraco.*, more-itertools) | MIT | |
| pywin32-ctypes | BSD-3-Clause | used by keyring on Windows |
| certifi | MPL-2.0 | unmodified; source: https://github.com/certifi/python-certifi |
| psutil | BSD-3-Clause | |
| construct, six | MIT | dependencies of pyrekordbox |
| bidict | MPL-2.0 | unmodified; source: https://github.com/jab/bidict |
| python-dateutil | BSD-3-Clause / Apache-2.0 (dual) | |
| packaging | Apache-2.0 OR BSD-2-Clause | |
| cffi | MIT-0 | loads libsndfile for soundfile |
| pycparser | BSD-3-Clause | dependency of cffi |
| typing_extensions | PSF-2.0 | |
| setuptools | MIT | pulled in by cffi's optional build helper, not used by the app. Its bundled vendored parts are importlib_metadata (Apache-2.0), packaging (Apache-2.0 OR BSD-2-Clause), backports.tarfile, jaraco.*, more_itertools, tomli, wheel, zipp (MIT). Its LGPL-3 vendored `autocommand` is **not** in the build (checked with pyi-archive_viewer) |
| Microsoft Visual C++ runtime (vcruntime140, msvcp140) | Microsoft Visual C++ Redistributable licence | copied in by PyInstaller with Python/onnxruntime; redistribution terms **not verified** here |

**Not included:** PyAV and its FFmpeg libraries are deliberately left out of the
builds, because the PyAV wheels bundle the GPL-licensed x264 and x265 encoders.
When running from source, PyAV is optional (it adds M4A/AAC/ALAC decoding).

**Downloaded at run time, not bundled:** the genre models. YAMNet (Google,
Apache-2.0; ONNX conversion by audiomagic/yamnet-onnx, Apache-2.0; AudioSet
labels CC BY 4.0). Discogs-EffNet (Music Technology Group, Universitat Pompeu
Fabra): CC BY-NC-SA 4.0, non-commercial use only; a proprietary licence is
available from the MTG.
