"""Locate the foma binaries. The FST tier compiles its grammar with `foma` and
segments with `flookup`; both come with foma (https://fomafst.github.io/):

    apt install foma-bin        # Debian / Ubuntu
    brew install foma           # macOS

or build from source (https://github.com/mhulden/foma). Set PAPAYA_FOMA_BIN to
the directory holding the two binaries if they are not on PATH. The regex tier
(papaya-regex-16k) needs neither.
"""
import os
import shutil


class FomaNotFound(RuntimeError):
    pass


def _find(name):
    d = os.environ.get("PAPAYA_FOMA_BIN")
    for cand in ([os.path.join(d, name)] if d else []) + [shutil.which(name),
                                                        os.path.expanduser(f"~/.local/bin/{name}")]:
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    raise FomaNotFound(f"`{name}` not found. The FST tier needs foma: `apt install foma-bin` "
                       f"or `brew install foma`, or set PAPAYA_FOMA_BIN. The regex-only tokenizer "
                       f"(papaya-regex-16k) works without it.")


def foma_bin():
    return _find("foma")


def flookup_bin():
    return _find("flookup")


def available():
    try:
        foma_bin(), flookup_bin()
        return True
    except FomaNotFound:
        return False
