"""Fake Binary Ninja project objects shared by the project test modules.

They model measured Binary Ninja behaviour: a `.bnpr` is a directory with
extensionless blobs under `data/<2 hex>/<34 hex>`, `create_file(b"")` writes
neither a blob nor its parent directory, and two files may share a path in the
project while ids stay unique.
"""
from __future__ import annotations

import uuid
from pathlib import Path

SQLITE_MAGIC = b"SQLite format 3\x00"


class FakeProjectException(Exception):
    pass


class FakeFolder:
    def __init__(self, name: str, parent: "FakeFolder | None" = None):
        self.id = str(uuid.uuid4())
        self.name = name
        self.parent = parent


class FakeProjectFile:
    def __init__(self, name: str, folder: FakeFolder | None, path_on_disk: str,
                 description: str = "", file_id: str | None = None):
        self.id = file_id or str(uuid.uuid4())
        self.name = name
        self.folder = folder
        self.path_on_disk = path_on_disk
        self.description = description

    @property
    def exists_on_disk(self) -> bool:
        return Path(self.path_on_disk).exists()

    def get_path_in_project(self) -> str:
        parts = [self.name]
        folder = self.folder
        while folder is not None:
            parts.append(folder.name)
            folder = folder.parent
        return "/" + "/".join(reversed(parts))

    def export(self, dest: str) -> bool:
        Path(dest).write_bytes(Path(self.path_on_disk).read_bytes())
        return True


class FakeProject:
    """A `.bnpr` rooted at *root*: blobs live at `data/<2 hex>/<34 hex>`."""

    def __init__(self, root: Path, name: str = "demo"):
        self.root = root
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = "fake project"
        self.files: list[FakeProjectFile] = []
        self.folders: list[FakeFolder] = []
        self.closed = False
        self.deleted: list[FakeProjectFile] = []

    def _blob_path(self) -> Path:
        digest = uuid.uuid4().hex + uuid.uuid4().hex[:2]
        return self.root / "data" / digest[:2] / digest[2:]

    def add_file(self, name: str, folder: FakeFolder | None = None, *,
                 data: bytes | None = b"\x7fELF", file_id: str | None = None,
                 description: str = "") -> FakeProjectFile:
        blob = self._blob_path()
        if data is not None:
            blob.parent.mkdir(parents=True, exist_ok=True)
            blob.write_bytes(data)
        pf = FakeProjectFile(name, folder, str(blob), description, file_id)
        self.files.append(pf)
        return pf

    def add_folder(self, name: str, parent: FakeFolder | None = None) -> FakeFolder:
        folder = FakeFolder(name, parent)
        self.folders.append(folder)
        return folder

    # -- the BN API surface project_ops uses --
    def get_files_in_folder(self, folder):
        return [f for f in self.files if f.folder is folder]

    def create_folder(self, parent, name):
        return self.add_folder(name, parent)

    def create_file(self, contents: bytes, folder, name: str, description: str = ""):
        # A row exists without a blob or its parent directory.
        assert contents == b""
        return self.add_file(name, folder, data=None, description=description)

    def create_file_from_path(self, path: str, folder, name: str, description: str = ""):
        return self.add_file(name, folder, data=Path(path).read_bytes(), description=description)

    def delete_file(self, pf) -> bool:
        self.files.remove(pf)
        self.deleted.append(pf)
        return True

    def close(self) -> None:
        self.closed = True
