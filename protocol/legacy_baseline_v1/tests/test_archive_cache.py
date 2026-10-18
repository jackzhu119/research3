import hashlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'tools' / 'fetch_holdout_archive.py'
SPEC = importlib.util.spec_from_file_location('fetch_holdout_archive', SCRIPT)
archive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archive)


class Response(io.BytesIO):
    def __init__(self, body, status=200, headers=None):
        super().__init__(body)
        self.status = status
        self.headers = headers or {}


class ArchiveIntegrityTests(unittest.TestCase):
    def spec(self, body):
        return {'id': 1, 'name': 'sample.bin', 'size': len(body),
                'computed_md5': hashlib.md5(body).hexdigest(),
                'download_url': 'https://example.invalid/sample.bin'}

    def test_resume_reconstructs_exact_verified_object(self):
        body = b'0123456789' * 100
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'sample.bin.part').write_bytes(body[:137])
            with patch.object(archive.urllib.request, 'urlopen', return_value=Response(
                    body[137:], 206, {'Content-Range': f'bytes 137-999/1000'})) as fetch:
                record = archive.fetch_one(self.spec(body), root)
            self.assertEqual(fetch.call_args.args[0].get_header('Range'), 'bytes=137-')
            self.assertEqual((root / 'sample.bin').read_bytes(), body)
            self.assertEqual(record['sha256'], hashlib.sha256(body).hexdigest())
            self.assertFalse((root / 'sample.bin.part').exists())

    def test_ignored_range_restarts_without_appending(self):
        body = b'exact published bytes'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'sample.bin.part').write_bytes(body[:5])
            with patch.object(archive.urllib.request, 'urlopen', return_value=Response(body)):
                archive.fetch_one(self.spec(body), root)
            self.assertEqual((root / 'sample.bin').read_bytes(), body)

    def test_corrupt_object_is_never_committed(self):
        body = b'correct'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(archive.urllib.request, 'urlopen', return_value=Response(b'corrupt')):
                with self.assertRaises(ValueError):
                    archive.fetch_one(self.spec(body), root)
            self.assertFalse((root / 'sample.bin').exists())
            self.assertTrue((root / 'sample.bin.part').exists())

    def test_existing_wrong_file_is_preserved_and_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'sample.bin'
            path.write_bytes(b'incorrect')
            with self.assertRaises(ValueError):
                archive.fetch_one(self.spec(b'correct'), root)
            self.assertEqual(path.read_bytes(), b'incorrect')


if __name__ == '__main__':
    unittest.main()
