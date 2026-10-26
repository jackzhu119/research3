"""Restoration verifies bytes and never replaces an existing local output."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from transport.restore_source import restore, ValidationError


class RestoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.blocks = [b'\xef\xbb\xbfid,note\r\n', '1,"中文\r\nline2"\r\n'.encode()]
        joined = b''.join(self.blocks)
        offset = 0
        parts = []
        for number, block in enumerate(self.blocks, 1):
            parts.append(dict(number=number, offset_bytes=offset, bytes=len(block),
                              sha256=hashlib.sha256(block).hexdigest(), path='results/part-%d.txt' % number))
            offset += len(block)
        self.manifest = dict(schema_version='paper3-lossless-source-v1', source_bytes=len(joined),
                             source_sha256=hashlib.sha256(joined).hexdigest(), parts=parts)

    def tearDown(self):
        self.temporary.cleanup()

    def test_exact_csv_and_manifest_locator_bytes_reconstruct_read_only(self):
        before = list(self.blocks)
        for schema in ('paper3-lossless-source-v1', 'paper3-lossless-json-locator-v1'):
            manifest = dict(self.manifest, schema_version=schema)
            target = self.base / schema
            receipt = restore(manifest, target, lambda part: self.blocks[part['number'] - 1])
            self.assertEqual(target.read_bytes(), b''.join(self.blocks))
            self.assertEqual(receipt['status'], 'RESTORED_EXACT_BYTES_SHA256_VERIFIED')
            self.assertEqual(receipt['sha256'], manifest['source_sha256'])
        self.assertEqual(self.blocks, before)

    def test_existing_wrong_output_and_symlink_are_never_replaced(self):
        target = self.base / 'important.csv'
        target.write_bytes(b'existing original bytes')
        with self.assertRaisesRegex(ValidationError, 'output_already_exists'):
            restore(self.manifest, target, lambda part: self.fail('must not download over an existing output'))
        self.assertEqual(target.read_bytes(), b'existing original bytes')
        linked = self.base / 'linked.csv'
        linked.symlink_to(target)
        with self.assertRaisesRegex(ValidationError, 'output_already_exists'):
            restore(self.manifest, linked, lambda part: self.fail('must not download through a symlink'))
        self.assertTrue(linked.is_symlink())

    def test_bad_part_cannot_publish_partial_output(self):
        target = self.base / 'new.csv'
        with self.assertRaisesRegex(ValidationError, 'part_SHA256_mismatch'):
            restore(self.manifest, target, lambda part: b'wrong bytes')
        self.assertFalse(target.exists())
        self.assertEqual(list(self.base.glob('*.restore-*')), [])


if __name__ == '__main__':
    unittest.main()
