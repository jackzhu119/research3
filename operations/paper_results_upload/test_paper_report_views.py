import csv
import hashlib
import io
import json
import unittest

from paper_report_views import render_files


def record(path, data):
    return {"path": path, "data": data, "sha256": hashlib.sha256(data).hexdigest()}


def restore_json(files, representation):
    root = [None]
    def parts(pointer):
        return [x.replace("~1", "/").replace("~0", "~") for x in pointer[1:].split("/")] if pointer else []
    def get(pointer):
        result = root[0]
        for part in parts(pointer):
            result = result[int(part)] if isinstance(result, list) else result[part]
        return result
    def put(pointer, value):
        keys = parts(pointer)
        if not keys:
            root[0] = value
            return
        parent = root[0]
        for part in keys[:-1]:
            parent = parent[int(part)] if isinstance(parent, list) else parent[part]
        key = int(keys[-1]) if isinstance(parent, list) else keys[-1]
        parent[key] = value
    for part in representation["parts"]:
        value = json.loads(files[part["path"]])
        pointer = value["pointer"]
        if value["kind"] == "container":
            container = value["container"]
            put(pointer, {} if container == "object" else ([None] * value["length"] if container == "array" else ""))
        elif value["kind"] == "value":
            put(pointer, value["value"])
        elif value["kind"] == "array-items":
            target = get(pointer)
            start = value["array_offset"]
            target[start:start + len(value["items"])] = value["items"]
        elif value["kind"] == "string-part":
            target = get(pointer)
            assert len(target) == value["character_offset"]
            put(pointer, target + value["text"])
    return root[0]


class ViewTests(unittest.TestCase):
    def test_small_files_retain_original_bytes(self):
        data = b'{ "x":  1, "nothing":null }\n'
        files, reps = render_files([record("reports/report.json", data)])
        self.assertEqual(files, {"files/reports/report.json": data})
        self.assertEqual(reps[0]["storage"], "original")

    def test_csv_chunks_preserve_multiline_unicode_and_row_order(self):
        rows = [["id", "note"], *[[str(i), 'line1\nline2,"中文"' * 4] for i in range(90)]]
        stream = io.StringIO(newline="")
        csv.writer(stream).writerows(rows)
        files, reps = render_files([record("pairs.csv", stream.getvalue().encode())], direct_limit=10, chunk_bytes=512)
        rebuilt = []
        for part in reps[0]["parts"]:
            self.assertLessEqual(part["bytes"], 512)
            parsed = list(csv.reader(io.StringIO(files[part["path"]].decode(), newline="")))
            self.assertEqual(parsed[0], rows[0])
            rebuilt.extend(parsed[1:])
        self.assertEqual(rebuilt, rows[1:])

    def test_csv_single_large_row_is_explicitly_warned(self):
        data = b"id,value\n1," + b"x" * 2000 + b"\n2,ok\n"
        files, reps = render_files([record("large.csv", data)], direct_limit=10, chunk_bytes=512)
        self.assertIn("single_unit_exceeds_chunk_target", [w["reason"] for w in reps[0]["warnings"]])
        self.assertEqual(reps[0]["data_rows"], 2)

    def test_json_nested_arrays_pointers_and_large_strings_reconstruct(self):
        value = {"a/b~c": [{"p": i, "text": "中文" * 12} for i in range(50)],
                 "nested": [{"large": "a\n\"中文" * 800}, [], {}], "scalar": 1.25}
        data = json.dumps(value, ensure_ascii=False).encode()
        files, reps = render_files([record("reports/report.json", data)], direct_limit=100, chunk_bytes=512)
        self.assertGreater(len(files), 5)
        self.assertTrue(all(len(data) <= 512 for data in files.values()))
        self.assertEqual(restore_json(files, reps[0]), value)
        self.assertEqual(reps[0]["source_sha256"], hashlib.sha256(data).hexdigest())

    def test_large_report_above_default_direct_limit_reconstructs(self):
        value = {"rows": [{"person": i, "text": "x" * 300} for i in range(7500)]}
        data = json.dumps(value).encode()
        self.assertGreater(len(data), 2 * 1024 * 1024)
        files, reps = render_files([record("reports/report.json", data)])
        self.assertEqual(reps[0]["storage"], "json-pointer-parts")
        self.assertTrue(all(len(data) <= 512 * 1024 for data in files.values()))
        self.assertEqual(restore_json(files, reps[0]), value)

    def test_bad_nan_json_is_preserved_as_exact_raw_bytes(self):
        data = b'{"number":NaN,"padding":"' + b"x" * 1000 + b'"}'
        files, reps = render_files([record("bad.json", data)], direct_limit=10, chunk_bytes=512)
        self.assertEqual(reps[0]["storage"], "raw-byte-chunks")
        self.assertEqual(b"".join(files[p["path"]] for p in reps[0]["parts"]), data)
        self.assertIn("invalid_or_nonfinite_json_retained_as_original_bytes", [w["reason"] for w in reps[0]["warnings"]])

    def test_binary_chunks_have_exact_offsets_and_original_digest(self):
        data = bytes(range(256)) * 20
        files, reps = render_files([record("other.bin", data)], direct_limit=10, chunk_bytes=512)
        self.assertEqual([p["byte_offset"] for p in reps[0]["parts"]], list(range(0, len(data), 512)))
        self.assertEqual(b"".join(files[p["path"]] for p in reps[0]["parts"]), data)

    def test_changed_frozen_bytes_are_rejected(self):
        row = record("report.json", b"{}")
        row["data"] = b"[]"
        with self.assertRaisesRegex(ValueError, "sha256_changed"):
            render_files([row])


if __name__ == "__main__":
    unittest.main()
