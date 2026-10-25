"""Turn large existing reports into small Git-readable views without statistics."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

MIB = 1024 * 1024


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _constant(value):
    raise ValueError("non_finite_json_constant")


def _escape(value):
    return str(value).replace("~", "~0").replace("/", "~1")


def _valid_path(value):
    return isinstance(value, str) and bool(value) and not Path(value).is_absolute() and all(
        part not in {"", ".", ".."} for part in value.split("/"))


def render_files(records, *, direct_limit=2 * MIB, chunk_bytes=512 * 1024):
    """Return (relative-path -> bytes, source representation records).

    Small files retain exact source bytes. CSV and JSON views preserve parsed
    values, not lexical formatting; source_sha256 always refers to original bytes.
    Invalid JSON is retained as explicitly labelled byte chunks, never repaired.
    """
    if direct_limit < 1 or chunk_bytes < 256:
        raise ValueError("positive_direct_limit_and_at_least_256_chunk_bytes_required")
    files, representations, seen = {}, [], set()
    for record in records:
        source = record["path"]
        if not _valid_path(source) or source in seen:
            raise ValueError("invalid_or_duplicate_source_path")
        seen.add(source)
        data = record.get("data")
        if data is None:
            data = Path(record["local_path"]).read_bytes()
        if not isinstance(data, bytes):
            raise TypeError("source_data_must_be_bytes")
        sha = hashlib.sha256(data).hexdigest()
        if record.get("sha256") and record["sha256"] != sha:
            raise ValueError("frozen_source_sha256_changed")
        rep = {"source_path": source, "source_sha256": sha, "source_bytes": len(data),
               "statistics_recomputed": False, "warnings": [], "parts": []}

        def add(content, suffix, **metadata):
            path = "views/" + source + "/part-%05d" % len(rep["parts"]) + suffix
            files[path] = content
            rep["parts"].append({"path": path, "bytes": len(content),
                                 "sha256": hashlib.sha256(content).hexdigest(), **metadata})
            if len(content) > chunk_bytes:
                rep["warnings"].append({"reason": "single_unit_exceeds_chunk_target", "path": path,
                                        "bytes": len(content)})

        def raw(reason=None):
            rep["storage"] = "raw-byte-chunks"
            rep["lossless_scope"] = "Exact original bytes: concatenate parts in part order."
            if reason:
                rep["warnings"].append({"reason": reason})
            for offset in range(0, len(data), chunk_bytes):
                add(data[offset:offset + chunk_bytes], ".bin", byte_offset=offset)

        if len(data) <= direct_limit:
            path = "files/" + source
            files[path] = data
            rep.update(storage="original", lossless_scope="Exact original bytes.")
            rep["parts"].append({"path": path, "bytes": len(data), "sha256": sha})
        elif Path(source).suffix.lower() == ".csv":
            try:
                text = data.decode("utf-8-sig")
            except UnicodeDecodeError:
                raw("csv_not_utf8_retained_as_original_bytes")
            else:
                rep["storage"] = "csv-row-parts"
                rep["lossless_scope"] = "Parsed CSV field strings and row order; formatting and BOM may differ. Each part repeats the original parsed header."
                old_limit = csv.field_size_limit()
                csv.field_size_limit(max(len(data), old_limit))
                try:
                    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
                    header = next(reader, [])
                    def encode_row(row):
                        stream = io.StringIO(newline="")
                        csv.writer(stream, lineterminator="\n").writerow(row)
                        return stream.getvalue().encode("utf-8")
                    prefix = encode_row(header)
                    pending, size, start, count = [], len(prefix), 0, 0
                    for row in reader:
                        line = encode_row(row)
                        if pending and size + len(line) > chunk_bytes:
                            add(prefix + b"".join(pending), ".csv", data_row_offset=start, data_rows=count,
                                header_repeated=True)
                            start += count
                            pending, size, count = [], len(prefix), 0
                        pending.append(line)
                        size += len(line)
                        count += 1
                    if pending or not rep["parts"]:
                        add(prefix + b"".join(pending), ".csv", data_row_offset=start, data_rows=count,
                            header_repeated=True)
                    rep["data_rows"] = start + count
                except csv.Error:
                    for part in rep["parts"]:
                        files.pop(part["path"], None)
                    rep["parts"] = []
                    raw("invalid_csv_retained_as_original_bytes")
                finally:
                    csv.field_size_limit(old_limit)
        elif Path(source).suffix.lower() == ".json":
            try:
                value = json.loads(data, parse_constant=_constant)
                _json(value)  # Reject numeric overflow / unpaired surrogates, without coercion.
            except (ValueError, UnicodeError, RecursionError):
                raw("invalid_or_nonfinite_json_retained_as_original_bytes")
            else:
                rep["storage"] = "json-pointer-parts"
                rep["lossless_scope"] = "JSON parsed values, object keys and array order; formatting and numeric lexical spelling may differ. Apply container/value/array-items/string-parts records by RFC6901 pointer."

                def node(pointer, item, force=False):
                    encoded = _json({"kind": "value", "pointer": pointer, "value": item})
                    if not force and len(encoded) <= chunk_bytes:
                        add(encoded + b"\n", ".json", pointer=pointer, kind="value")
                    elif isinstance(item, dict):
                        add(_json({"kind": "container", "pointer": pointer, "container": "object"}) + b"\n",
                            ".json", pointer=pointer, kind="container")
                        for key, child in item.items():
                            node(pointer + "/" + _escape(key), child)
                    elif isinstance(item, list):
                        add(_json({"kind": "container", "pointer": pointer, "container": "array", "length": len(item)}) + b"\n",
                            ".json", pointer=pointer, kind="container")
                        pending, size, offset = [], 0, 0
                        def flush():
                            nonlocal pending, size, offset
                            if not pending:
                                return
                            content = _json({"kind": "array-items", "pointer": pointer,
                                             "array_offset": offset, "items": pending}) + b"\n"
                            add(content, ".json", pointer=pointer, kind="array-items", array_offset=offset,
                                array_items=len(pending))
                            offset += len(pending)
                            pending, size = [], 0
                        overhead = len(_json({"kind": "array-items", "pointer": pointer,
                                             "array_offset": len(item), "items": []})) + 2
                        for index, child in enumerate(item):
                            child_size = len(_json(child))
                            if child_size + overhead > chunk_bytes:
                                flush()
                                node(pointer + "/" + str(index), child)
                                offset = index + 1
                            else:
                                if pending and size + child_size + overhead + len(pending) > chunk_bytes:
                                    flush()
                                pending.append(child)
                                size += child_size
                        flush()
                    elif isinstance(item, str):
                        add(_json({"kind": "container", "pointer": pointer, "container": "string"}) + b"\n",
                            ".json", pointer=pointer, kind="container")
                        # Split by UTF-8 character positions; JSON escaping expands at most six bytes per character.
                        width = max(1, (chunk_bytes - len(_json({"kind":"string-part", "pointer":pointer,
                                                              "character_offset":len(item), "text":""})) - 2) // 6)
                        for offset in range(0, len(item), width):
                            content = _json({"kind": "string-part", "pointer": pointer,
                                             "character_offset": offset, "text": item[offset:offset + width]}) + b"\n"
                            add(content, ".json", pointer=pointer, kind="string-part", character_offset=offset)
                    else:
                        add(encoded + b"\n", ".json", pointer=pointer, kind="value")
                node("", value, force=True)
        else:
            raw()
        representations.append(rep)
    return files, representations
