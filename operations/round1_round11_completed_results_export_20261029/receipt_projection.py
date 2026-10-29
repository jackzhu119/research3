"""Stream small R1 planning fields without retaining scientific receipt payloads.

Only a management process installs this reader.  Scientific workers, receipt
writers and validation kernels retain their original implementations.  Every
byte is still parsed: a malformed or truncated skipped payload cannot become a
successful receipt.  Cache lifetime remains the original plan_ready context.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

FIELDS = frozenset(('status', 'arm', 'schema_version', 'configuration_index',
                    'configuration', 'dimension', 'radius'))
CHUNK_CHARS = 64 * 1024
MAX_METADATA_COST = 256 * 1024
MAX_DEPTH = 512
_SPECIAL = re.compile(r'["\\\x00-\x1f]')
_DIGITS = re.compile(r'[0-9]+')
_HEX = frozenset('0123456789abcdefABCDEF')
_TOO_LONG = object()
_ABSENT = object()


class ProjectionError(ValueError):
    """Invalid JSON or unreasonably large planning metadata; never completion."""


class _Stream:
    def __init__(self, handle):
        self.handle = handle
        self.buffer = ''
        self.position = 0
        self.metadata_cost = 0

    def error(self, message):
        raise ProjectionError(message)

    def charge(self, count):
        self.metadata_cost += count
        if self.metadata_cost > MAX_METADATA_COST:
            self.error('R1 planning metadata exceeds bounded projection size')

    def peek(self):
        if self.position == len(self.buffer):
            self.buffer = self.handle.read(CHUNK_CHARS)
            self.position = 0
        return self.buffer[self.position:self.position + 1]

    def take(self):
        char = self.peek()
        if char:
            self.position += 1
        return char

    def expect(self, char):
        if self.take() != char:
            self.error('Expected JSON punctuation or literal: ' + repr(char))

    def whitespace(self):
        while self.peek() in (' ', '\t', '\n', '\r'):
            self.position += 1

    def string(self, keep, *, key=False):
        self.expect('"')
        parts = [] if keep else None
        size = 0

        def retain(part):
            nonlocal parts, size
            if parts is None:
                return
            size += len(part)
            if key and size > 256:
                parts = None
                return
            if not key:
                self.charge(len(part) * 4)
            parts.append(part)

        while True:
            if not self.peek():
                self.error('Unterminated JSON string')
            match = _SPECIAL.search(self.buffer, self.position)
            if match is None:
                retain(self.buffer[self.position:])
                self.position = len(self.buffer)
                continue
            retain(self.buffer[self.position:match.start()])
            char = match.group()
            self.position = match.end()
            if char == '"':
                if not keep:
                    return None
                if parts is None:
                    return _TOO_LONG
                # This bounded scalar decoder preserves escaped Unicode,
                # including Python json's handling of lone surrogates.
                return json.loads('"' + ''.join(parts) + '"')
            if char != '\\':
                self.error('Unescaped control character in JSON string')
            escaped = self.take()
            if escaped == 'u':
                digits = ''.join(self.take() for _ in range(4))
                if len(digits) != 4 or any(c not in _HEX for c in digits):
                    self.error('Invalid JSON Unicode escape')
                retain('\\u' + digits)
            elif escaped and escaped in '"\\/bfnrt':
                retain('\\' + escaped)
            else:
                self.error('Invalid JSON string escape')

    def digits(self, keep):
        parts = [] if keep else None
        count = 0
        while self.peek():
            match = _DIGITS.match(self.buffer, self.position)
            if match is None:
                break
            chunk = match.group()
            count += len(chunk)
            self.position = match.end()
            if keep:
                self.charge(len(chunk) * 4)
                parts.append(chunk)
        return count, ''.join(parts) if keep else ''

    def number(self, keep):
        parts = []
        if self.peek() == '-':
            self.position += 1
            if keep:
                parts.append('-')
        if self.peek() == '0':
            self.position += 1
            if keep:
                parts.append('0')
            if self.peek() and self.peek() in '0123456789':
                self.error('Leading zero in JSON number')
        else:
            count, digits = self.digits(keep)
            if not count:
                self.error('Missing integer part in JSON number')
            if keep:
                parts.append(digits)
        if self.peek() == '.':
            self.position += 1
            count, digits = self.digits(keep)
            if not count:
                self.error('Missing fractional digits in JSON number')
            if keep:
                parts.extend(('.', digits))
        if self.peek() in ('e', 'E'):
            exponent = self.take()
            sign = ''
            if self.peek() in ('+', '-'):
                sign = self.take()
            count, digits = self.digits(keep)
            if not count:
                self.error('Missing exponent digits in JSON number')
            if keep:
                parts.extend((exponent, sign, digits))
        return json.loads(''.join(parts)) if keep else None

    def value(self, keep, depth=0):
        if depth > MAX_DEPTH:
            self.error('JSON nesting exceeds bounded projection depth')
        self.whitespace()
        char = self.peek()
        if char == '"':
            return self.string(keep)
        if char == '{':
            self.position += 1
            result = {} if keep else None
            self.whitespace()
            if self.peek() == '}':
                self.position += 1
                return result
            while True:
                self.whitespace()
                if self.peek() != '"':
                    self.error('JSON object key must be a string')
                name = self.string(keep)
                self.whitespace()
                self.expect(':')
                if keep:
                    self.charge(96)
                child = self.value(keep, depth + 1)
                if keep:
                    result[name] = child
                self.whitespace()
                char = self.take()
                if char == '}':
                    return result
                if char != ',':
                    self.error('Expected comma or closing brace in JSON object')
        if char == '[':
            self.position += 1
            result = [] if keep else None
            self.whitespace()
            if self.peek() == ']':
                self.position += 1
                return result
            while True:
                if keep:
                    self.charge(64)
                child = self.value(keep, depth + 1)
                if keep:
                    result.append(child)
                self.whitespace()
                char = self.take()
                if char == ']':
                    return result
                if char != ',':
                    self.error('Expected comma or closing bracket in JSON array')
        for literal, value in (('true', True), ('false', False), ('null', None)):
            if char == literal[0]:
                for expected in literal:
                    self.expect(expected)
                return value if keep else None
        if char and char in '-0123456789':
            if keep:
                self.charge(64)
            return self.number(keep)
        self.error('Invalid or truncated JSON value')

    def document(self):
        self.whitespace()
        if self.peek() != '{':
            # Preserve scalar/nonobject semantics for native isinstance checks.
            value = self.value(True)
            self.whitespace()
            if self.peek():
                self.error('Trailing data after JSON value')
            return value
        self.position += 1
        result = {}
        self.whitespace()
        if self.peek() == '}':
            self.position += 1
        else:
            while True:
                self.whitespace()
                if self.peek() != '"':
                    self.error('JSON object key must be a string')
                name = self.string(True, key=True)
                self.whitespace()
                self.expect(':')
                keep = name in FIELDS
                value = self.value(keep, 1)
                if keep:
                    result[name] = value
                self.whitespace()
                char = self.take()
                if char == '}':
                    break
                if char != ',':
                    self.error('Expected comma or closing brace in JSON object')
        self.whitespace()
        if self.peek():
            self.error('Trailing data after JSON object')
        return result


def read_projected(path, default=None):
    """Read a receipt completely, retain only bounded top-level planning fields."""
    try:
        with Path(path).open('r', encoding='utf-8', errors='strict', newline='') as handle:
            return _Stream(handle).document()
    except FileNotFoundError:
        return default


def install(bridge):
    """Patch only this management module, retaining its action-local read cache."""
    previous = bridge._read
    missing = getattr(bridge, '_MISSING', _ABSENT)

    def projected(path, default=None):
        context = getattr(bridge, '_READ_CACHE', None)
        cache = context.get() if context is not None else None
        key = str(path)
        if cache is not None and key in cache:
            return default if cache[key] is missing else cache[key]
        value = read_projected(path, missing)
        if cache is not None:
            cache[key] = value
        return default if value is missing else value

    bridge._read = projected
    return previous
