#!/usr/bin/env python3
"""Report expert-tensor quant types of a remote GGUF without downloading it.

GGUF puts the header, metadata KVs and the full tensor-info table at the front of
the file, so an HTTP range request over the first few tens of MiB is enough. Used
to decide whether a quant's routed experts are i-quants (slow on the CPU path) or
K-quants, which on this hardware matters more than file size.

Usage: gguf-remote-types.py <repo> <file> [<file> ...]
"""

import struct
import subprocess
import sys

# ggml type id -> name, mirroring gguf-py/gguf/constants.py
TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
    20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 29: "IQ1_M",
    30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4",
    41: "Q1_0", 42: "Q2_0",
}


class Reader:
    def __init__(self, buf):
        self.b = buf
        self.p = 0

    def take(self, n):
        if self.p + n > len(self.b):
            raise EOFError("header truncated - fetch more bytes")
        v = self.b[self.p:self.p + n]
        self.p += n
        return v

    def u32(self): return struct.unpack("<I", self.take(4))[0]
    def u64(self): return struct.unpack("<Q", self.take(8))[0]
    def i32(self): return struct.unpack("<i", self.take(4))[0]
    def string(self): return self.take(self.u64()).decode("utf-8", "replace")

    def value(self, t):
        # scalar widths by gguf value type
        fixed = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if t == 8:
            self.string()
        elif t == 9:
            et = self.u32()
            n = self.u64()
            if et == 8:
                for _ in range(n):
                    self.string()
            elif et == 9:
                for _ in range(n):
                    self.value(9)
            else:
                self.take(fixed[et] * n)
        else:
            self.take(fixed[t])


def fetch(repo, name, nbytes):
    url = f"https://huggingface.co/{repo}/resolve/main/{name}"
    out = subprocess.run(
        ["curl", "-sL", "-H", f"Range: bytes=0-{nbytes - 1}", url],
        capture_output=True)
    return out.stdout


def parse(buf):
    r = Reader(buf)
    if r.take(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    r.u32()                       # version
    n_tensors = r.u64()
    n_kv = r.u64()
    for _ in range(n_kv):
        r.string()
        r.value(r.u32())
    hist, exps = {}, {}
    for _ in range(n_tensors):
        nm = r.string()
        dims = r.u32()
        for _ in range(dims):
            r.u64()
        t = TYPES.get(r.u32(), "?")
        r.u64()               # offset
        hist[t] = hist.get(t, 0) + 1
        if "_exps" in nm:
            exps[t] = exps.get(t, 0) + 1
    return n_tensors, hist, exps


def main():
    repo, files = sys.argv[1], sys.argv[2:]
    for name in files:
        buf = None
        for mb in (24, 64, 160):
            buf = fetch(repo, name, mb * 1024 * 1024)
            try:
                n, hist, exps = parse(buf)
                break
            except EOFError:
                continue
        else:
            print(f"{name}: header did not fit in 160 MiB")
            continue
        iq = sorted(t for t in exps if t.startswith("IQ"))
        print(f"\n{name}  ({n} tensors)")
        print("  expert tensors: " + ", ".join(
            f"{t} x{c}" for t, c in sorted(exps.items(), key=lambda kv: -kv[1])))
        print("  i-quant experts: " + (", ".join(iq) if iq else "NONE"))
        print("  all tensors: " + ", ".join(
            f"{t} x{c}" for t, c in sorted(hist.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
