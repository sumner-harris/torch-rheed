"""Input parsing for standalone ``torch_rheed`` simulations."""

from __future__ import annotations

from pathlib import Path
import re

from .models import BulkInput, SurfaceInput


_TOKEN_RE = re.compile(r"[+-]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[dDeE][+-]?\d+)?")


class _TokenStream:
    """Sequential numeric token reader for the original text input format."""

    def __init__(self, path: Path) -> None:
        self._tokens: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            for segment in line.split(","):
                stripped = segment.strip()
                if not stripped:
                    continue
                tokens = _TOKEN_RE.findall(segment)
                remainder = _TOKEN_RE.sub("", segment).strip()
                if tokens and remainder == "":
                    self._tokens.extend(tokens)
                    continue
                break
        self._index = 0

    def read_int(self) -> int:
        token = self._tokens[self._index]
        self._index += 1
        return int(float(token.replace("D", "E").replace("d", "e")))

    def read_float(self) -> float:
        token = self._tokens[self._index]
        self._index += 1
        return float(token.replace("D", "E").replace("d", "e"))

    def remaining(self) -> int:
        return len(self._tokens) - self._index


def load_bulk_input(path: Path) -> BulkInput:
    """Parse a `bulk.txt` file into a structured `BulkInput` object."""

    stream = _TokenStream(path)
    nh = stream.read_int()
    nk = stream.read_int()
    ndom = stream.read_int()

    nb = [stream.read_int() for _ in range(ndom)]
    rdom_deg = [stream.read_float() for _ in range(ndom)]

    ih: list[int] = []
    ik: list[int] = []
    for count in nb:
        for _ in range(count):
            ih.append(stream.read_int())
            ik.append(stream.read_int())

    be = stream.read_float()
    azi_deg = stream.read_float()
    azf_deg = stream.read_float()
    daz_deg = stream.read_float()
    gi_deg = stream.read_float()
    gf_deg = stream.read_float()
    dg_deg = stream.read_float()

    dz_input = stream.read_float()
    ml = stream.read_int()

    nelm = stream.read_int()
    iz: list[int] = []
    da1: list[float] = []
    sap: list[float] = []
    bh: list[float] = []
    bk: list[float] = []
    bz: list[float] = []
    for _ in range(nelm):
        iz.append(stream.read_int())
        da1.append(stream.read_float())
        sap.append(stream.read_float())
        bh.append(stream.read_float())
        bk.append(stream.read_float())
        bz.append(stream.read_float())

    nsg = stream.read_int()
    aa = stream.read_float()
    bb = stream.read_float()
    gam_deg = stream.read_float()
    cc = stream.read_float()
    dx = stream.read_float()
    dy = stream.read_float()

    natm = stream.read_int()
    ielm: list[int] = []
    ocr: list[float] = []
    x: list[float] = []
    y: list[float] = []
    z: list[float] = []
    for _ in range(natm):
        ielm.append(stream.read_int())
        ocr.append(stream.read_float())
        x.append(stream.read_float())
        y.append(stream.read_float())
        z.append(stream.read_float())

    return BulkInput(
        nh=nh,
        nk=nk,
        ndom=ndom,
        nb=nb,
        rdom_deg=rdom_deg,
        ih=ih,
        ik=ik,
        be=be,
        azi_deg=azi_deg,
        azf_deg=azf_deg,
        daz_deg=daz_deg,
        gi_deg=gi_deg,
        gf_deg=gf_deg,
        dg_deg=dg_deg,
        dz_input=dz_input,
        ml=ml,
        nelm=nelm,
        iz=iz,
        da1=da1,
        sap=sap,
        bh=bh,
        bk=bk,
        bz=bz,
        nsg=nsg,
        aa=aa,
        bb=bb,
        gam_deg=gam_deg,
        cc=cc,
        dx=dx,
        dy=dy,
        natm=natm,
        ielm=ielm,
        ocr=ocr,
        x=x,
        y=y,
        z=z,
        source_path=path,
    )


def load_surface_input(path: Path, ndom: int) -> SurfaceInput:
    """Parse a `surf.txt` file into a structured `SurfaceInput` object."""

    stream = _TokenStream(path)
    nelms = stream.read_int()

    iz: list[int] = []
    da1: list[float] = []
    sap: list[float] = []
    bh: list[float] = []
    bk: list[float] = []
    bz: list[float] = []
    for _ in range(nelms):
        iz.append(stream.read_int())
        da1.append(stream.read_float())
        sap.append(stream.read_float())
        bh.append(stream.read_float())
        bk.append(stream.read_float())
        bz.append(stream.read_float())

    nsgs = stream.read_int()
    msa = stream.read_int()
    msb = stream.read_int()
    nsa = stream.read_int()
    nsb = stream.read_int()
    dthick = stream.read_float()
    dxs = stream.read_float()
    dys = stream.read_float()

    natms = stream.read_int()
    ielm: list[int] = []
    ocr: list[float] = []
    x: list[float] = []
    y: list[float] = []
    z: list[float] = []
    for _ in range(natms):
        ielm.append(stream.read_int())
        ocr.append(stream.read_float())
        x.append(stream.read_float())
        y.append(stream.read_float())
        z.append(stream.read_float())

    wdom = [stream.read_float() for _ in range(ndom)] if stream.remaining() > 0 else [1.0] * ndom

    return SurfaceInput(
        nelms=nelms,
        iz=iz,
        da1=da1,
        sap=sap,
        bh=bh,
        bk=bk,
        bz=bz,
        nsgs=nsgs,
        msa=msa,
        msb=msb,
        nsa=nsa,
        nsb=nsb,
        dthick=dthick,
        dxs=dxs,
        dys=dys,
        natms=natms,
        ielm=ielm,
        ocr=ocr,
        x=x,
        y=y,
        z=z,
        wdom=wdom,
        source_path=path,
    )
