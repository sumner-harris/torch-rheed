"""Standalone bulk and surface simulation engine for torch_rheed."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from .asf import asfparam
from .inputs import load_bulk_input, load_surface_input
from .models import (
    BulkDomain,
    BulkInput,
    BulkSimulation,
    RockingCurveBatchResult,
    RockingCurvePairBatchResult,
    RockingCurveResult,
    ScreenImageConfig,
    SurfaceInput,
)
from .screen import DEFAULT_SCREEN_IMAGE_CONFIG, render_screen_stack


torch.set_default_dtype(torch.float64)


def _resolve_device(device: str | torch.device | None) -> torch.device:
    """Normalize the requested execution device for solver entrypoints."""

    if device is None:
        return torch.device("cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is False on this machine")
    return resolved


def _scan_range(azi_deg: float, azf_deg: float, daz_deg: float, gi_deg: float, gf_deg: float, dg_deg: float) -> tuple[float, float, int, float, float, int]:
    """Convert the angular scan definition to radians and discrete counts."""

    rad = math.pi / 180.0
    if abs(daz_deg) < 1e-4:
        naz = 1
    else:
        naz = max(int((azf_deg - azi_deg) / daz_deg + 0.1) + 1, 1)

    if abs(dg_deg) < 1e-4:
        ng = 1
    else:
        ng = max(int((gf_deg - gi_deg) / dg_deg + 0.1) + 1, 1)

    return azi_deg * rad, daz_deg * rad, naz, gi_deg * rad, dg_deg * rad, ng


def _build_scan_vectors(
    *,
    naz: int,
    ng: int,
    azi_rad: float,
    daz_rad: float,
    gi_rad: float,
    dg_rad: float,
    rdom_rad: float,
    wn: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct the scan-dependent beam-projection vectors in loop order."""

    # Screen rendering currently requires ``naz == 1`` and expects the shared
    # scan axis to be the glancing-angle grid, even for a single-point run.
    if naz == 1 or ng > naz:
        az_values = torch.arange(naz, dtype=torch.float64, device=device) * daz_rad + (azi_rad + rdom_rad)
        ga_values = torch.arange(ng, dtype=torch.float64, device=device) * dg_rad + gi_rad
        az_grid = az_values[:, None].expand(naz, ng)
        ga_grid = ga_values[None, :].expand(naz, ng)
        angle_deg = ga_grid.reshape(-1) * (180.0 / math.pi)
    else:
        ga_values = torch.arange(ng, dtype=torch.float64, device=device) * dg_rad + gi_rad
        az_values = torch.arange(naz, dtype=torch.float64, device=device) * daz_rad + (azi_rad + rdom_rad)
        ga_grid = ga_values[:, None].expand(ng, naz)
        az_grid = az_values[None, :].expand(ng, naz)
        angle_deg = az_grid.reshape(-1) * (180.0 / math.pi)

    cga = torch.cos(ga_grid.reshape(-1))
    sga = torch.sin(ga_grid.reshape(-1))
    caz = torch.cos(az_grid.reshape(-1))
    saz = torch.sin(az_grid.reshape(-1))
    wnsga2 = (wn * sga * sga).contiguous()
    wnx = (wn * cga * caz).contiguous()
    wny = (wn * cga * saz).contiguous()
    return angle_deg.contiguous(), wnsga2, torch.stack((wnx, wny), dim=1)


def _complex_sqrt_from_sval(sval: torch.Tensor) -> torch.Tensor:
    """Return the physical square-root branch for propagating or evanescent beams."""

    real_part = torch.sqrt(torch.clamp(sval, min=0.0))
    imag_part = torch.sqrt(torch.clamp(-sval, min=0.0))
    return torch.complex(real_part, imag_part)


def _surface_parameter_signature(surface: SurfaceInput) -> tuple[object, ...]:
    """Return the batch-compatibility signature for one surface parameter block."""

    return (
        surface.nelms,
        tuple(surface.iz),
        tuple(surface.da1),
        tuple(surface.sap),
        tuple(surface.bh),
        tuple(surface.bk),
        tuple(surface.bz),
    )


def _bulk_input_signature(bulk: BulkInput) -> tuple[object, ...]:
    """Return a hashable signature for one parsed bulk input."""

    return (
        bulk.nh,
        bulk.nk,
        bulk.ndom,
        tuple(bulk.nb),
        tuple(bulk.rdom_deg),
        tuple(bulk.ih),
        tuple(bulk.ik),
        bulk.be,
        bulk.azi_deg,
        bulk.azf_deg,
        bulk.daz_deg,
        bulk.gi_deg,
        bulk.gf_deg,
        bulk.dg_deg,
        bulk.dz_input,
        bulk.ml,
        bulk.nelm,
        tuple(bulk.iz),
        tuple(bulk.da1),
        tuple(bulk.sap),
        tuple(bulk.bh),
        tuple(bulk.bk),
        tuple(bulk.bz),
        bulk.nsg,
        bulk.aa,
        bulk.bb,
        bulk.gam_deg,
        bulk.cc,
        bulk.dx,
        bulk.dy,
        bulk.natm,
        tuple(bulk.ielm),
        tuple(bulk.ocr),
        tuple(bulk.x),
        tuple(bulk.y),
        tuple(bulk.z),
    )


def _require_supported_geometry(nsg: int, label: str) -> None:
    if nsg not in (0, 1):
        raise NotImplementedError(
            f"torch_rheed currently supports only p1 symmetry for the {label} layer; got nsg={nsg}"
        )


def _reduce01(x: float) -> float:
    return x - math.floor(x + 1e-4)


def _strfac_p1(
    nv: int,
    ma: int,
    mb: int,
    na: int,
    nb: int,
    gh: torch.Tensor,
    gk: torch.Tensor,
    x: float,
    y: float,
    dx: float,
    dy: float,
) -> torch.Tensor:
    """Evaluate the p1 structure factor contribution for one atom."""

    s = float(ma * nb - mb * na)
    xs = _reduce01((nb * x - na * y) / s)
    ys = _reduce01((ma * y - mb * x) / s)
    xb = ma * xs + na * ys
    yb = mb * xs + nb * ys
    gr = gh[:nv] * (dx + xb) + gk[:nv] * (dy + yb)
    return torch.cos(gr).to(torch.complex128) + 1j * torch.sin(gr).to(torch.complex128)


def _prepare_atomic_factor_tables(
    iz: list[int],
    da1: list[float],
    bz: list[float],
    be: float,
    aa: float,
    bb: float,
    gam_rad: float,
    device: torch.device,
) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Precompute energy-corrected atomic-scattering-factor tables."""

    nelm = len(iz)
    a_all = torch.empty((4, nelm), dtype=torch.float64, device=device)
    b_all = torch.empty((4, nelm), dtype=torch.float64, device=device)
    c_all = torch.empty((4, nelm), dtype=torch.float64, device=device)

    c2m = 511.001
    ek = 0.262466
    wn = math.sqrt(1e3 * be * ek * (1.0 + 0.5 * be / c2m))
    rel = (1.0 + be / c2m) * 4.0 * math.pi / (aa * bb * math.sin(gam_rad))

    for i, iz_i in enumerate(iz):
        a, b = asfparam(iz_i, device=device)
        if iz_i > 0:
            a = a.clone()
            a[0] = a[0] - da1[i]
        else:
            a = a * da1[i]

        scale = torch.sqrt(torch.tensor(4.0 * math.pi, dtype=torch.float64, device=device) / (b + bz[i])) * rel
        a_all[:, i] = a * scale
        c_all[:, i] = b / (16.0 * math.pi * math.pi)
        b_all[:, i] = 4.0 * math.pi * math.pi / (b + bz[i])

    return wn, a_all, b_all, c_all


def _asfcef(
    igh: list[int],
    igk: list[int],
    bh: list[float],
    bk: list[float],
    ghx: float,
    ghy: float,
    gky: float,
    a_all: torch.Tensor,
    b_all: torch.Tensor,
    c_all: torch.Tensor,
) -> torch.Tensor:
    """Compute beam-dependent atomic-scattering-factor coefficients."""

    pi2 = 2.0 * math.pi
    uf = 70.0
    device = a_all.device

    igh_tensor = torch.tensor(igh, dtype=torch.float64, device=device).unsqueeze(1)
    igk_tensor = torch.tensor(igk, dtype=torch.float64, device=device).unsqueeze(1)
    bh_tensor = torch.tensor(bh, dtype=torch.float64, device=device).unsqueeze(0)
    bk_tensor = torch.tensor(bk, dtype=torch.float64, device=device).unsqueeze(0)

    xh = ghx * igh_tensor
    yh = ghy * igh_tensor
    yk = gky * igk_tensor
    s = xh.square() + (yh + yk).square()

    uh = torch.sqrt(bh_tensor * 0.5) / pi2
    uk = torch.sqrt(bk_tensor * 0.5) / pi2
    xi = uh * xh
    yi = uh * yh + uk * yk
    si = 0.5 * (xi.square() + yi.square())

    ex = c_all.transpose(0, 1).unsqueeze(0) * s.unsqueeze(-1) + si.unsqueeze(-1)
    asf = a_all.transpose(0, 1).unsqueeze(0) * torch.exp(-ex)
    asf = torch.where(ex > uf, torch.zeros_like(asf), asf)
    return asf.permute(0, 2, 1).contiguous()


def _scpot(
    nv_use: int,
    ns: int,
    ielm: list[int],
    ocr: list[float],
    x: list[float],
    y: list[float],
    z: list[float],
    zo: float,
    dz: float,
    sap: list[float],
    nsg: int,
    ma: int,
    mb: int,
    na: int,
    nb: int,
    gh: torch.Tensor,
    gk: torch.Tensor,
    dx: float,
    dy: float,
    asf: torch.Tensor,
    b_all: torch.Tensor,
    v: torch.Tensor,
    vi: torch.Tensor,
    *,
    negpos: int = -1,
    iclr: int = 1,
) -> None:
    """Accumulate the scattering potential for one layer stack."""

    _require_supported_geometry(nsg, "bulk" if ma == 1 and mb == 0 and na == 0 and nb == 1 else "surface")
    natm = len(ielm)
    if natm == 0 or ns == 0:
        return

    if iclr == 1:
        v[:, :] = 0.0j
        vi[:, :] = 0.0j

    uf = 70.0
    rmesh = 1.0 / abs(ma * nb - mb * na)
    device = gh.device
    atom_x = torch.tensor(x, dtype=torch.float64, device=device)
    atom_y = torch.tensor(y, dtype=torch.float64, device=device)
    atom_z = torch.tensor(z, dtype=torch.float64, device=device)
    atom_ielm = torch.tensor(ielm, dtype=torch.long, device=device) - 1
    atom_ocr = torch.tensor(ocr, dtype=torch.float64, device=device) * rmesh
    sap_tensor = torch.tensor(sap, dtype=torch.float64, device=device)

    s = float(ma * nb - mb * na)
    xs = (nb * atom_x - na * atom_y) / s
    xs = xs - torch.floor(xs + 1e-4)
    ys = (ma * atom_y - mb * atom_x) / s
    ys = ys - torch.floor(ys + 1e-4)
    xb = ma * xs + na * ys
    yb = mb * xs + nb * ys
    gr = gh[:nv_use].unsqueeze(1) * (dx + xb).unsqueeze(0) + gk[:nv_use].unsqueeze(1) * (dy + yb).unsqueeze(0)
    st = torch.complex(torch.cos(gr), torch.sin(gr))

    layer_z = dz * 0.5 + zo + dz * torch.arange(ns, dtype=torch.float64, device=device)
    z2 = (layer_z.unsqueeze(1) - atom_z.unsqueeze(0)).square()
    positive_sap = torch.clamp(sap_tensor[atom_ielm], min=0.0)
    abs_sap = torch.abs(sap_tensor[atom_ielm])
    vi_scale = positive_sap.unsqueeze(0).expand(nv_use, natm).clone()
    vi_scale[0, :] = abs_sap

    weighted_structure: list[torch.Tensor] = []
    for term in range(4):
        ex = b_all[term, atom_ielm].unsqueeze(0) * z2
        exv = torch.where(ex < uf, torch.exp(-ex) * atom_ocr.unsqueeze(0), torch.zeros_like(ex))
        weighted_st = torch.index_select(asf[:nv_use, term, :], 1, atom_ielm) * st
        contribution = (weighted_st.unsqueeze(1) * exv.unsqueeze(0)).sum(dim=2)
        weighted_structure.append(contribution)
        v[:nv_use, :] += contribution
        vi[:nv_use, :] += 1j * (weighted_st.mul(vi_scale).unsqueeze(1) * exv.unsqueeze(0)).sum(dim=2)

    if iclr == -1 and negpos > 0:
        v[:, :] = -v


def _blkibg(nb: int, nh: int, nk: int, ih: list[int], ik: list[int]) -> tuple[list[int], list[int]]:
    """Group bulk beams that interact through the same modulo class."""

    if nh == 1 and nk == 1:
        return list(range(nb)), [nb]

    jorg: list[int] = []
    nbg: list[int] = []
    done = [False] * nb
    init = 0
    while True:
        group = [init]
        done[init] = True
        ih0 = ih[init]
        ik0 = ik[init]
        next_init: int | None = None
        for i in range(init + 1, nb):
            if done[i]:
                continue
            if (ih[i] - ih0) % nh == 0 and (ik[i] - ik0) % nk == 0:
                done[i] = True
                group.append(i)
            elif next_init is None:
                next_init = i
        jorg.extend(group)
        nbg.append(len(group))
        if len(jorg) >= nb:
            break
        if next_init is None:
            raise RuntimeError("failed to choose next beam group in bulk interaction partition")
        init = next_init
    return jorg, nbg


def _blkghk(nb: int, jorg: list[int], nbg: list[int], ih: list[int], ik: list[int]) -> tuple[int, list[int], list[int], list[list[int]]]:
    """Build the bulk beam-difference lookup table."""

    nbgm = max(nbg)
    nvm = 1 + sum(count * (count - 1) // 2 for count in nbg)
    igh = [0]
    igk = [0]
    iv = [[0 for _ in range(nb)] for _ in range(nbgm)]
    ibas = 0
    for group_size in nbg:
        iend = ibas + group_size
        for k in range(ibas, iend - 1):
            for l in range(k + 1, iend):
                ih0 = ih[jorg[l]] - ih[jorg[k]]
                ik0 = ik[jorg[l]] - ik[jorg[k]]
                found = False
                for m in range(1, len(igh)):
                    if ih0 == igh[m] and ik0 == igk[m]:
                        iv[l - ibas][k] = m
                        found = True
                        break
                if not found:
                    if len(igh) >= nvm:
                        raise ValueError(f"bulk beam-difference table overflow: {len(igh)} >= {nvm}")
                    igh.append(ih0)
                    igk.append(ik0)
                    iv[l - ibas][k] = len(igh) - 1
        ibas = iend
    return len(igh), igh, igk, iv


def _srfghk(nb: int, ih: list[int], ik: list[int], igh_seed: list[int], igk_seed: list[int]) -> tuple[int, list[int], list[int], list[list[int]]]:
    """Build the surface beam-difference lookup table."""

    nvm = 1 + nb * (nb - 1) // 2
    igh = list(igh_seed)
    igk = list(igk_seed)
    iv = [[0 for _ in range(nb)] for _ in range(nb)]
    for k in range(nb - 1):
        for l in range(k + 1, nb):
            ih0 = ih[l] - ih[k]
            ik0 = ik[l] - ik[k]
            found = False
            for m in range(1, len(igh)):
                if ih0 == igh[m] and ik0 == igk[m]:
                    iv[l][k] = m
                    found = True
                    break
            if not found:
                if len(igh) >= nvm:
                    raise ValueError(f"surface beam-difference table overflow: {len(igh)} >= {nvm}")
                igh.append(ih0)
                igk.append(ik0)
                iv[l][k] = len(igh) - 1
    return len(igh), igh, igk, iv


def _right_solve(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute `a @ inv(b)` in a numerically stable way."""

    return torch.linalg.solve(b.transpose(-2, -1), a.transpose(-2, -1)).transpose(-2, -1).contiguous()


def _trmat(nb: int, gma: torch.Tensor, v: torch.Tensor, vi: torch.Tensor, iv: list[list[int]], dz: float) -> torch.Tensor:
    """Compute the transfer matrix for one slice."""

    batched = gma.ndim == 2
    if not batched:
        gma = gma.unsqueeze(0)
    device = gma.device

    batch = gma.shape[0]
    if v.ndim == 1:
        v_batch = v.unsqueeze(0).expand(batch, -1)
    else:
        v_batch = v
    if vi.ndim == 1:
        vi_batch = vi.unsqueeze(0).expand(batch, -1)
    else:
        vi_batch = vi

    if nb == 1:
        eigvals = (v_batch[:, 0] + vi_batch[:, 0] + gma[:, 0] * gma[:, 0]).unsqueeze(1)
        eigvecs = torch.ones((batch, 1, 1), dtype=torch.complex128, device=device)
    else:
        h = torch.zeros((batch, nb, nb), dtype=torch.complex128, device=device)
        base = (v_batch[:, 0] + vi_batch[:, 0]).unsqueeze(1)
        diag_idx = torch.arange(nb, device=device)
        h[:, diag_idx, diag_idx] = base + gma * gma
        for i in range(nb):
            for j in range(i + 1, nb):
                idx = iv[j][i]
                h[:, j, i] = v_batch[:, idx] + vi_batch[:, idx]
                h[:, i, j] = torch.conj(v_batch[:, idx] - vi_batch[:, idx])
        eigvals, eigvecs = torch.linalg.eig(h)

    x1 = gma.unsqueeze(2)
    x2 = torch.sqrt(eigvals).unsqueeze(1)
    x3 = eigvecs
    tau = (x1 + x2) * x3
    rou = (x1 - x2) * x3
    ph = torch.exp((1j * dz) * x2)
    ph2 = 1.0 / ph

    nb2 = nb + nb
    t1 = torch.zeros((batch, nb2, nb2), dtype=torch.complex128, device=device)
    t3 = torch.zeros((batch, nb2, nb2), dtype=torch.complex128, device=device)
    t1[:, :nb, :nb] = tau * ph
    t1[:, :nb, nb:] = rou * ph2
    t1[:, nb:, :nb] = rou * ph
    t1[:, nb:, nb:] = tau * ph2
    t3[:, :nb, :nb] = tau
    t3[:, :nb, nb:] = rou
    t3[:, nb:, :nb] = rou
    t3[:, nb:, nb:] = tau

    result = _right_solve(t1, t3)
    return result if batched else result[0]


def _simulate_bulk_domain(
    bulk_input: BulkInput,
    *,
    rdom_rad: float,
    ih: list[int],
    ik: list[int],
    dz: float,
    naz: int,
    ng: int,
    azi_rad: float,
    daz_rad: float,
    gi_rad: float,
    dg_rad: float,
    epsb: float,
    gam_rad: float,
    wn: float,
    a_all: torch.Tensor,
    b_all: torch.Tensor,
    c_all: torch.Tensor,
) -> BulkDomain:
    """Simulate the bulk scattering matrices for a single domain."""

    nb = len(ih)
    jorg, nbg = _blkibg(nb, bulk_input.nh, bulk_input.nk, ih, ik)
    nv, igh, igk, iv = _blkghk(nb, jorg, nbg, ih, ik)
    device = a_all.device

    ghx = (2.0 * math.pi) / (bulk_input.aa * bulk_input.nh)
    ghy = -(2.0 * math.pi) / (bulk_input.aa * math.tan(gam_rad) * bulk_input.nh)
    gky = (2.0 * math.pi) / (bulk_input.bb * math.sin(gam_rad) * bulk_input.nk)
    asf = _asfcef(igh, igk, bulk_input.bh, bulk_input.bk, ghx, ghy, gky, a_all, b_all, c_all)

    gh = torch.tensor([(-(2.0 * math.pi) / bulk_input.nh) * float(v) for v in igh], dtype=torch.float64, device=device)
    gk = torch.tensor([(-(2.0 * math.pi) / bulk_input.nk) * float(v) for v in igk], dtype=torch.float64, device=device)

    ns = int(bulk_input.cc / bulk_input.dz_input) + 1
    if ns < 2:
        raise ValueError(f"too few bulk slices after discretization: ns={ns}")
    v = torch.zeros((nv, ns), dtype=torch.complex128, device=device)
    vi = torch.zeros((nv, ns), dtype=torch.complex128, device=device)
    _scpot(
        nv_use=nv,
        ns=ns,
        ielm=bulk_input.ielm,
        ocr=bulk_input.ocr,
        x=bulk_input.x,
        y=bulk_input.y,
        z=bulk_input.z,
        zo=0.0,
        dz=dz,
        sap=bulk_input.sap,
        nsg=bulk_input.nsg,
        ma=1,
        mb=0,
        na=0,
        nb=1,
        gh=gh,
        gk=gk,
        dx=0.0,
        dy=0.0,
        asf=asf,
        b_all=b_all,
        v=v,
        vi=vi,
        iclr=1,
    )
    _scpot(
        nv_use=nv,
        ns=ns,
        ielm=bulk_input.ielm,
        ocr=bulk_input.ocr,
        x=bulk_input.x,
        y=bulk_input.y,
        z=bulk_input.z,
        zo=bulk_input.cc,
        dz=dz,
        sap=bulk_input.sap,
        nsg=bulk_input.nsg,
        ma=1,
        mb=0,
        na=0,
        nb=1,
        gh=gh,
        gk=gk,
        dx=-bulk_input.dx,
        dy=-bulk_input.dy,
        asf=asf,
        b_all=b_all,
        v=v,
        vi=vi,
        iclr=0,
    )
    _scpot(
        nv_use=nv,
        ns=ns,
        ielm=bulk_input.ielm,
        ocr=bulk_input.ocr,
        x=bulk_input.x,
        y=bulk_input.y,
        z=bulk_input.z,
        zo=-bulk_input.cc,
        dz=dz,
        sap=bulk_input.sap,
        nsg=bulk_input.nsg,
        ma=1,
        mb=0,
        na=0,
        nb=1,
        gh=gh,
        gk=gk,
        dx=bulk_input.dx,
        dy=bulk_input.dy,
        asf=asf,
        b_all=b_all,
        v=v,
        vi=vi,
        iclr=-1,
    )

    if abs(bulk_input.dx) > 1e-4 or abs(bulk_input.dy) > 1e-4:
        pih = bulk_input.dx * 2.0 * math.pi / bulk_input.nh
        pik = bulk_input.dy * 2.0 * math.pi / bulk_input.nk
        ph = [
            complex(math.cos(pih * ih_i + pik * ik_i), math.sin(pih * ih_i + pik * ik_i))
            for ih_i, ik_i in zip(ih, ik)
        ]
    else:
        ph = None

    nrep1 = naz if ng > naz else ng
    nrep2 = ng if ng > naz else naz
    angle_deg, _, wave_xy = _build_scan_vectors(
        naz=naz,
        ng=ng,
        azi_rad=azi_rad,
        daz_rad=daz_rad,
        gi_rad=gi_rad,
        dg_rad=dg_rad,
        rdom_rad=rdom_rad,
        wn=wn,
        device=device,
    )
    n_angles = angle_deg.numel()
    wnx = wave_xy[:, 0].unsqueeze(1)
    wny = wave_xy[:, 1].unsqueeze(1)
    ih_tensor = torch.tensor(ih, dtype=torch.float64, device=device)
    ik_tensor = torch.tensor(ik, dtype=torch.float64, device=device)
    ph_tensor = torch.tensor(ph, dtype=torch.complex128, device=device) if ph is not None else None

    group_blocks: list[torch.Tensor] = []
    ibas = 0
    for group_size in nbg:
        iend = ibas + group_size
        nbf = group_size
        indices = torch.tensor(jorg[ibas:iend], dtype=torch.long, device=device)
        ih_group = ih_tensor.index_select(0, indices).unsqueeze(0)
        ik_group = ik_tensor.index_select(0, indices).unsqueeze(0)
        sval = wn * wn - (ghx * ih_group + wnx).square() - (ghy * ih_group + gky * ik_group + wny).square()
        gma = _complex_sqrt_from_sval(sval)

        transfer_product: torch.Tensor | None = None
        for layer in range(ns):
            transfer = _trmat(nbf, gma, v[:, layer], vi[:, layer], iv, dz)
            transfer_product = transfer if transfer_product is None else transfer @ transfer_product

        if transfer_product is None:
            raise RuntimeError("bulk transfer product was not initialized")

        if ph_tensor is not None:
            factor = ph_tensor.index_select(0, indices)
            transfer_product[:, :, :nbf] = transfer_product[:, :, :nbf] * factor.view(1, 1, -1)
            transfer_product[:, :, nbf:] = transfer_product[:, :, nbf:] * factor.view(1, 1, -1)

        top_left = transfer_product[:, :nbf, :nbf]
        top_right = transfer_product[:, :nbf, nbf:]
        bottom_left = transfer_product[:, nbf:, :nbf]
        bottom_right = transfer_product[:, nbf:, nbf:]
        r20 = torch.zeros((n_angles, nbf), dtype=torch.float64, device=device)
        active = torch.ones((n_angles,), dtype=torch.bool, device=device)
        t4_block = torch.zeros((n_angles, nbf, nbf), dtype=torch.complex128, device=device)

        for layer_count in range(1, bulk_input.ml + 1):
            t1 = top_right if layer_count == 1 else top_right + top_left @ t4_block
            t2 = bottom_right if layer_count == 1 else bottom_right + bottom_left @ t4_block
            new_block = _right_solve(t1, t2)

            if layer_count < 20:
                t4_block = new_block
                continue

            diag_power = new_block.diagonal(dim1=-2, dim2=-1).abs().square()
            if layer_count == 20:
                t4_block = new_block
                r20 = diag_power
                continue

            rat = torch.abs(diag_power - r20).amax(dim=1)
            t4_block = torch.where(active.view(-1, 1, 1), new_block, t4_block)
            r20 = torch.where(active.view(-1, 1), diag_power, r20)
            active = active & (rat >= epsb)
            if not torch.any(active):
                break

        group_blocks.append(t4_block.clone())
        ibas = iend

    matrices: list[torch.Tensor] = []
    for angle_index in range(n_angles):
        for group_index in range(len(nbg)):
            matrices.append(group_blocks[group_index][angle_index].clone())

    return BulkDomain(
        nb=nb,
        rdom_rad=rdom_rad,
        ih=ih,
        ik=ik,
        jorg=jorg,
        ngr=len(nbg),
        nvb=nv,
        nv=nv,
        nbg=nbg,
        igh=igh,
        igk=igk,
        iv=iv,
        matrices=matrices,
    )


def _build_surface_common_state(
    bulk: BulkSimulation,
    domain: BulkDomain,
) -> tuple[int, list[int], list[int], list[list[int]], float, float, float, torch.Tensor, torch.Tensor]:
    """Build beam-difference and reciprocal-space state shared across a surface batch."""

    nv, igh, igk, iv = _srfghk(domain.nb, domain.ih, domain.ik, domain.igh, domain.igk)
    ghx = (2.0 * math.pi) / (bulk.source.aa * bulk.source.nh)
    ghy = -(2.0 * math.pi) / (bulk.source.aa * math.tan(bulk.gam_rad) * bulk.source.nh)
    gky = (2.0 * math.pi) / (bulk.source.bb * math.sin(bulk.gam_rad) * bulk.source.nk)
    gh = torch.tensor(
        [(-(2.0 * math.pi) / bulk.source.nh) * float(v) for v in igh],
        dtype=torch.float64,
        device=bulk.device,
    )
    gk = torch.tensor(
        [(-(2.0 * math.pi) / bulk.source.nk) * float(v) for v in igk],
        dtype=torch.float64,
        device=bulk.device,
    )
    return nv, igh, igk, iv, ghx, ghy, gky, gh, gk


def _build_surface_potential(
    bulk: BulkSimulation,
    surface: SurfaceInput,
    domain: BulkDomain,
    *,
    nv: int,
    igh: list[int],
    igk: list[int],
    ghx: float,
    ghy: float,
    gky: float,
    gh: torch.Tensor,
    gk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Assemble the surface-side scattering potential for one structure."""

    iz_all = surface.iz + bulk.source.iz
    da1_all = surface.da1 + bulk.source.da1
    sap_all = surface.sap + bulk.source.sap
    bh_all = surface.bh + bulk.source.bh
    bk_all = surface.bk + bulk.source.bk
    bz_all = surface.bz + bulk.source.bz

    _, a_all, b_all, c_all = _prepare_atomic_factor_tables(
        iz_all,
        da1_all,
        bz_all,
        bulk.source.be,
        bulk.source.aa,
        bulk.source.bb,
        bulk.gam_rad,
        bulk.device,
    )
    asf = _asfcef(igh, igk, bh_all, bk_all, ghx, ghy, gky, a_all, b_all, c_all)

    if surface.nsgs < 1:
        ns = 0
    elif surface.natms > 0:
        ns = int((max(surface.z) + surface.dthick + bulk.source.cc) / bulk.dz) + 1
    else:
        ns = int((surface.dthick + bulk.source.cc) / bulk.dz) + 1

    v = torch.zeros((nv, ns), dtype=torch.complex128, device=bulk.device)
    vi = torch.zeros((nv, ns), dtype=torch.complex128, device=bulk.device)

    ielm_bulk = [value + surface.nelms for value in bulk.source.ielm]
    _scpot(
        nv_use=nv,
        ns=ns,
        ielm=surface.ielm,
        ocr=surface.ocr,
        x=surface.x,
        y=surface.y,
        z=surface.z,
        zo=-bulk.source.cc,
        dz=bulk.dz,
        sap=sap_all,
        nsg=surface.nsgs,
        ma=surface.msa,
        mb=surface.msb,
        na=surface.nsa,
        nb=surface.nsb,
        gh=gh,
        gk=gk,
        dx=surface.dxs + bulk.source.dx,
        dy=surface.dys + bulk.source.dy,
        asf=asf,
        b_all=b_all,
        v=v,
        vi=vi,
        iclr=1,
    )
    _scpot(
        nv_use=domain.nvb,
        ns=ns,
        ielm=ielm_bulk,
        ocr=bulk.source.ocr,
        x=bulk.source.x,
        y=bulk.source.y,
        z=bulk.source.z,
        zo=0.0,
        dz=bulk.dz,
        sap=sap_all,
        nsg=bulk.source.nsg,
        ma=1,
        mb=0,
        na=0,
        nb=1,
        gh=gh,
        gk=gk,
        dx=bulk.source.dx,
        dy=bulk.source.dy,
        asf=asf,
        b_all=b_all,
        v=v,
        vi=vi,
        iclr=0,
    )
    _scpot(
        nv_use=domain.nvb,
        ns=ns,
        ielm=ielm_bulk,
        ocr=bulk.source.ocr,
        x=bulk.source.x,
        y=bulk.source.y,
        z=bulk.source.z,
        zo=bulk.source.cc,
        dz=bulk.dz,
        sap=sap_all,
        nsg=bulk.source.nsg,
        ma=1,
        mb=0,
        na=0,
        nb=1,
        gh=gh,
        gk=gk,
        dx=0.0,
        dy=0.0,
        asf=asf,
        b_all=b_all,
        v=v,
        vi=vi,
        iclr=-1,
    )
    return v, vi, ns


def _assemble_initial_reflection_batch(domain: BulkDomain, n_angles: int) -> torch.Tensor:
    """Assemble the bulk-reflection seed matrices for every scan angle."""

    device = domain.matrices[0].device
    f = torch.zeros((n_angles, domain.nb, domain.nb), dtype=torch.complex128, device=device)
    matrix_index = 0
    ib1 = 0
    for group_index in range(domain.ngr):
        ib2 = ib1 + domain.nbg[group_index]
        block_batch = torch.stack(
            [domain.matrices[angle_index * domain.ngr + group_index] for angle_index in range(n_angles)],
            dim=0,
        )
        group_indices = torch.tensor(domain.jorg[ib1:ib2], dtype=torch.long, device=device)
        f[:, group_indices.unsqueeze(1), group_indices.unsqueeze(0)] = block_batch
        ib1 = ib2
        matrix_index += n_angles
    return f


def simulate_bulk(bulk_input: BulkInput, *, device: str | torch.device | None = None) -> BulkSimulation:
    """Run the bulk stage from a parsed `bulk.txt` input."""

    if bulk_input.ndom != 1:
        raise NotImplementedError("torch_rheed currently supports only single-domain inputs")
    _require_supported_geometry(bulk_input.nsg, "bulk")
    resolved_device = _resolve_device(device)

    azi_rad, daz_rad, naz, gi_rad, dg_rad, ng = _scan_range(
        bulk_input.azi_deg,
        bulk_input.azf_deg,
        bulk_input.daz_deg,
        bulk_input.gi_deg,
        bulk_input.gf_deg,
        bulk_input.dg_deg,
    )
    gam_rad = bulk_input.gam_deg * math.pi / 180.0
    dz = bulk_input.cc / (int(bulk_input.cc / bulk_input.dz_input) + 1)
    epsb = 1e-10
    inegpos = -1
    idiag = 0

    wn, a_all, b_all, c_all = _prepare_atomic_factor_tables(
        bulk_input.iz,
        bulk_input.da1,
        bulk_input.bz,
        bulk_input.be,
        bulk_input.aa,
        bulk_input.bb,
        gam_rad,
        resolved_device,
    )

    domains: list[BulkDomain] = []
    ibt = 0
    for domain_index, count in enumerate(bulk_input.nb):
        ih = bulk_input.ih[ibt : ibt + count]
        ik = bulk_input.ik[ibt : ibt + count]
        domain = _simulate_bulk_domain(
            bulk_input,
            rdom_rad=bulk_input.rdom_deg[domain_index] * math.pi / 180.0,
            ih=ih,
            ik=ik,
            dz=dz,
            naz=naz,
            ng=ng,
            azi_rad=azi_rad,
            daz_rad=daz_rad,
            gi_rad=gi_rad,
            dg_rad=dg_rad,
            epsb=epsb,
            gam_rad=gam_rad,
            wn=wn,
            a_all=a_all,
            b_all=b_all,
            c_all=c_all,
        )
        domains.append(domain)
        ibt += count

    return BulkSimulation(
        source=bulk_input,
        device=resolved_device,
        inegpos=inegpos,
        idiag=idiag,
        naz=naz,
        ng=ng,
        azi_rad=azi_rad,
        daz_rad=daz_rad,
        gi_rad=gi_rad,
        dg_rad=dg_rad,
        dz=dz,
        epsb=epsb,
        gam_rad=gam_rad,
        wn=wn,
        domains=domains,
    )


def simulate_rocking_curve(
    bulk: BulkSimulation,
    surface: SurfaceInput,
    *,
    screen_config: ScreenImageConfig | None = DEFAULT_SCREEN_IMAGE_CONFIG,
) -> RockingCurveResult:
    """Run the surface rocking-curve solver and detector-image renderer for one structure."""

    return simulate_rocking_curve_batch(bulk, [surface], screen_config=screen_config).get_result(0)


def simulate_rocking_curve_batch(
    bulk: BulkSimulation,
    surfaces: list[SurfaceInput],
    *,
    screen_config: ScreenImageConfig | None = DEFAULT_SCREEN_IMAGE_CONFIG,
) -> RockingCurveBatchResult:
    """Run a batched rocking-curve solve and detector-image render for compatible structures."""

    if bulk.source.ndom != 1:
        raise NotImplementedError("torch_rheed currently supports only single-domain inputs")
    if not surfaces:
        raise ValueError("at least one surface structure is required for batched simulation")

    domain = bulk.domains[0]
    for surface in surfaces:
        _require_supported_geometry(surface.nsgs, "surface")
    signature0 = _surface_parameter_signature(surfaces[0])
    for surface in surfaces[1:]:
        if _surface_parameter_signature(surface) != signature0:
            raise NotImplementedError(
                "batched multi-structure simulation currently requires identical surface element parameter blocks"
            )

    nv, igh, igk, iv, ghx, ghy, gky, gh, gk = _build_surface_common_state(bulk, domain)
    nb0 = next(i for i, (ih, ik) in enumerate(zip(domain.ih, domain.ik)) if ih == 0 and ik == 0)
    angle_tensor, wnsga2, wave_xy = _build_scan_vectors(
        naz=bulk.naz,
        ng=bulk.ng,
        azi_rad=bulk.azi_rad,
        daz_rad=bulk.daz_rad,
        gi_rad=bulk.gi_rad,
        dg_rad=bulk.dg_rad,
        rdom_rad=domain.rdom_rad,
        wn=bulk.wn,
        device=bulk.device,
    )
    n_angles = angle_tensor.numel()
    n_structures = len(surfaces)

    potentials: list[torch.Tensor] = []
    imaginary_potentials: list[torch.Tensor] = []
    layer_counts: list[int] = []
    for surface in surfaces:
        v, vi, ns = _build_surface_potential(
            bulk,
            surface,
            domain,
            nv=nv,
            igh=igh,
            igk=igk,
            ghx=ghx,
            ghy=ghy,
            gky=gky,
            gh=gh,
            gk=gk,
        )
        potentials.append(v)
        imaginary_potentials.append(vi)
        layer_counts.append(ns)

    max_ns = max(layer_counts)
    v_batch = torch.zeros((n_structures, nv, max_ns), dtype=torch.complex128, device=bulk.device)
    vi_batch = torch.zeros((n_structures, nv, max_ns), dtype=torch.complex128, device=bulk.device)
    active_layers = torch.zeros((n_structures, max_ns), dtype=torch.bool, device=bulk.device)
    for index, (v, vi, ns) in enumerate(zip(potentials, imaginary_potentials, layer_counts)):
        if ns > 0:
            v_batch[index, :, :ns] = v
            vi_batch[index, :, :ns] = vi
            active_layers[index, :ns] = True

    ih_tensor = torch.tensor(domain.ih, dtype=torch.float64, device=bulk.device).unsqueeze(0)
    ik_tensor = torch.tensor(domain.ik, dtype=torch.float64, device=bulk.device).unsqueeze(0)
    wnx = wave_xy[:, 0].unsqueeze(1)
    wny = wave_xy[:, 1].unsqueeze(1)
    sval = bulk.wn * bulk.wn - (ghx * ih_tensor + wnx).square() - (ghy * ih_tensor + gky * ik_tensor + wny).square()
    gma = _complex_sqrt_from_sval(sval)

    f = _assemble_initial_reflection_batch(domain, n_angles).unsqueeze(0).expand(n_structures, -1, -1, -1).clone()
    gma_batch = gma.unsqueeze(0).expand(n_structures, -1, -1)
    identity = torch.eye(domain.nb + domain.nb, dtype=torch.complex128, device=bulk.device).view(
        1,
        1,
        domain.nb + domain.nb,
        domain.nb + domain.nb,
    )

    for layer in range(max_ns):
        v_flat = v_batch[:, :, layer].unsqueeze(1).expand(-1, n_angles, -1).reshape(-1, nv)
        vi_flat = vi_batch[:, :, layer].unsqueeze(1).expand(-1, n_angles, -1).reshape(-1, nv)
        transfer = _trmat(
            domain.nb,
            gma_batch.reshape(-1, domain.nb),
            v_flat,
            vi_flat,
            iv,
            bulk.dz,
        ).reshape(n_structures, n_angles, domain.nb + domain.nb, domain.nb + domain.nb)
        transfer = torch.where(active_layers[:, layer].view(-1, 1, 1, 1), transfer, identity)
        t2 = transfer[:, :, : domain.nb, domain.nb :] + transfer[:, :, : domain.nb, : domain.nb] @ f
        t3 = transfer[:, :, domain.nb :, domain.nb :] + transfer[:, :, domain.nb :, : domain.nb] @ f
        f = _right_solve(t2, t3)

    sval_real = gma.real.unsqueeze(0)
    amp2 = torch.abs(f[:, :, :, nb0]).square()
    numerator = amp2 * wnsga2.view(1, n_angles, 1)
    intensity_tensor = torch.where(
        sval_real > 1e-10,
        numerator / sval_real,
        torch.zeros_like(numerator),
    )
    screen_images = None
    resolved_screen_config = None
    if screen_config is not None:
        screen_images, resolved_screen_config = render_screen_stack(
            bulk,
            list(zip(domain.ih, domain.ik)),
            angle_tensor,
            intensity_tensor,
            config=screen_config,
        )
    return RockingCurveBatchResult(
        beam_indices=list(zip(domain.ih, domain.ik)),
        angles_deg=angle_tensor,
        intensities=intensity_tensor,
        naz=bulk.naz,
        ng=bulk.ng,
        surface_paths=[surface.source_path for surface in surfaces],
        screen_images=screen_images,
        screen_config=resolved_screen_config,
    )


def simulate_from_files(
    bulk_path: Path,
    surface_path: Path,
    *,
    device: str | torch.device | None = None,
    screen_config: ScreenImageConfig | None = DEFAULT_SCREEN_IMAGE_CONFIG,
) -> RockingCurveResult:
    """Run a complete standalone simulation from `bulk.txt` and `surf.txt`."""

    bulk_input = load_bulk_input(bulk_path)
    surface_input = load_surface_input(surface_path, bulk_input.ndom)
    bulk = simulate_bulk(bulk_input, device=device)
    return simulate_rocking_curve(bulk, surface_input, screen_config=screen_config)


def simulate_from_files_batch(
    bulk_path: Path,
    surface_paths: list[Path],
    *,
    device: str | torch.device | None = None,
    screen_config: ScreenImageConfig | None = DEFAULT_SCREEN_IMAGE_CONFIG,
) -> RockingCurveBatchResult:
    """Run the optimized shared-bulk batch path for multiple surface files."""

    if not surface_paths:
        raise ValueError("at least one surface path is required for batched simulation")
    bulk_input = load_bulk_input(bulk_path)
    bulk = simulate_bulk(bulk_input, device=device)
    surfaces = [load_surface_input(path, bulk_input.ndom) for path in surface_paths]
    return simulate_rocking_curve_batch(bulk, surfaces, screen_config=screen_config)


def simulate_pairs_batch(
    input_pairs: list[tuple[Path, Path]],
    *,
    device: str | torch.device | None = None,
    screen_config: ScreenImageConfig | None = DEFAULT_SCREEN_IMAGE_CONFIG,
) -> RockingCurvePairBatchResult:
    """Run a high-level batch of ``(bulk.txt, surf.txt)`` pairs.

    The scheduler groups compatible pairs onto the shared-bulk fast path where
    possible, while preserving the original input order in the returned results.
    """

    if not input_pairs:
        raise ValueError("at least one (bulk_path, surface_path) pair is required")

    loaded_pairs: list[tuple[int, Path, Path, BulkInput, SurfaceInput, tuple[object, ...], tuple[object, ...]]] = []
    for index, (bulk_path, surface_path) in enumerate(input_pairs):
        bulk_input = load_bulk_input(bulk_path)
        surface_input = load_surface_input(surface_path, bulk_input.ndom)
        loaded_pairs.append(
            (
                index,
                bulk_path,
                surface_path,
                bulk_input,
                surface_input,
                _bulk_input_signature(bulk_input),
                _surface_parameter_signature(surface_input),
            )
        )

    grouped_pairs: dict[tuple[object, ...], list[tuple[int, Path, Path, BulkInput, SurfaceInput, tuple[object, ...], tuple[object, ...]]]] = {}
    for item in loaded_pairs:
        group_key = (item[5], item[6])
        grouped_pairs.setdefault(group_key, []).append(item)

    bulk_cache: dict[tuple[object, ...], BulkSimulation] = {}
    ordered_results: list[RockingCurveResult | None] = [None] * len(input_pairs)

    for group_items in grouped_pairs.values():
        bulk_input = group_items[0][3]
        bulk_signature = group_items[0][5]
        bulk = bulk_cache.get(bulk_signature)
        if bulk is None:
            bulk = simulate_bulk(bulk_input, device=device)
            bulk_cache[bulk_signature] = bulk

        surfaces = [item[4] for item in group_items]
        if len(surfaces) == 1:
            ordered_results[group_items[0][0]] = simulate_rocking_curve(
                bulk,
                surfaces[0],
                screen_config=screen_config,
            )
            continue

        batch_result = simulate_rocking_curve_batch(
            bulk,
            surfaces,
            screen_config=screen_config,
        )
        for local_index, item in enumerate(group_items):
            ordered_results[item[0]] = batch_result.get_result(local_index)

    if any(result is None for result in ordered_results):
        raise RuntimeError("pair-batch scheduling failed to populate every result")

    return RockingCurvePairBatchResult(
        input_pairs=list(input_pairs),
        results=[result for result in ordered_results if result is not None],
    )
