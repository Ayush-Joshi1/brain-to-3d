"""Create the concise assignment report as a dependency-free PDF."""

from __future__ import annotations

import textwrap
from pathlib import Path


LINES = [
    "Reconstructing 3D Objects from Brain Activity",
    "Final report | Iteration 4 | 11 October 2026",
    "",
    "Problem and data",
    "The task maps fMRI recordings to canonical 2048-point 3D object clouds.",
    "The supplied training split has 533 scans and 292 unique objects; validation",
    "has 125 scans and 67 objects; test has 167 scans and IDs only. Training",
    "fMRI was averaged over 10 time points for the retrieval baseline. The selected",
    "technical fallback uses only training object point clouds and the test IDs.",
    "Outputs preserve the supplied origin-centered, z-up coordinate frame.",
    "",
    "Model selection and validation",
    "The generic training medoid was strongest overall. Official evaluate.py",
    "metrics on 125 validation scans:",
    "  Chamfer 0.1562 | F-score 0.3947 | silhouette IoU 0.4384",
    "  silhouette SSIM 0.6919 | identification 48.76% | composite -0.013",
    "Direct standardized-fMRI retrieval scored composite -0.158. The Iteration 2",
    "predicted-shape-codebook model scored Chamfer 0.2211, F-score 0.2659,",
    "IoU 0.2939, SSIM 0.6844, identification 52.59%, composite -0.141.",
    "Its shuffled-association control preserved validation IDs, permuted only",
    "predictions, and used the same official evaluator. It scored identification",
    "57.59% and composite -0.055, better than the unshuffled model. This single",
    "shuffle is not a significance test; brain dependence is not established.",
    "",
    "A fold-local subject-centering experiment used five object-grouped training",
    "folds; global vs subject-centered latent MSE was 5.2491 vs 5.2651. This",
    "approximately 0.3% regression is a negative exploratory result, not official",
    "geometry evaluation. Group IDs were treated as split-local indices. Review",
    "also found Iteration 2 PCA/scaling were fitted before CV folds, a limitation",
    "of its internal CV estimate; no held-out validation targets fit the model.",
    "No validation target was used to fit transforms or model weights.",
    "",
    "Test outputs and reproducibility",
    "submission/predictions.npz contains 167 unique test IDs in original order",
    "and float32 points with shape (167, 2048, 3). Coordinates are finite. The",
    "same verified training-medoid cloud is repeated for all test scans. Therefore",
    "this fallback ignores test fMRI and is not a brain-conditioned solution; it",
    "may fail the assignment brain-use gate. No test ground truth was accessed.",
    "Calibrated PNG renders and PLY clouds are produced by the supplied starter",
    "helpers using cameras 0, 8, 16, and 24. No pretrained model was used.",
    "",
    "Reproduction prerequisites and working directory",
    "Run these commands in the assignment workspace root: the directory containing",
    "data/, starter.py, metrics.py, evaluate.py, and check_submission.py. Use",
    "Python 3.13 with NumPy, SciPy, and Pillow (install with python -m pip install",
    "-r submission/code/requirements.txt). The generator reuses the verified",
    "Iteration 3 archive when available; otherwise it uses the bundled training-",
    "derived generic_medoid.npy. The validation script invokes the supplied",
    "official evaluator on validation targets; no test targets are used.",
    "  python submission/code/generate_submission.py",
    "  python submission/code/evaluate_validation.py",
    "  python check_submission.py submission",
    "The official checker passed: predictions.npz, 167 PLY files, 668 renders,",
    "validation metrics, code, and report were verified.",
    "",
    "Limitations and next steps",
    "No tested model beats the generic medoid composite and shuffled control",
    "results do not support reliable scan-specific reconstruction. The submission",
    "is a technical backup, not a solved fMRI-to-3D model. The official submission",
    "checker passed for the listed artifacts, but that structural pass does not",
    "validate brain dependence. Any follow-on work should improve train-only",
    "brain-to-shape learning and use repeated permutation controls.",
    "",
    "References",
    "Gao et al., MinD-3D: Reconstruct High-quality 3D objects in Human Brain,",
    "arXiv:2312.07485 (2023).",
    "Gao et al., MinD-3D++: Advancing fMRI-Based 3D Reconstruction with",
    "High-Quality Textured Mesh Generation and a Comprehensive Dataset,",
    "arXiv:2409.11315.",
]


def pdf_escape(value: str) -> bytes:
    return value.encode("cp1252", errors="replace").hex().upper().encode("ascii")


def wrap_lines() -> list[str]:
    wrapped: list[str] = []
    for line in LINES:
        if not line:
            wrapped.append("")
        else:
            wrapped.extend(textwrap.wrap(line, width=92, subsequent_indent="  "))
    return wrapped


def build_pdf(path: Path) -> None:
    lines = wrap_lines()
    lines_per_page = 48
    pages = [lines[i : i + lines_per_page] for i in range(0, len(lines), lines_per_page)]
    if len(pages) > 3:
        raise ValueError(f"Report exceeds three pages: {len(pages)}")

    objects: list[bytes] = []
    page_ids = [4 + index * 2 for index in range(len(pages))]
    content_ids = [page_id + 1 for page_id in page_ids]
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = b" ".join(f"{page_id} 0 R".encode("ascii") for page_id in page_ids)
    objects.append(b"<< /Type /Pages /Kids [" + kids + b"] /Count " + str(len(pages)).encode() + b" >>")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    for page_id, content_id, page_lines in zip(page_ids, content_ids, pages, strict=True):
        commands = [b"BT", b"/F1 9 Tf", b"48 760 Td", b"12 TL"]
        for line in page_lines:
            commands.append(b"<" + pdf_escape(line) + b"> Tj")
            commands.append(b"T*")
        commands.append(b"ET")
        stream = b"\n".join(commands)
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 3 0 R >> >> /Contents "
            + str(content_id).encode()
            + b" 0 R >>"
        )
        objects.append(
            b"<< /Length "
            + str(len(stream)).encode()
            + b" >>\nstream\n"
            + stream
            + b"\nendstream"
        )

    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode("ascii"))
        output.extend(obj)
        output.extend(b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    path.write_bytes(output)
    print(f"Created {path} ({len(pages)} pages, {len(output)} bytes)")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    build_pdf(root / "submission" / "report.pdf")
