
















from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


MVTEC_CLASSES: Sequence[str] = (
    "carpet",
    "grid",
    "leather",
    "tile",
    "wood",
    "bottle",
    "cable",
    "capsule",
    "hazelnut",
    "metal_nut",
    "pill",
    "screw",
    "toothbrush",
    "transistor",
    "zipper",
)

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


class MetaGenerationError(RuntimeError):
    pass

def image_files(directory: Path) -> List[Path]:

    if not directory.is_dir():
        return []
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name.casefold(),
    )


def relative_posix(path: Path, root: Path) -> str:

    return path.resolve().relative_to(root).as_posix()


def resolve_mask(image_path: Path, mask_dir: Path) -> Path:

    if not mask_dir.is_dir():
        raise MetaGenerationError(
            f"Missing ground-truth directory for {image_path}: {mask_dir}"
        )

    
    
    candidate_suffixes = [image_path.suffix.lower()] + sorted(
        IMAGE_EXTENSIONS - {image_path.suffix.lower()}
    )
    candidates = [mask_dir / f"{image_path.stem}_mask{suffix}" for suffix in candidate_suffixes]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    
    expected_stem = f"{image_path.stem}_mask".casefold()
    matches = [
        path
        for path in image_files(mask_dir)
        if path.stem.casefold() == expected_stem
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise MetaGenerationError(
            f"Multiple masks match {image_path}: {names}"
        )
    raise MetaGenerationError(
        f"No mask found for {image_path}. Expected a file such as "
        f"{mask_dir / (image_path.stem + '_mask.png')}"
    )


def discover_classes(root: Path) -> List[str]:

    return sorted(
        (
            path.name
            for path in root.iterdir()
            if path.is_dir()
            and (path / "train").is_dir()
            and (path / "test").is_dir()
        ),
        key=str.casefold,
    )


def select_classes(
    root: Path,
    requested: Sequence[str] | None,
    allow_partial: bool,
) -> List[str]:
    available = set(discover_classes(root))
    if requested:
        classes = list(dict.fromkeys(requested))
    elif allow_partial:
        classes = [name for name in MVTEC_CLASSES if name in available]
        classes.extend(sorted(available - set(MVTEC_CLASSES), key=str.casefold))
    else:
        classes = list(MVTEC_CLASSES)

    missing = [name for name in classes if name not in available]
    if missing:
        hint = (
            " Use --allow-partial only if this is intentionally a subset of MVTec AD."
            if not allow_partial and not requested
            else ""
        )
        raise MetaGenerationError(
            f"Missing or incomplete class directories under {root}: "
            f"{', '.join(missing)}.{hint}"
        )
    if not classes:
        raise MetaGenerationError(
            f"No MVTec classes were found under {root}. The root must contain "
            "directories such as bottle/train and bottle/test."
        )
    return classes


def build_phase_entries(root: Path, class_name: str, phase: str) -> List[dict]:
    class_dir = root / class_name
    phase_dir = class_dir / phase
    if not phase_dir.is_dir():
        raise MetaGenerationError(f"Missing directory: {phase_dir}")

    species = sorted(
        (path for path in phase_dir.iterdir() if path.is_dir()),
        key=lambda path: path.name.casefold(),
    )
    if not species:
        raise MetaGenerationError(f"No defect-type directories found in {phase_dir}")

    entries: List[dict] = []
    for specie_dir in species:
        specie_name = specie_dir.name
        anomaly = int(specie_name.casefold() != "good")
        images = image_files(specie_dir)
        if not images:
            raise MetaGenerationError(f"No images found in {specie_dir}")

        for image_path in images:
            mask_path = ""
            if anomaly:
                mask = resolve_mask(
                    image_path,
                    class_dir / "ground_truth" / specie_name,
                )
                mask_path = relative_posix(mask, root)

            entries.append(
                {
                    "img_path": relative_posix(image_path, root),
                    "mask_path": mask_path,
                    "cls_name": class_name,
                    "specie_name": specie_name,
                    "anomaly": anomaly,
                }
            )
    return entries


def build_meta(root: Path, classes: Iterable[str]) -> Dict[str, Dict[str, List[dict]]]:
    meta: Dict[str, Dict[str, List[dict]]] = {"train": {}, "test": {}}
    for class_name in classes:
        for phase in ("train", "test"):
            meta[phase][class_name] = build_phase_entries(root, class_name, phase)
    return meta


def validate_meta(meta: dict, root: Path) -> None:

    required = {"img_path", "mask_path", "cls_name", "specie_name", "anomaly"}
    errors: List[str] = []
    for phase in ("train", "test"):
        if phase not in meta or not isinstance(meta[phase], dict):
            errors.append(f"Missing top-level object: {phase}")
            continue
        for class_name, entries in meta[phase].items():
            for index, item in enumerate(entries):
                location = f"{phase}/{class_name}[{index}]"
                missing_keys = required - set(item)
                if missing_keys:
                    errors.append(f"{location}: missing keys {sorted(missing_keys)}")
                    continue
                if item["cls_name"] != class_name:
                    errors.append(f"{location}: cls_name does not match its class group")
                if item["anomaly"] not in (0, 1):
                    errors.append(f"{location}: anomaly must be 0 or 1")
                if not (root / item["img_path"]).is_file():
                    errors.append(f"{location}: image does not exist: {item['img_path']}")
                if item["anomaly"] == 1:
                    if not item["mask_path"]:
                        errors.append(f"{location}: anomalous sample has no mask_path")
                    elif not (root / item["mask_path"]).is_file():
                        errors.append(f"{location}: mask does not exist: {item['mask_path']}")
                elif item["mask_path"] != "":
                    errors.append(f"{location}: normal sample mask_path must be empty")

    if errors:
        preview = "\n".join(f"  - {error}" for error in errors[:20])
        remainder = len(errors) - 20
        suffix = f"\n  ... and {remainder} more" if remainder > 0 else ""
        raise MetaGenerationError(f"Metadata validation failed:\n{preview}{suffix}")


def print_summary(meta: dict, output: Path) -> None:
    totals = defaultdict(int)
    print("\nGenerated class counts:")
    print(f"{'class':<14} {'train':>7} {'test':>7} {'test_bad':>10}")
    print("-" * 42)
    for class_name in meta["train"]:
        train_count = len(meta["train"][class_name])
        test_count = len(meta["test"][class_name])
        test_bad = sum(item["anomaly"] for item in meta["test"][class_name])
        totals["train"] += train_count
        totals["test"] += test_count
        totals["test_bad"] += test_bad
        print(f"{class_name:<14} {train_count:>7} {test_count:>7} {test_bad:>10}")
    print("-" * 42)
    print(
        f"{'TOTAL':<14} {totals['train']:>7} {totals['test']:>7} "
        f"{totals['test_bad']:>10}"
    )
    print(f"\nSaved: {output}")
    if len(meta["train"]) == len(MVTEC_CLASSES):
        if totals["train"] == 3629 and totals["test"] == 1725:
            print("Dataset count check: PASS (standard MVTec AD 15-class release)")
        else:
            print(
                "WARNING: all 15 classes were found, but the totals differ from "
                "the commonly used MVTec AD release (train=3629, test=1725)."
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate mvtec/meta.json in the schema expected by ADer."
    )
    parser.add_argument(
        "--root",
        "--data-root",
        dest="root",
        default="mvtec",
        help="MVTec dataset root (default: mvtec).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: <root>/meta.json).",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Generate only the named classes, for example: --classes bottle cable.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Discover and accept a partial/non-standard class set.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing meta.json after validation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else root / "meta.json"
    )

    try:
        if not root.is_dir():
            raise MetaGenerationError(f"Dataset root does not exist: {root}")
        if output.exists() and not args.overwrite:
            raise MetaGenerationError(
                f"Output already exists: {output}. Add --overwrite to replace it."
            )

        classes = select_classes(root, args.classes, args.allow_partial)
        print(f"Dataset root: {root}")
        print(f"Classes ({len(classes)}): {', '.join(classes)}")

        meta = build_meta(root, classes)
        validate_meta(meta, root)

        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp")
        temporary.write_text(
            json.dumps(meta, ensure_ascii=False, indent=4) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
        print_summary(meta, output)
        return 0
    except (MetaGenerationError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
