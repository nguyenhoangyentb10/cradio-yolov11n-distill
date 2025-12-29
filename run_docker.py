import argparse
import os
import shlex
import subprocess
from pathlib import Path


def run(cmd: str, check=True):
    """Run shell command and stream output."""
    print(f"\n$ {cmd}")
    p = subprocess.run(cmd, shell=True)
    if check and p.returncode != 0:
        raise SystemExit(f"Command failed ({p.returncode}): {cmd}")
    return p.returncode


def capture(cmd: str) -> str:
    """Run shell command and capture stdout."""
    p = subprocess.run(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    return p.stdout.strip()


def docker_ok():
    out = capture("docker --version")
    if "Docker version" not in out:
        raise SystemExit(
            "Docker chưa sẵn sàng trong WSL. Kiểm tra Docker Desktop + WSL integration."
        )
    print(out)
    run("docker info > /dev/null && echo DOCKER_OK", check=True)


def image_exists(tag: str) -> bool:
    out = capture(
        "docker images --format '{{.Repository}}:{{.Tag}}' | "
        f"grep -x {shlex.quote(tag)} || true"
    )
    return bool(out.strip())


def build_image(tag: str, dockerfile: str):
    run("docker buildx use default", check=False)
    run(f"docker build -t {shlex.quote(tag)} -f {shlex.quote(dockerfile)} .", check=True)


def test_gpu(tag: str) -> bool:
    """Return True if CUDA is available inside container (requires NVIDIA runtime)."""
    cmd = (
        f"docker run --rm --gpus all {shlex.quote(tag)} "
        "python -c \"import torch; "
        "print('cuda:', torch.cuda.is_available()); "
        "print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)\""
    )
    out = capture(cmd)
    print(out)
    return "cuda: True" in out


def run_pipeline(
    tag: str,
    det_pt: str,
    cls_pt: str,
    img: str,
    out_dir: str,
    device: str,
    smoking_action_class: str = "smoke",
    cigarette_class: str = "smoke",
    cls_positive_labels: str = "smoking,smoke,cigarette,hút_thuốc",
    cls_threshold: float = 0.6,
):
    """
    Run src/pipeline.py inside docker container.
    Notes:
      - Your detection model uses class name 'smoke' (NOT 'smoking').
      - If you don't have a separate 'cigarette' class, set cigarette_class='smoke'
        so the pipeline still crops something.
    """
    pwd = Path(os.getcwd())
    models_dir = pwd / "models"
    input_dir = pwd / "input"
    output_dir = pwd / out_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Decide whether to request GPU from docker
    use_gpu_flag = (device != "cpu")

    docker_gpu = "--gpus all " if use_gpu_flag else ""
    yolo_config_dir = "/workspace/output/.ultralytics"

    cmd = (
        f"docker run --rm {docker_gpu}"
        f"-e YOLO_CONFIG_DIR={shlex.quote(yolo_config_dir)} "
        f"-v {shlex.quote(str(models_dir))}:/workspace/models "
        f"-v {shlex.quote(str(input_dir))}:/workspace/input "
        f"-v {shlex.quote(str(output_dir))}:/workspace/output "
        f"{shlex.quote(tag)} "
        f"python src/pipeline.py "
        f"--det /workspace/models/{shlex.quote(Path(det_pt).name)} "
        f"--cls /workspace/models/{shlex.quote(Path(cls_pt).name)} "
        f"--img /workspace/input/{shlex.quote(Path(img).name)} "
        f"--out /workspace/output "
        f"--device {shlex.quote(device)} "
        f"--smoking_action_class {shlex.quote(smoking_action_class)} "
        f"--cigarette_class {shlex.quote(cigarette_class)} "
        f"--cls_positive_labels {shlex.quote(cls_positive_labels)} "
        f"--cls_threshold {cls_threshold}"
    )
    run(cmd, check=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="smoking-pipeline:gpu", help="docker image tag")
    ap.add_argument("--dockerfile", default="docker/Dockerfile.gpu", help="dockerfile path")
    ap.add_argument("--build", action="store_true", help="force build image (even if exists)")

    ap.add_argument("--det", default="models/det_best.pt", help="path to detection pt (local)")
    ap.add_argument("--cls", default="models/cls_best.pt", help="path to classification pt (local)")
    ap.add_argument("--img", default="input/img1.png", help="input image (local)")
    ap.add_argument("--out", default="output", help="output folder (local)")
    ap.add_argument("--device", default="0", help='YOLO device inside container: "0" or "cpu"')

    # IMPORTANT: your detection class is 'smoke'
    ap.add_argument("--smoking_action_class", default="smoke", help="detection class name for smoking action")
    ap.add_argument("--cigarette_class", default="smoke", help="detection class name for cigarette (or same as smoke)")
    ap.add_argument("--cls_positive_labels", default="smoking,smoke,cigarette,hút_thuốc", help="comma labels")
    ap.add_argument("--cls_threshold", type=float, default=0.6, help="classification threshold")

    return ap.parse_args()


def main():
    args = parse_args()

    # basic file checks
    for p in [args.det, args.cls, args.img]:
        if not Path(p).exists():
            raise SystemExit(f"Không thấy file: {p}")

    docker_ok()

    if args.build or (not image_exists(args.tag)):
        print(f"Building image: {args.tag}")
        build_image(args.tag, args.dockerfile)
    else:
        print(f"Image exists: {args.tag}")

    # GPU test only if user intends to use GPU
    if args.device != "cpu":
        print("\nTesting CUDA inside container...")
        cuda_ok = test_gpu(args.tag)
        if not cuda_ok:
            print("⚠️ CUDA trong container = False. Bạn có thể chạy CPU bằng --device cpu.")

    print("\nRunning pipeline...")
    run_pipeline(
        tag=args.tag,
        det_pt=args.det,
        cls_pt=args.cls,
        img=args.img,
        out_dir=args.out,
        device=args.device,
        smoking_action_class=args.smoking_action_class,
        cigarette_class=args.cigarette_class,
        cls_positive_labels=args.cls_positive_labels,
        cls_threshold=args.cls_threshold,
    )

    print("\nDone. Check outputs:")
    print(f" - {args.out}/json/")
    print(f" - {args.out}/crops/")
    print(f" - {args.out}/annotated/")


if __name__ == "__main__":
    main()
