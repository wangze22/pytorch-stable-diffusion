#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ================== 固定参数（按需改这里） ==================
DATASET_NAME     = "xingjianleng/laion_aesthetics_v2_6.5plus"
SPLIT            = "train"
SCORE_THRESHOLD  = 6.8                # 仅保留美学分 > 该阈值的样本
OUT_DIR          = "./laion_aesthetic6"
INDEX_CSV        = "index.csv"        # 先保存的 CSV 文件名（在 OUT_DIR 下）
FIELDS_IN_CSV    = ["url","caption","score","width","height"]

# 估算体积：从 CSV 抽样多少条做 HEAD/GET 获取 Content-Length
ESTIMATE_SAMPLE_SIZE = 500            # 抽样条数
HEAD_TIMEOUT_SECS    = 8
HEAD_RETRIES         = 1

# 是否在估算后继续真正下载图片
START_DOWNLOAD       = True

# 下载配置
MAX_IMAGES           = 0              # 0 不限；>0 则为过滤阶段写 CSV 的最大行数（上游限流）
NUM_WORKERS          = 128
TIMEOUT_SECS         = 5
RETRIES              = 2
MIN_SIZE             = 256            # 最小边像素，低于则丢弃
SAVE_ORIGINAL        = True

# 新增：下载过程中的硬性上限（任一命中即停止）
MAX_TOTAL_SIZE_GB    = 10.0           # 累计下载体积上限（GB）；设为 0 表示无限制
MAX_DOWNLOAD_IMAGES  = 100000         # 下载图片张数上限；设为 0 表示无限制
# ==========================================================

import sys, asyncio
# Windows: 使用 Selector 事件循环，避免 Proactor 噪音异常
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import os
import io
import csv
import json
import math
import random
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import aiohttp
from aiohttp import ClientTimeout
from datasets import load_dataset
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

DEFAULT_HEADERS = {"User-Agent": "laion-downloader/1.0 (+aiohttp)"}

# -------------------- 工具函数 --------------------
def safe_mkdirs(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

EXT_FROM_PIL = {
    "JPEG": ".jpg",
    "PNG":  ".png",
    "WEBP": ".webp",
    "BMP":  ".bmp",
    "GIF":  ".gif",
    "TIFF": ".tiff",
}

def pick_field(d: Dict[str, Any], names):
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return None

def get_url(row):    # 兼容不同字段名
    return pick_field(row, ["URL", "url", "URL_PROMPT", "image_url", "IMAGE_URL"])

def get_score(row):
    return pick_field(row, ["AESTHETIC_SCORE", "aesthetic_score", "aesthetic", "aesthetics", "score"])

def get_caption(row):
    return pick_field(row, ["TEXT", "text", "caption", "CAPTION", "prompt"]) or ""

def get_size(row) -> Tuple[Optional[int], Optional[int]]:
    w = pick_field(row, ["WIDTH", "width", "W", "ImageWidth"])
    h = pick_field(row, ["HEIGHT", "height", "H", "ImageHeight"])
    try:
        w = int(w) if w is not None else None
        h = int(h) if h is not None else None
    except Exception:
        w, h = None, None
    return w, h

def infer_ext_from_bytes(raw: bytes) -> Optional[str]:
    try:
        im = Image.open(io.BytesIO(raw))
        fmt = (im.format or "").upper()
        return EXT_FROM_PIL.get(fmt, None)
    except Exception:
        return None

def pil_convert_and_bytes(raw: bytes, target_ext: str = ".jpg") -> Optional[bytes]:
    try:
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        bio = io.BytesIO()
        im.save(bio, format="JPEG", quality=95)
        return bio.getvalue()
    except Exception:
        return None

def sizeof_fmt(num_bytes: float) -> str:
    units = ["B","KB","MB","GB","TB"]
    i = 0
    while num_bytes >= 1024 and i < len(units)-1:
        num_bytes /= 1024.0
        i += 1
    return f"{num_bytes:.2f} {units[i]}"

# -------------------- 第一步：写 CSV --------------------
def write_index_csv(out_dir: Path) -> Tuple[int, Path]:
    """
    流式读取数据集，过滤分数 > 阈值，写出 CSV。
    返回：(样本数, CSV 路径)
    """
    ds_stream = load_dataset(DATASET_NAME, split=SPLIT, streaming=True)

    safe_mkdirs(out_dir)
    csv_path = out_dir / INDEX_CSV
    count = 0

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS_IN_CSV)
        writer.writeheader()

        pbar = tqdm(desc="Streaming & filtering -> CSV", unit="row")
        for row in ds_stream:
            score = get_score(row)
            if score is None or float(score) <= float(SCORE_THRESHOLD):
                pbar.update(1)
                continue
            url = get_url(row)
            if not url:
                pbar.update(1)
                continue
            cap = get_caption(row)
            w, h = get_size(row)

            writer.writerow({
                "url": url,
                "caption": cap,
                "score": float(score),
                "width": w if w is not None else "",
                "height": h if h is not None else "",
            })
            count += 1
            pbar.update(1)
            if MAX_IMAGES > 0 and count >= MAX_IMAGES:
                break
        pbar.close()

    print(f"\nCSV 已保存：{csv_path}（{count} 行）")
    return count, csv_path

# -------------------- 第二步：用 Content-Length 预估总大小 --------------------
async def head_or_range_size(session: aiohttp.ClientSession, url: str) -> Optional[int]:
    # 轻量退避重试（HEAD）
    backoff = 0.2
    for attempt in range(HEAD_RETRIES + 1):
        try:
            async with session.head(url, allow_redirects=True, timeout=HEAD_TIMEOUT_SECS) as r:
                if r.status < 400:
                    cl = r.headers.get("Content-Length")
                    if cl and cl.isdigit():
                        return int(cl)
        except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionResetError):
            pass
        if attempt < HEAD_RETRIES:
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 2.0)

    # fallback: 部分 GET（Range）
    try:
        async with session.get(url, headers={"Range": "bytes=0-0"}, allow_redirects=True, timeout=HEAD_TIMEOUT_SECS) as r:
            if r.status in (200, 206):
                cr = r.headers.get("Content-Range")  # bytes 0-0/123456
                if cr and "/" in cr:
                    total = cr.split("/")[-1]
                    if total.isdigit():
                        return int(total)
                cl = r.headers.get("Content-Length")
                if cl and cl.isdigit():
                    return int(cl)
    except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionResetError):
        return None
    return None

async def estimate_total_size(csv_path: Path, sample_size: int) -> Tuple[int, int, float]:
    """
    从 CSV 抽样做 HEAD/GET，估算总大小。
    返回：(样本总数 N、成功获取大小的样本数 K、估算总字节数 est_bytes)
    """
    # 蓄水池抽样 URL
    urls: List[str] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            url = row["url"]
            if len(urls) < sample_size:
                urls.append(url)
            else:
                j = random.randint(0, i)
                if j < sample_size:
                    urls[j] = url

    N = sum(1 for _ in csv.DictReader(csv_path.open("r", encoding="utf-8")))
    if N == 0:
        return 0, 0, 0.0

    # 估算阶段并发不宜过大，保守一些更稳
    connector = aiohttp.TCPConnector(limit=32, ttl_dns_cache=300, enable_cleanup_closed=True)
    timeout = ClientTimeout(total=None)
    sizes: List[int] = []
    async with aiohttp.ClientSession(connector=connector, timeout=timeout, headers=DEFAULT_HEADERS) as session:
        pbar = tqdm(total=len(urls), desc="Estimating total size (HEAD/Range)", unit="url")
        sem = asyncio.Semaphore(32)

        async def worker(u):
            nonlocal sizes
            async with sem:
                s = await head_or_range_size(session, u)
                if s and s > 0:
                    sizes.append(s)
                pbar.update(1)

        tasks = [asyncio.create_task(worker(u)) for u in urls]
        await asyncio.gather(*tasks)
        pbar.close()

    K = len(sizes)
    if K == 0:
        est = 0.0
    else:
        avg = sum(sizes) / K
        est = avg * N
    return N, K, est

# -------------------- 下载器（含“已下载跳过”与“历史计入上限”） --------------------
class Downloader:
    def __init__(self, out_dir: str):
        self.out_dir = Path(out_dir)
        self.img_dir = self.out_dir / "images"
        self.meta_path = self.out_dir / "metadata.jsonl"
        safe_mkdirs(self.img_dir)
        self._seen_urls = set()
        self._seen_sha256 = set()
        self._load_existing_index()
        self._scan_disk_existing()  # 扫描磁盘已有文件名（sha）以兜底

    def _load_existing_index(self):
        if self.meta_path.exists():
            with self.meta_path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        j = json.loads(line)
                        if "url" in j and j["url"]:
                            self._seen_urls.add(j["url"])
                        if "sha256" in j and j["sha256"]:
                            self._seen_sha256.add(j["sha256"])
                    except Exception:
                        continue

    def _scan_disk_existing(self):
        # 扫描 images/** 把文件名(不含扩展)当作 sha 记录到 _seen_sha256
        if not self.img_dir.exists():
            return
        for root, _, files in os.walk(self.img_dir):
            for name in files:
                stem = os.path.splitext(name)[0]
                if len(stem) == 64 and all(c in "0123456789abcdef" for c in stem.lower()):
                    self._seen_sha256.add(stem)

    def preload_existing_stats(self) -> tuple[int, int]:
        """
        从 metadata.jsonl 读取已下载图片 path，统计张数与磁盘体积（字节）。
        若文件缺失，则只统计存在的部分。
        """
        if not self.meta_path.exists():
            return 0, 0
        count = 0
        total_bytes = 0
        with self.meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    j = json.loads(line)
                    rel = j.get("path")
                    if not rel:
                        continue
                    p = (self.out_dir / rel)
                    if p.exists() and p.is_file():
                        count += 1
                        try:
                            total_bytes += p.stat().st_size
                        except OSError:
                            pass
                except Exception:
                    continue
        return count, total_bytes

    def already_have_url(self, url: str) -> bool:
        return url in self._seen_urls

    def already_have_sha(self, sha: str) -> bool:
        return sha in self._seen_sha256

    def write_record(self, rec: Dict[str, Any]):
        with self.meta_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if "url" in rec and rec["url"]:
            self._seen_urls.add(rec["url"])
        if "sha256" in rec and rec["sha256"]:
            self._seen_sha256.add(rec["sha256"])

    def path_from_sha(self, sha: str, ext: str) -> Path:
        sub1, sub2 = sha[:2], sha[2:4]
        p = self.img_dir / sub1 / sub2
        safe_mkdirs(p)
        return p / f"{sha}{ext}"

# -------------------- HTTP helpers --------------------
async def fetch_bytes(session: aiohttp.ClientSession, url: str, timeout: int) -> Optional[bytes]:
    backoff = 0.2
    for _ in range(RETRIES + 1):
        try:
            async with session.get(url, timeout=timeout) as r:
                if r.status == 200:
                    return await r.read()
        except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionResetError):
            pass
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 2.0)
    return None

# -------------------- 单个下载任务（先用 URL 快速跳过） --------------------
async def download_one(
    session: aiohttp.ClientSession,
    dl: Downloader,
    rec: Dict[str, Any],
    pbar: tqdm,
) -> Tuple[Optional[Path], int]:
    """
    返回：(保存路径或 None, 实际保存的字节数)
    """
    url = rec["url"]

    # 1) 无网跳过：若 URL 已在 metadata 里，直接跳过
    if dl.already_have_url(url):
        pbar.update(1)
        return None, 0

    # 2) 开始网络请求
    raw: Optional[bytes] = await fetch_bytes(session, url, TIMEOUT_SECS)
    if not raw:
        pbar.update(1)
        return None, 0

    sha = sha256_bytes(raw)

    # 3) 兜底去重：若磁盘已有相同 sha（含历史散落文件），跳过写入
    if dl.already_have_sha(sha):
        pbar.update(1)
        return None, 0

    # 4) 判格式/转换
    ext = infer_ext_from_bytes(raw)
    if not ext:
        conv = pil_convert_and_bytes(raw, ".jpg")
        if conv:
            raw = conv
            ext = ".jpg"
        else:
            pbar.update(1)
            return None, 0

    # 5) 最小尺寸过滤
    try:
        im = Image.open(io.BytesIO(raw))
        w, h = im.size
        if min(w, h) < MIN_SIZE:
            pbar.update(1)
            return None, 0
    except Exception:
        pbar.update(1)
        return None, 0

    # 6) 落盘
    out_path = dl.path_from_sha(sha, ext)
    if not out_path.exists():
        out_path.write_bytes(raw)

    # 7) 写 metadata
    dl.write_record({
        "path": str(out_path.relative_to(dl.out_dir)),
        "url": url,
        "sha256": sha,
        "caption": rec.get("caption",""),
        "score": float(rec.get("score", 0.0)),
        "width": int(rec["width"]) if rec.get("width") else None,
        "height": int(rec["height"]) if rec.get("height") else None,
    })
    pbar.update(1)
    return out_path, len(raw)

def iter_csv_rows(csv_path: Path):
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row

# -------------------- 下载主逻辑（含上限控制 & 历史计数） --------------------
async def download_from_csv(csv_path: Path, out_dir: Path):
    dl = Downloader(str(out_dir))

    # 将上限转为字节数；0 表示无限制
    size_cap_bytes = int(MAX_TOTAL_SIZE_GB * (1024**3)) if MAX_TOTAL_SIZE_GB and MAX_TOTAL_SIZE_GB > 0 else None
    count_cap = MAX_DOWNLOAD_IMAGES if MAX_DOWNLOAD_IMAGES and MAX_DOWNLOAD_IMAGES > 0 else None

    # 历史计数并入上限
    existing_count, existing_bytes = dl.preload_existing_stats()
    total_bytes = existing_bytes
    total_ok = existing_count

    rows = list(iter_csv_rows(csv_path))
    planned_total = len(rows) if MAX_IMAGES == 0 else min(len(rows), MAX_IMAGES)

    # 若一开始就命中上限，直接返回
    if (count_cap is not None and total_ok >= count_cap) or (size_cap_bytes is not None and total_bytes >= size_cap_bytes):
        print("\n已达到上限，无需继续下载。")
        print(f"已存在：{total_ok} 张，累计体积：{sizeof_fmt(total_bytes)}")
        return

    connector = aiohttp.TCPConnector(limit=NUM_WORKERS, ttl_dns_cache=300, enable_cleanup_closed=True)
    timeout = ClientTimeout(total=None)

    stop_event = asyncio.Event()

    async with aiohttp.ClientSession(connector=connector, timeout=timeout, headers=DEFAULT_HEADERS) as session:
        pbar = tqdm(total=planned_total, desc="Downloading images", unit="img")
        sem = asyncio.Semaphore(NUM_WORKERS)
        tasks: List[asyncio.Task] = []

        async def worker(r):
            nonlocal total_bytes, total_ok
            if stop_event.is_set():
                pbar.update(1)
                return
            async with sem:
                if stop_event.is_set():
                    pbar.update(1)
                    return
                path, saved = await download_one(session, dl, r, pbar)
                if path is not None:
                    total_ok += 1
                    total_bytes += saved
                # 命中任一上限则触发停止
                hit_count_cap = (count_cap is not None and total_ok >= count_cap)
                hit_size_cap = (size_cap_bytes is not None and total_bytes >= size_cap_bytes)
                if hit_count_cap or hit_size_cap:
                    stop_event.set()

        for i, r in enumerate(rows):
            if MAX_IMAGES > 0 and i >= MAX_IMAGES:
                break
            if stop_event.is_set():
                break
            tasks.append(asyncio.create_task(worker(r)))
            # 批量回收，避免堆积
            if len(tasks) >= NUM_WORKERS * 4:
                await asyncio.gather(*tasks)
                tasks = []
        if tasks:
            await asyncio.gather(*tasks)
        pbar.close()

    print("\n—— 下载阶段结束 ——")
    print(f"成功下载（含历史）：{total_ok} 张")
    print(f"累计体积（含历史）：{sizeof_fmt(total_bytes)}")
    if count_cap:
        print(f"张数上限：{count_cap}（命中即提前停止）")
    if size_cap_bytes:
        print(f"体积上限：{MAX_TOTAL_SIZE_GB} GB（命中即提前停止）")

# -------------------- 主程序 --------------------
async def main():
    out_dir = Path(OUT_DIR)
    safe_mkdirs(out_dir)

    # 1) 先写 CSV
    total_rows, csv_path = write_index_csv(out_dir)

    # 2) 估算整批大小
    if total_rows == 0:
        print("没有符合阈值的样本，跳过估算与下载。")
        return

    N, K, est_bytes = await estimate_total_size(csv_path, ESTIMATE_SAMPLE_SIZE)
    print("\n===== 预计下载体积（粗略估算）=====")
    if K == 0:
        print("未能获取任何 Content-Length，无法估算。")
    else:
        avg = est_bytes / N if N > 0 else 0
        print(f"样本总数 N = {N}；抽样成功 K = {K}/{min(ESTIMATE_SAMPLE_SIZE, N)}")
        print(f"样本平均大小 ≈ {sizeof_fmt(avg)} / 图")
        print(f"预计总大小 ≈ {sizeof_fmt(est_bytes)}")
        low, high = est_bytes * 0.7, est_bytes * 1.3
        print(f"置信区间（粗略 ±30%）：{sizeof_fmt(low)} ~ {sizeof_fmt(high)}")

    # 3) 可选：继续下载（启用体积/张数上限 & 跳过已下载）
    if START_DOWNLOAD:
        print("\n开始按 CSV 下载图片…")
        await download_from_csv(csv_path, out_dir)
        print(f"\n输出：{OUT_DIR}/images 与 {OUT_DIR}/metadata.jsonl")
    else:
        print("\n已生成 CSV 并完成体积估算。如需下载，将 START_DOWNLOAD 设为 True。")

if __name__ == "__main__":
    """
    直接运行此脚本即可：
      pip install -U datasets pillow aiohttp tqdm huggingface_hub
      huggingface-cli login
      python this_script.py

    特性：
      - 先写 CSV → 抽样估算总体积 → 下载
      - 下载前用 metadata 的 URL 快速跳过；启动时扫描磁盘已有 <sha>.* 文件兜底
      - 上限：MAX_TOTAL_SIZE_GB / MAX_DOWNLOAD_IMAGES 任一命中即停止（包含历史计数）
      - Windows 切换 Selector 事件循环，稳
    """
    asyncio.run(main())
