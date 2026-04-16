import argparse
import asyncio
import faulthandler
import json
import os
import random
import sys
import threading


def _arm_hard_timeout(timeout_s: float) -> callable:
    if timeout_s is None or timeout_s <= 0:
        return lambda: None

    faulthandler.enable(all_threads=True)

    dump_timer = threading.Timer(
        timeout_s, lambda: faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
    )
    dump_timer.daemon = True
    dump_timer.start()

    kill_timer = threading.Timer(timeout_s + 2.0, lambda: os._exit(124))  # noqa: SLF001
    kill_timer.daemon = True
    kill_timer.start()

    def _cancel() -> None:
        dump_timer.cancel()
        kill_timer.cancel()

    return _cancel


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return float(xs[f])
    return float(xs[f] + (k - f) * (xs[c] - xs[f]))


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"avg": 0.0, "p90": 0.0, "p99": 0.0}
    return {
        "avg": float(sum(values) / max(1, len(values))),
        "p90": float(_percentile(values, 90)),
        "p99": float(_percentile(values, 99)),
    }


def _extract_timings(ro) -> tuple[float | None, float | None, float | None]:
    metrics = getattr(ro, "metrics", None)
    if metrics is None:
        return None, None, None

    ttft_s = None
    itl_mean_s = None
    e2e_s = None

    first_token_latency = getattr(metrics, "first_token_latency", None)
    if first_token_latency is not None and float(first_token_latency) > 0:
        ttft_s = float(first_token_latency)
    else:
        arrival_time = getattr(metrics, "arrival_time", None)
        first_token_time = getattr(metrics, "first_token_time", None)
        if arrival_time is not None and first_token_time is not None:
            ttft_s = float(first_token_time - arrival_time)

    arrival_time = getattr(metrics, "arrival_time", None)
    finished_time = getattr(metrics, "finished_time", None)
    if arrival_time is not None and finished_time is not None:
        e2e_s = float(finished_time - arrival_time)
    elif arrival_time is not None:
        last_token_time = getattr(metrics, "last_token_time", None)
        if last_token_time is not None:
            e2e_s = float(last_token_time - arrival_time)

    last_token_time = getattr(metrics, "last_token_time", None)
    first_token_time = getattr(metrics, "first_token_time", None)
    if last_token_time is not None and first_token_time is not None:
        itl_mean_s = float(last_token_time - first_token_time)

    first_token_ts = getattr(metrics, "first_token_ts", None)
    last_token_ts = getattr(metrics, "last_token_ts", None)
    if itl_mean_s is None and first_token_ts and last_token_ts:
        itl_mean_s = float(last_token_ts - first_token_ts)

    if e2e_s is None and ttft_s is not None and itl_mean_s is not None:
        e2e_s = float(ttft_s + itl_mean_s)

    return ttft_s, itl_mean_s, e2e_s


def _build_tokenizer(model: str, trust_remote_code: bool):
    from vllm.transformers_utils.tokenizer import get_tokenizer

    return get_tokenizer(
        model,
        tokenizer_mode="auto",
        trust_remote_code=bool(trust_remote_code),
    )


def _sample_random_token_ids(
    rng: random.Random, tokenizer, token_count: int
) -> list[int]:
    if token_count <= 0:
        return []
    vocab_size = int(len(tokenizer))
    if vocab_size <= 0:
        return []
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    token_ids: list[int] = []
    while len(token_ids) < token_count:
        tid = rng.randrange(vocab_size)
        if tid in special_ids:
            continue
        token_ids.append(int(tid))
    return token_ids


def _make_token_prompt(token_ids: list[int]) -> dict:
    return {"prompt_token_ids": list(token_ids)}


async def _generate_one(
    engine,
    request_id: str,
    prompt,
    sampling_params,
    timeout_s: float,
):
    async def _collect_final():
        final_out = None
        async for out in engine.generate(
            request_id=request_id, prompt=prompt, sampling_params=sampling_params
        ):
            final_out = out
        return final_out

    cancel_timeout = _arm_hard_timeout(float(timeout_s))
    try:
        return await asyncio.wait_for(_collect_final(), timeout=float(timeout_s))
    finally:
        cancel_timeout()


async def _run_stage(
    *,
    engine,
    prompts: list,
    sampling_params,
    concurrency: int,
    timeout_s: float,
    request_id_prefix: str,
) -> list:
    if not prompts:
        return []
    concurrency = max(1, int(concurrency))
    results: list = [None] * len(prompts)

    async def _run_one(i: int) -> None:
        results[i] = await _generate_one(
            engine,
            request_id=f"{request_id_prefix}{i}",
            prompt=prompts[i],
            sampling_params=sampling_params,
            timeout_s=timeout_s,
        )

    next_i = 0
    pending: set[asyncio.Task] = set()
    while next_i < len(prompts) and len(pending) < concurrency:
        pending.add(asyncio.create_task(_run_one(next_i)))
        next_i += 1

    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_EXCEPTION)
        for t in done:
            exc = t.exception()
            if exc is not None:
                for p in pending:
                    p.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                raise exc
        while next_i < len(prompts) and len(pending) < concurrency:
            pending.add(asyncio.create_task(_run_one(next_i)))
            next_i += 1

    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--warmup-prompt-tokens", type=int, default=0)
    ap.add_argument("--gen-tokens", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--warmup-batch-size", type=int, default=1)
    ap.add_argument("--num-requests", type=int, default=0)
    ap.add_argument("--warmup-prefix-rate", type=int, default=0)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-num-seqs", type=int, default=0)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    ap.add_argument("--enforce-eager", action="store_true", default=False)
    ap.add_argument(
        "--enable-prefix-caching",
        action="store_true",
        default=True,
        help="启用 vLLM prefix caching（默认开启）。",
    )
    ap.add_argument(
        "--disable-prefix-caching",
        action="store_true",
        default=False,
        help="强制关闭 vLLM prefix caching（用于 OOM 对比与公平性控制）。",
    )
    ap.add_argument("--enable-sparse", action="store_true", default=False)
    ap.add_argument(
        "--enable-cuda-topk",
        action="store_true",
        default=False,
    )
    ap.add_argument(
        "--topk-type",
        type=str,
        default="block-mean",
        choices=("block-mean", "paged-kmeans"),
        help="BMSA TopK 类型：block-mean（默认）或 paged-kmeans（KMeans 聚类 TopK）。",
    )
    ap.add_argument(
        "--kmeans-centroid-topk-ratio",
        type=float,
        default=0.20,
        help="paged-kmeans：decode 阶段对 centroid 做 TopK 的比例（默认 0.20）。",
    )
    ap.add_argument(
        "--kmeans-topk-backend",
        type=str,
        default="cutlass",
        choices=("cutlass", "torch"),
        help="paged-kmeans：centroid TopK 后端实现（默认 cutlass）。",
    )
    ap.add_argument(
        "--enable-prefetch",
        action="store_true",
        default=None,
    )
    ap.add_argument("--disable-prefetch", action="store_true", default=False)
    ap.add_argument("--kv-store-capacity", type=int, default=2048)
    ap.add_argument(
        "--kv-transfer-config-json",
        type=str,
        default=None,
    )
    ap.add_argument("--lc-sparse-threshold", type=int, default=512)
    ap.add_argument("--total-budget", type=float, default=0.15)
    ap.add_argument(
        "--topk-update-interval",
        type=int,
        default=3,
        help="TopK 更新间隔（step 数）。生产建议保持 3。",
    )
    ap.add_argument(
        "--timeout-s",
        type=float,
        default=300.0,
        help="单次 generate 超时阈值：超时会 dump 栈并强制退出。",
    )
    ap.add_argument(
        "--result-type",
        type=str,
        default="text",
        choices=("json", "text"),
    )
    args = ap.parse_args()

    use_prefix_caching = bool(args.enable_prefix_caching) and not bool(
        args.disable_prefix_caching
    )

    sparse_cfg = None
    kv_transfer_cfg = None
    if args.enable_sparse:
        try:
            from vllm.config import BMSAConfig, SparseConfig
        except ImportError:
            from wings_engine_patch.patch_vllm_container.v0_17_0.sparse_kv_config import (
                BMSAConfig, SparseConfig,
            )

        if args.disable_prefetch:
            prefetch_enabled = False
        elif args.enable_prefetch is None:
            prefetch_enabled = True
        else:
            prefetch_enabled = bool(args.enable_prefetch)

        bmsa_algo_cfg = BMSAConfig()
        bmsa_algo_cfg.ptopk_prefetch_enable = bool(prefetch_enabled)
        bmsa_algo_cfg.enable_cuda_topk = bool(args.enable_cuda_topk)
        bmsa_algo_cfg.topk_update_interval = max(1, int(args.topk_update_interval))
        bmsa_algo_cfg.topk_type = str(args.topk_type)
        bmsa_algo_cfg.kmeans_centroid_topk_ratio = float(args.kmeans_centroid_topk_ratio)
        bmsa_algo_cfg.kmeans_topk_backend = str(args.kmeans_topk_backend)

        sparse_cfg = SparseConfig(
            enable_sparse=True,
            sparse_algo_type="BMSA",
            lc_sparse_threshold=int(args.lc_sparse_threshold),
            total_budget=float(args.total_budget),
            sparse_algo_config=bmsa_algo_cfg,
        )

        if prefetch_enabled:
            if args.kv_transfer_config_json is not None:
                kv_transfer_cfg = json.loads(args.kv_transfer_config_json)
            else:
                connector_name = "LocalStoreKVStore"
                connector_config: dict[str, object] = {
                    "capacity": int(args.kv_store_capacity),
                }
                kv_transfer_cfg = {
                    "kv_connector": "SparseConnector",
                    "kv_connector_module_path":
                    "vllm.v1.sparse.connectors.sparse_connector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {
                        "sparse_connectors": [
                            {
                                "connector_name": connector_name,
                                "connector_config": connector_config,
                            }
                        ]
                    },
                }

    warmup_batch_size = max(0, int(args.warmup_batch_size))
    batch_size = max(1, int(args.batch_size))
    target_num_requests = int(args.num_requests) if int(args.num_requests) > 0 else batch_size
    warmup_prompt_tokens = int(args.warmup_prompt_tokens) if int(
        args.warmup_prompt_tokens) > 0 else int(args.prompt_tokens)
    prompt_tokens = int(args.prompt_tokens)
    gen_tokens = int(args.gen_tokens)

    max_prompt_len = max(prompt_tokens, warmup_prompt_tokens if warmup_batch_size > 0 else prompt_tokens)
    if max_prompt_len + gen_tokens > int(args.max_model_len):
        raise ValueError(
            f"(max(prompt_tokens,warmup_prompt_tokens)+gen_tokens)={max_prompt_len}+{gen_tokens} "
            f"exceeds --max-model-len={int(args.max_model_len)}"
        )

    max_num_seqs = int(args.max_num_seqs)
    if max_num_seqs <= 0:
        max_num_seqs = max(batch_size, warmup_batch_size if warmup_batch_size > 0 else 1)
    if max_num_seqs < max(batch_size, warmup_batch_size if warmup_batch_size > 0 else 1):
        raise ValueError(
            f"--max-num-seqs({max_num_seqs}) must be >= max(batch-size, warmup-batch-size)"
        )

    tokenizer = _build_tokenizer(args.model, trust_remote_code=True)
    rng = random.Random()

    warmup_prefix_rate = max(0, min(100, int(args.warmup_prefix_rate)))
    shared_prefix_len = int(min(prompt_tokens, warmup_prompt_tokens) * warmup_prefix_rate / 100)

    warmup_prefixes: list[list[int]] = []
    if warmup_batch_size > 0 and shared_prefix_len > 0:
        for _ in range(warmup_batch_size):
            warmup_prefixes.append(
                _sample_random_token_ids(rng, tokenizer, shared_prefix_len)
            )

    warmup_prompt_ids: list[list[int]] = []
    if warmup_batch_size > 0:
        for i in range(warmup_batch_size):
            base = _sample_random_token_ids(rng, tokenizer, warmup_prompt_tokens)
            if shared_prefix_len > 0 and warmup_prefixes:
                base[:shared_prefix_len] = warmup_prefixes[i % len(warmup_prefixes)]
            warmup_prompt_ids.append(base)

    target_prompt_ids: list[list[int]] = []
    for i in range(target_num_requests):
        base = _sample_random_token_ids(rng, tokenizer, prompt_tokens)
        if shared_prefix_len > 0 and warmup_prefixes:
            base[:shared_prefix_len] = warmup_prefixes[i % len(warmup_prefixes)]
        target_prompt_ids.append(base)

    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm import SamplingParams

    engine_args = AsyncEngineArgs(
        model=args.model,
        trust_remote_code=True,
        max_model_len=int(args.max_model_len),
        max_num_seqs=int(max_num_seqs),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        enforce_eager=bool(args.enforce_eager),
        sparse_config=sparse_cfg,
        kv_transfer_config=kv_transfer_cfg,
        swap_space=0,
        enable_prefix_caching=bool(use_prefix_caching),
        disable_log_stats=False,
    )

    cancel_init_timeout = _arm_hard_timeout(float(args.timeout_s))
    try:
        engine = AsyncLLM.from_engine_args(engine_args)
    finally:
        cancel_init_timeout()

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=int(gen_tokens),
        min_tokens=int(gen_tokens),
        ignore_eos=True,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )

    warmup_outputs: list = []
    target_outputs: list = []

    async def _run_all():
        w_outs: list = []
        t_outs: list = []
        if warmup_prompt_ids:
            w_outs = await _run_stage(
                engine=engine,
                prompts=[_make_token_prompt(x) for x in warmup_prompt_ids],
                sampling_params=sampling_params,
                concurrency=warmup_batch_size,
                timeout_s=float(args.timeout_s),
                request_id_prefix="warmup-",
            )
        t_outs = await _run_stage(
            engine=engine,
            prompts=[_make_token_prompt(x) for x in target_prompt_ids],
            sampling_params=sampling_params,
            concurrency=batch_size,
            timeout_s=float(args.timeout_s),
            request_id_prefix="target-",
        )
        return w_outs, t_outs

    try:
        warmup_outputs, target_outputs = asyncio.run(_run_all())
    finally:
        engine.shutdown()

    def _stage_metrics(outs: list) -> dict:
        ttft_s: list[float] = []
        itl_s: list[float] = []
        e2e_s: list[float] = []
        tpot_s: list[float] = []
        total_tps: list[float] = []
        decode_tps: list[float] = []
        prompt_toks: list[int] = []
        out_toks: list[int] = []

        for ro in outs:
            if ro is None:
                continue
            prompt_len = len(ro.prompt_token_ids) if getattr(ro, "prompt_token_ids", None) else 0
            out_len = sum(len(o.token_ids) for o in ro.outputs if o)
            prompt_toks.append(int(prompt_len))
            out_toks.append(int(out_len))

            ttft, itl_span, e2e = _extract_timings(ro)
            if ttft is not None:
                ttft_s.append(float(ttft))
            if e2e is not None:
                e2e_s.append(float(e2e))
            if itl_span is not None and out_len > 1:
                itl_s.append(float(itl_span) / float(out_len - 1))
            if ttft is not None and e2e is not None and out_len > 0:
                tpot_s.append(float(e2e - ttft) / float(out_len))
            if e2e is not None and float(e2e) > 0:
                total_tps.append(float(prompt_len + out_len) / float(e2e))
            if ttft is not None and e2e is not None and float(e2e) > float(ttft) and out_len > 0:
                decode_tps.append(float(out_len) / float(e2e - ttft))

        return {
            "count": int(len([x for x in outs if x is not None])),
            "prompt_tokens_avg": float(sum(prompt_toks) / max(1, len(prompt_toks))) if prompt_toks else 0.0,
            "output_tokens_avg": float(sum(out_toks) / max(1, len(out_toks))) if out_toks else 0.0,
            "ttft_s": _summarize(ttft_s),
            "itl_s": _summarize(itl_s),
            "tpot_s": _summarize(tpot_s),
            "e2e_s": _summarize(e2e_s),
            "total_tps": _summarize(total_tps),
            "decode_tps": _summarize(decode_tps),
        }

    warmup_summary = _stage_metrics(warmup_outputs)
    target_summary = _stage_metrics(target_outputs)

    output_preview = ""
    output_preview_tokens = 0
    output_tokens_first = 0
    prompt_tokens_first = 0
    if target_outputs and target_outputs[0] is not None and target_outputs[0].outputs:
        first_out = target_outputs[0].outputs[0]
        prompt_tokens_first = (
            len(target_outputs[0].prompt_token_ids)
            if getattr(target_outputs[0], "prompt_token_ids", None)
            else 0
        )
        token_ids = list(getattr(first_out, "token_ids", []) or [])
        output_tokens_first = int(len(token_ids))
        output_preview_tokens = min(32, len(token_ids))
        if output_preview_tokens > 0:
            output_preview = tokenizer.decode(token_ids[:output_preview_tokens])
        else:
            output_preview = str(getattr(first_out, "text", "") or "")

    mode = "BMSA" if bool(args.enable_sparse) else "DENSE"
    prefetch_enabled = bool(
        sparse_cfg is not None
        and sparse_cfg.sparse_algo_config is not None
        and getattr(sparse_cfg.sparse_algo_config, "ptopk_prefetch_enable", False)
    )

    res = {
        "mode": mode,
        "prefetch_enabled": bool(prefetch_enabled),
        "prefix_caching_enabled": bool(use_prefix_caching),
        "prompt_tokens": int(prompt_tokens),
        "warmup_prompt_tokens": int(warmup_prompt_tokens),
        "gen_tokens": int(args.gen_tokens),
        "batch_size": int(batch_size),
        "warmup_batch_size": int(warmup_batch_size),
        "num_requests": int(target_num_requests),
        "warmup_prefix_rate": int(warmup_prefix_rate),
        "prompt_tokens_first": int(prompt_tokens_first),
        "output_tokens_first": int(output_tokens_first),
        "lc_sparse_threshold": int(args.lc_sparse_threshold) if sparse_cfg else None,
        "total_budget": float(args.total_budget) if sparse_cfg else None,
        "topk_update_interval": int(args.topk_update_interval) if sparse_cfg else None,
        "enable_cuda_topk": bool(args.enable_cuda_topk) if sparse_cfg else None,
        "topk_type": str(args.topk_type) if sparse_cfg else None,
        "kmeans_centroid_topk_ratio": float(args.kmeans_centroid_topk_ratio)
        if sparse_cfg and str(args.topk_type) == "paged-kmeans"
        else None,
        "kmeans_topk_backend": str(args.kmeans_topk_backend)
        if sparse_cfg and str(args.topk_type) == "paged-kmeans"
        else None,
        "max_model_len": int(args.max_model_len),
        "max_num_seqs": int(max_num_seqs),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "timeout_s": float(args.timeout_s),
        "output_preview_first32tokens": output_preview,
        "warmup": warmup_summary,
        "target": target_summary,
    }

    if str(args.result_type) == "json":
        print(json.dumps(res, ensure_ascii=False))
        return

    def _fmt_stage(name: str, summary: dict) -> str:
        ttft = summary['ttft_s']
        itl = summary['itl_s']
        tpot = summary['tpot_s']
        e2e = summary['e2e_s']
        total_tps = summary['total_tps']
        decode_tps = summary['decode_tps']
        count = summary['count']
        pt_avg = summary['prompt_tokens_avg']
        ot_avg = summary['output_tokens_avg']
        return "\n".join(
            [
                f"{name}: count={count},"
                f" prompt_tokens_avg={pt_avg:.2f},"
                f" output_tokens_avg={ot_avg:.2f}",
                f"  TTFT(s):  avg={ttft['avg']:.6f},"
                f" p90={ttft['p90']:.6f}, p99={ttft['p99']:.6f}",
                f"  ITL(s):   avg={itl['avg']:.6f},"
                f" p90={itl['p90']:.6f}, p99={itl['p99']:.6f}",
                f"  TPOT(s):  avg={tpot['avg']:.6f},"
                f" p90={tpot['p90']:.6f}, p99={tpot['p99']:.6f}",
                f"  E2E(s):   avg={e2e['avg']:.6f},"
                f" p90={e2e['p90']:.6f}, p99={e2e['p99']:.6f}",
                f"  Total TPS(tok/s):  avg={total_tps['avg']:.3f},"
                f" p90={total_tps['p90']:.3f}, p99={total_tps['p99']:.3f}",
                f"  Decode TPS(tok/s): avg={decode_tps['avg']:.3f},"
                f" p90={decode_tps['p90']:.3f}, p99={decode_tps['p99']:.3f}",
            ]
        )

    lines = [
        f"mode={mode}, prefetch={prefetch_enabled},"
        f" prefix_caching={use_prefix_caching}",
        f"prompt_tokens={prompt_tokens},"
        f" warmup_prompt_tokens={warmup_prompt_tokens},"
        f" gen_tokens={int(args.gen_tokens)}",
        f"batch_size={batch_size}, warmup_batch_size={warmup_batch_size},"
        f" num_requests={target_num_requests},"
        f" warmup_prefix_rate={warmup_prefix_rate}",
    ]
    if sparse_cfg is not None:
        lines.append(
            f"lc_sparse_threshold={int(args.lc_sparse_threshold)},"
            f" total_budget={float(args.total_budget)},"
            f" topk_update_interval={int(args.topk_update_interval)},"
            f" enable_cuda_topk={bool(args.enable_cuda_topk)},"
            f" topk_type={str(args.topk_type)}"
        )
        if str(args.topk_type) == "paged-kmeans":
            lines.append(
                f"kmeans_centroid_topk_ratio={float(args.kmeans_centroid_topk_ratio)},"
                f" kmeans_topk_backend={str(args.kmeans_topk_backend)}"
            )
    lines.append(f"prompt_tokens_first={prompt_tokens_first}, output_tokens_first={output_tokens_first}")
    lines.append(f"output_preview_first32tokens={output_preview}")
    lines.append(_fmt_stage("warmup", warmup_summary))
    lines.append(_fmt_stage("target", target_summary))
    print("\n".join(lines))


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        out = {
            "success": False,
            "error": repr(e),
        }
        print(json.dumps(out, ensure_ascii=False))
        raise
