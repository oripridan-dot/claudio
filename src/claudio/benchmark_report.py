"""
benchmark_report.py — Summary Report for Real-Time Benchmark

Extracted from realtime_benchmark.py for 300-line compliance.
Handles the final scorecard display. The test functions remain in
realtime_benchmark.py.
"""

from __future__ import annotations


def _pass_fail(ok: bool) -> str:
    return "✅ PASS" if ok else "❌ FAIL"


def print_scorecard(r1: list[dict], r2: list[dict], r3: dict, r4: dict, r5: dict) -> None:
    """Print the final summary scorecard from all benchmark tests."""
    print("\n" + "═" * 70)
    print("  FINAL SCORE CARD")
    print("═" * 70)

    all_results = []

    # Test 1 summary
    for r in r1:
        label = f"T1 Latency [{r['config']}]"
        print(f"  {_pass_fail(r['passed'])}  {label:45s}  {r['mean_us']:.0f}µs / {r['total_latency_ms']:.2f}ms")
        all_results.append(r["passed"])

    # Test 2 summary
    max_rt_sources = 0
    for r in r2:
        if r["passed"]:
            max_rt_sources = r["sources"]
    label_t2 = "T2 Multi-Source Ceiling"
    print(
        f"  {'✅ PASS' if max_rt_sources >= 16 else '❌ FAIL'}  {label_t2:45s}  {max_rt_sources} sources in real-time"
    )
    all_results.append(max_rt_sources >= 16)

    # Test 3
    label_t3 = "T3 Head-Track Storm (120Hz + 4 sources)"
    print(f"  {_pass_fail(r3['passed'])}  {label_t3:45s}  {r3['mean_us']:.0f}µs mean")
    all_results.append(r3["passed"])

    # Test 4
    label_t4 = "T4 Audio Fidelity"
    print(f"  {_pass_fail(r4['passed'])}  {label_t4:45s}  SNR={r4['snr_db']:.1f}dB THD={r4['thdn_pct']:.3f}%")
    all_results.append(r4["passed"])

    # Test 5
    label_t5 = "T5 Sustained Load (10s)"
    print(f"  {_pass_fail(r5['passed'])}  {label_t5:45s}  {r5['rt_factor']:.0f}× RT, {r5['overruns']} overruns")
    all_results.append(r5["passed"])

    total_pass = sum(all_results)
    total = len(all_results)
    all_pass = all(all_results)

    print(f"\n{'═' * 70}")
    if all_pass:
        print(f"  🏆 ALL {total}/{total} TESTS PASSED — REAL-TIME CERTIFIED")
    else:
        print(f"  ⚠️  {total_pass}/{total} TESTS PASSED")
    print(f"{'═' * 70}")
