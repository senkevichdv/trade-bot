import vectorbt_bot as vb

cfg = vb.load_config()
ex = vb.create_spot_exchange(cfg)
df = vb.build_spot_dca_dataframe(ex, symbol="ETH/USDT:USDT", bars_5m=8000, entry_timeframe="15m")

rows = []
for tranche_pct in [0.35, 0.40, 0.50, 0.60]:
    for max_buys in [5, 6, 8, 10]:
        for dca_step in [0.008, 0.01, 0.012, 0.015, 0.02]:
            for tp in [0.012, 0.015, 0.02, 0.025]:
                for partial_tp in [0.004, 0.006, 0.008, 0.012, 0.016, 0.02]:
                    for regime_break in [192, 288, 384, 576]:
                        res = vb.simulate_spot_dca(
                            df=df,
                            budget_usdt=100.0,
                            tranche_pct=tranche_pct,
                            max_buys=max_buys,
                            dca_step_pct=dca_step,
                            tp_pct=tp,
                            partial_tp_pct=partial_tp,
                            regime_break_bars=regime_break,
                            recycle_last_lot=True,
                        )
                        ret = float(res['total_return'])
                        dd = float(res['max_dd'])
                        cycles = int(res['cycles'])
                        wr = (float(res['win_cycles']) / cycles) if cycles else 0.0
                        span_days = (df.index[-1] - df.index[0]).total_seconds() / 86400.0 if len(df) > 1 else 0.0
                        monthly = ((1.0 + ret) ** (30.0 / span_days) - 1.0) if span_days > 0 else 0.0

                        # aggressive but still bounded
                        if dd > 0.20:
                            continue
                        if cycles < 4:
                            continue

                        # prioritize monthly return heavily
                        score = monthly * 4.0 + ret * 1.0 - dd * 0.8 + wr * 0.15
                        rows.append((score, monthly, ret, dd, cycles, wr, tranche_pct, max_buys, dca_step, tp, partial_tp, regime_break, int(res['partial_exits'])))

rows.sort(key=lambda x: x[0], reverse=True)
print('candidates', len(rows))
for i, r in enumerate(rows[:25], 1):
    score, monthly, ret, dd, cycles, wr, tr, mb, step, tp, ptp, rb, pex = r
    print(
        f"#{i} score={score:.4f} monthly={monthly*100:.2f}% ret={ret*100:.2f}% dd={dd*100:.2f}% cycles={cycles} wr={wr*100:.1f}% "
        f"tr={tr:.2f} buys={mb} step={step*100:.2f}% tp={tp*100:.2f}% ptp={ptp*100:.2f}% rb={rb} pex={pex}"
    )
