"""The whole deck: forecasting half, generative half, summary, references."""
from fx import *
from slides_a import A, MASE, NAME, TS as _TS
from slides_b import B, N, FR, PRIV, GN as _GN

U = GN['util']; TF = TS['tft']
best = min(('prophet', 'arimax', 'lstm', 'tft'), key=lambda m: MASE[m])
end = []
end.append(slide('Summary', 'Summary', 'What the notebook measured',
                 'Eight numbers from the session, each one reproducible from the notebook.',
                 '<div class="outlbl">forecasting: 14 days ahead, 182 test days, 6 segments</div>'
                 '<div class="kv">' +
                 tile('seasonal naive, MASE', f"{MASE['naive']:.3f}", 'bad') +
                 tile('ARIMA, MASE', f"{MASE['arima']:.3f}", 'b') +
                 tile(f'best with calendar ({NAME[best]})', f"{MASE[best]:.3f}", 'ok') +
                 tile('TFT P10 to P90 band held', pct(TF['cover']), 'bad' if TF['cover'] < 0.75 else 'ok') +
                 '</div>'
                 f"<div class=\"outlbl\">generative models: {comma(N['tx'])} transactions, {N['fraud_all']} fraud</div>"
                 '<div class="kv">' +
                 tile('PR-AUC, trained on real', f"{U['trtr'][1]:.3f}", 'ok') +
                 tile('PR-AUC, trained on synthetic', f"{U['tstr'][1]:.3f}", 'bad') +
                 tile('synthetic rows equal to a real row', f"{PRIV['exact']}", 'ok') +
                 tile('fraud caught at 10 alerts a day', f"{FR['budget']['10']['gb']['caught']} of {N['fraud_test']}", 'b') +
                 '</div>'
                 '<div class="cols nat"><div class="callout"><b>Forecasting.</b> Score every model against the seasonal naive forecast on held-back days. Put the business calendar in before changing the model. Move to an LSTM or TFT for many related series, interacting inputs, or a quantile, and backtest the quantile.</div>'
                 '<div class="callout"><b>Generative models.</b> A synthetic table is released with three test results: fidelity, utility, privacy. In a fraud pipeline the threshold comes from the alert budget, the split is by time, and generated fraud is for testing systems more than for training models.</div></div>'))

end.append(slide('References', 'References', 'References and the notebook',
                 'Everything on the slides can be rerun from the folder.',
                 '<div class="cols"><div class="colstack"><h2 class="s">Papers</h2><div class="refs">'
                 '<p>Hochreiter and Schmidhuber (1997). Long short-term memory. <i>Neural Computation</i> 9(8).</p>'
                 '<p>Taylor and Letham (2018). Forecasting at scale. <i>The American Statistician</i> 72(1). Prophet.</p>'
                 '<p>Lim, Arik, Loeff and Pfister (2021). Temporal Fusion Transformers for interpretable multi-horizon time series forecasting. <i>International Journal of Forecasting</i> 37(4).</p>'
                 '<p>Hyndman and Koehler (2006). Another look at measures of forecast accuracy. <i>International Journal of Forecasting</i> 22(4). MASE.</p>'
                 '<p>Goodfellow et al. (2014). Generative adversarial nets. <i>NeurIPS</i>.</p>'
                 '<p>Gulrajani et al. (2017). Improved training of Wasserstein GANs. <i>NeurIPS</i>. The gradient penalty.</p>'
                 '<p>Xu et al. (2019). Modeling tabular data using conditional GAN. <i>NeurIPS</i>. CTGAN.</p>'
                 '<p>Abadi et al. (2016). Deep learning with differential privacy. <i>ACM CCS</i>. DP-SGD.</p>'
                 '</div></div><div class="colstack"><h2 class="s">In the folder</h2>' +
                 out("""forecasting_and_generative_models.ipynb
    every model and every number, executed

data/cashflow.csv             6 segments x 1,461 days
data/transactions.csv         100,000 rows
data/synthetic_transactions.csv   the GAN's table

build/make_data.py            the simulator""", 'xs', '') +
                 code("""pip install torch statsmodels prophet scikit-learn \\
            pytorch-forecasting lightning""", 'xs') +
                 '<div class="callout sm">The notebook runs end to end on a CPU. The TFT is the slowest cell.</div>'
                 '<div class="callout good sm"><b>Questions?</b></div>'
                 '</div></div>'))

S = A + B + end
