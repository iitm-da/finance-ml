"""The LLM deck: RAG, evaluation and hallucination, prompt engineering, LoRA, summary, references."""
from common import *
from slides_rag import A
from slides_eval import B
from slides_prompt import C
from slides_peft import D

TITLE = 'LLMs in Depth'
RAIL = 'Machine Learning for Finance &#183; LLMs in Depth'
OUT = 'llms_in_depth.html'
CSS = '''
.cols3{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.tag.b{background:#125F9E}.tag.s{background:#53666F}
'''
GEN = RG['gen']; LAD = EV['ladder']; SO = PR['so']; PN = list(SO)
L8 = PF['enc']['LoRA, rank 8']; FU = PF['enc']['full fine-tuning']; LR_ = PF['llm']['LoRA, rank 8, 3,000 labels']
best_det = max(EV['det'], key=lambda k: EV['det'][k]['recall'])
end = []
end.append(slide('Summary', 'Summary', 'What the notebook measured',
                 'Eight numbers from the session, each one reproducible from the notebook.',
                 '<div class="outlbl">retrieval and hallucination</div>'
                 '<div class="kv">' +
                 tile('correct, closed book', pct(GEN['closed book']['ans']['correct'], 0), 'bad') +
                 tile('correct, retrieval + strict prompt', pct(GEN['+ strict prompt']['ans']['correct'], 0), 'ok') +
                 tile('wrong answers, plain RAG', pct(LAD['RAG, dense, top 3']['wrong'], 0), 'bad') +
                 tile('wrong answers, all guards', pct(LAD['+ number check on the output']['wrong'], 0), 'ok') +
                 '</div>'
                 '<div class="outlbl">prompting and fine-tuning</div>'
                 '<div class="kv">' +
                 tile('loan fields right, plain prompt', pct(SO[PN[0]]['all'], 0), 'bad') +
                 tile('loan fields right, full prompt', pct(SO[PN[-1]]['all'], 0), 'ok') +
                 tile('sentiment, zero-shot prompt', f"{PF['prompt']['zero-shot, 1.5B model']:.3f}", 'b') +
                 tile(f"sentiment, LoRA ({L8['trainable'] / L8['total']:.1%} of weights)", f"{L8['f1']:.3f}", 'ok') +
                 '</div>'
                 '<div class="cols nat"><div class="callout"><b>Facts come from documents.</b> Retrieval supplies them, a strict prompt keeps the model to them, and a check outside the model catches what slips through. Each stage has its own score.</div>'
                 '<div class="callout"><b>Skills come from prompts and training.</b> A format, a definition and a chosen example go a long way. With thousands of labels and a fixed task, a small adapter on a small model goes further.</div></div>'))

end.append(slide('References', 'References', 'References and the notebook',
                 'Everything on the slides can be rerun from the folder.',
                 '<div class="cols"><div class="colstack"><h2 class="s">Papers</h2><div class="refs">'
                 '<p>Lewis et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. <i>NeurIPS</i>.</p>'
                 '<p>Robertson and Zaragoza (2009). The probabilistic relevance framework: BM25 and beyond. <i>Foundations and Trends in Information Retrieval</i> 3(4).</p>'
                 '<p>Cormack, Clarke and Buettcher (2009). Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. <i>SIGIR</i>.</p>'
                 '<p>Hu et al. (2022). LoRA: low-rank adaptation of large language models. <i>ICLR</i>.</p>'
                 '<p>Manakul, Liusie and Gales (2023). SelfCheckGPT: zero-resource black-box hallucination detection for generative large language models. <i>EMNLP</i>.</p>'
                 '<p>Wei et al. (2022). Chain-of-thought prompting elicits reasoning in large language models. <i>NeurIPS</i>.</p>'
                 '<p>Liu et al. (2022). What makes good in-context examples for GPT-3? <i>DeeLIO</i>.</p>'
                 '<p>Greshake et al. (2023). Not what you&#8217;ve signed up for: compromising real-world LLM-integrated applications with indirect prompt injection. <i>AISec</i>.</p>'
                 '</div></div><div class="colstack"><h2 class="s">In the folder</h2>' +
                 out("""llm/llms_in_depth.ipynb
    every experiment and every number, executed

llm/data/kb.jsonl          67 sections of a fictional bank
llm/data/questions.jsonl   225 questions with answers

llm/build/make_data.py     writes the knowledge base""", 'xs', '') +
                 '<div class="callout sm">Models: Qwen2.5-1.5B-Instruct and Qwen2.5-0.5B-Instruct (Apache 2.0), bge-small-en-v1.5, ms-marco-MiniLM cross-encoder. All are downloaded by the notebook; no API key is needed.</div>'
                 '<div class="callout sm">The notebook opens in Colab and runs on the free T4 GPU.</div>'
                 '<div class="callout good sm"><b>Questions?</b></div>'
                 '</div></div>'))

S = A + B + C + D + end
