"""The NLP deck: attention, sentiment, entities, documents, summary, references."""
from common import *
from slides_attn import A
from slides_senti import B
from slides_ner import C
from slides_doc import D

TITLE = 'Advanced NLP for Finance'
RAIL = 'Machine Learning for Finance &#183; Advanced NLP for Finance'
OUT = 'advanced_nlp_for_finance.html'
CSS = '''
.cols3{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.tag.b{background:#125F9E}.tag.s{background:#53666F}
table.d td .tag{margin-right:0}
#sb-body td,#sb-body th{padding:5px 8px}
'''
M = SE['m']; mean = lambda d: sum(d.values()) / len(d)
end = []
end.append(slide('Summary', 'Summary', 'What the notebook measured',
                 'Eight numbers from the session, each one reproducible from the notebook.',
                 '<div class="outlbl">attention and sentiment</div>'
                 '<div class="kv">' +
                 tile('weight from "its" to "bank", one head', f"{AT['weights'][AT['i_bank']]:.3f}", 'b') +
                 tile('sentiment, best model as it comes', f"{max(M['general tweet model, as is']['f1'], M['FinBERT, as is']['f1']):.3f}", 'bad') +
                 tile('sentiment, TF-IDF baseline', f"{M['TF-IDF + logistic regression']['f1']:.3f}", 'b') +
                 tile('sentiment, fine-tuned DistilBERT', f"{M['DistilBERT, fine-tuned']['f1']:.3f}", 'ok') +
                 '</div>'
                 '<div class="outlbl">entities and documents</div>'
                 '<div class="kv">' +
                 tile('NER on narratives, general model, 3 types', f"{NE['m']['general_genre']['narrative']['f1']:.3f}", 'bad') +
                 tile('NER on narratives, fine-tuned, 6 types', f"{NE['m']['ft_genre']['narrative']['f1']:.3f}", 'ok') +
                 tile('loan fields right, rules', pct(mean(IE['m']['rules'])), 'b') +
                 tile('loan fields right, QA model', pct(mean(IE['m']['qa'])), 'ok') +
                 '</div>'
                 '<div class="cols nat"><div class="callout"><b>A model as it comes is a starting point.</b> On sentiment and on entities, the download lost to a model that had seen the bank&#8217;s own kind of text. A few hundred to a few thousand labels changed the result.</div>'
                 '<div class="callout"><b>Measure each stage, then route.</b> Rules where the text has a shape, a model where it does not, a score on every output, and a person for the outputs with low scores.</div></div>'))

end.append(slide('References', 'References', 'References and the notebook',
                 'Everything on the slides can be rerun from the folder.',
                 '<div class="cols"><div class="colstack"><h2 class="s">Papers and models</h2><div class="refs">'
                 '<p>Vaswani et al. (2017). Attention is all you need. <i>NeurIPS</i>.</p>'
                 '<p>Devlin, Chang, Lee and Toutanova (2019). BERT: pre-training of deep bidirectional transformers for language understanding. <i>NAACL</i>.</p>'
                 '<p>Sanh, Debut, Chaumond and Wolf (2019). DistilBERT, a distilled version of BERT. <i>arXiv:1910.01108</i>.</p>'
                 '<p>Araci (2019). FinBERT: financial sentiment analysis with pre-trained language models. <i>arXiv:1908.10063</i>.</p>'
                 '<p>Clark, Khandelwal, Levy and Manning (2019). What does BERT look at? An analysis of BERT&#8217;s attention. <i>BlackboxNLP</i>.</p>'
                 '<p>Jain and Wallace (2019). Attention is not explanation. <i>NAACL</i>.</p>'
                 '<p>Rajpurkar, Jia and Liang (2018). Know what you don&#8217;t know: unanswerable questions for SQuAD. <i>ACL</i>.</p>'
                 '<p>Xu et al. (2020). LayoutLM: pre-training of text and layout for document image understanding. <i>KDD</i>.</p>'
                 '<p>Pfitzmann et al. (2022). DocLayNet. <i>KDD</i>.</p>'
                 '</div></div><div class="colstack"><h2 class="s">In the folder</h2>' +
                 out("""nlp/advanced_nlp_for_finance.ipynb
    every model and every number, executed

nlp/data/ner_train.jsonl, ner_test.jsonl   tagged sentences
nlp/data/pages_train.csv, pages_test.csv   document pages
nlp/data/loans.jsonl                       400 loan agreements

nlp/build/make_data.py                     the simulator""", 'xs', '') +
                 '<div class="callout sm">Headlines: <code class="k">zeroshot/twitter-financial-news-sentiment</code> on the Hugging Face hub, MIT licence. The notebook downloads it.</div>'
                 '<div class="callout sm">The notebook opens in Colab and runs on the free T4 GPU.</div>'
                 '<div class="callout good sm"><b>Questions?</b></div>'
                 '</div></div>'))

S = A + B + C + D + end
