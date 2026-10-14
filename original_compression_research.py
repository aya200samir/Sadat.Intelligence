import re,json,hashlib,random,time,collections,os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,classification_report
from sklearn.model_selection import GroupShuffleSplit
import joblib
ROOT=Path('/mnt/data'); lines=[s.strip() for s in (ROOT/'training_8000_raw_complaints.txt').read_text(encoding='utf8').splitlines() if s.strip()]
# Keep only conservative introductory/closing templates; no paraphrased claims.
PREFIX=[r'^أنا لاحظت إن\s*',r'^لما رجعت أتأكد لقيت إن\s*',r'^كنت بتابع الموضوع ولقيت إن\s*',r'^اللي حصل معايا إن\s*',r'^محتاج توضيح بخصوص الآتي:\s*',r'^عندي مشكلة وعايز حل:\s*',r'^عايز أعرف أتصرف إزاي[،,:]\s*',r'^عايز أعرف سبب إن\s*',r'^ممكن توضحوا موضوع إن\s*',r'^أنا مش فاهم ليه\s*',r'^ممكن أفهم ليه\s*',r'^هو المفروض أعمل إيه لو\s*',r'^هو ليه\s*',r'^بستفسر عن حاجة[،,:]\s*',r'^كنت عايز أسأل:\s*',r'^أنا محتاج مساعدة[،,:]\s*',r'^ممكن حد يفهمني[،,:]\s*',r'^يا جماعة\s*',r'^لو سمحت[،,:]?\s*',r'^من فضلكم[،,:]?\s*',r'^بعد إذنكم[،,:]?\s*',r'^برجاء الإفادة[،,:]?\s*',r'^برجاء المساعدة[،,:]?\s*',r'^أرجو المساعدة[،,:]?\s*',r'^أود الاستفسار[،,:]?\s*',r'^السلام عليكم[،,:]?\s*',r'^معلش[،,:]?\s*']
# These are conversational afterthoughts; only remove as separate sentences.
CLOSINGS=['ممكن مساعدة','أرجو مراجعة الموضوع','يا ريت تراجعوا ده','محتاج أعرف الخطوة الجاية','محتاج الموضوع يتراجع','محتاج رد لو سمحتم','أعمل إيه دلوقتي','أتصرف إزاي','ممكن توضيح','هل فيه حل متاح','ممكن أعرف الحل','مش عارف أتصرف إزاي','ومش عارف أبدأ منين','وأتمنى مراجعة الأمر','وأرجو توضيح الإجراء المناسب','ممكن تساعدوني أفهم السبب','يا ريت حد يوضح لي الخطوة الجاية','أتواصل مع مين بخصوص ده','هل في طريقة أتابع بيها','عايز أتأكد إن الموضوع اتسجل']
FILLERS=['ما عنديش معلومات زيادة غير اللي ظاهر قدامي','أنا مش متأكد إذا الخطأ من عندي ولا من النظام','مش عايز أعمل خطوة غلط وأضيع وقت','سألت قبل كده ولسه ما وصلت لرد واضح','جربت أستفسر بس الردود كانت مختلفة','كلمت حد وقال لي أتابع بعدين من غير تفاصيل','حاولت أراجع بنفسي لكن ما فهمتش اللي حصل','دخلت على حسابي تاني ونفس الوضع موجود','يهمني أعرف هل لازم أقدم طلب جديد','الموضوع مقلقني لأن عندي إجراء تاني مرتبط بيه','وده معطلني في اللي ورايا']
CONTEXT=['من ساعة ما قدمت الطلب','بعد ما راجعت الصفحة كذا مرة','بعد ما رجعت أراجع البيانات','بعد ما سألت في المكان اللي اتقال لي عليه','لما دخلت أتأكد بنفسي','بعد آخر تحديث ظهر عندي','رغم إني حاولت أكتر من مرة','من وقت ما لاحظت المشكلة','من الأسبوع اللي فات','من أول الشهر','من يومين','امبارح','النهارده','من فترة']

def derive(s):
    t=s.strip(); flags=[]
    for p in PREFIX:
        n=re.sub(p,'',t,count=1)
        if n!=t:t=n;flags.append('prefix');break
    # remove filler independent sentences only, avoiding semantic corruption
    for f in FILLERS:
        t=re.sub(r'(?:[.؟!]\s*|،\s*)'+re.escape(f)+r'(?=[.؟!،]|$)','',t)
    # strip known closing clauses at end, not inside core
    for c in CLOSINGS:
        t=re.sub(r'(?:[،.؟!]\s*|\s+)'+re.escape(c)+r'[؟.!\s]*$','',t)
    # Remove context only when end clause; time is a useful fact, so preserve it
    t=re.sub(r'\s+',' ',t).strip(' ،.؟!:')
    if re.search(r'(?:مش عارف|مش قادر|ومحتاج|ومش|وعايز|لكن|بس)$',t): return s.strip(), 'unchanged_unsafe'
    if not t or len(t)<12: return s.strip(), 'unchanged_short'
    if len(t)>len(s):return s.strip(),'unchanged'
    return t,('derived' if t!=s.strip(' ،.؟!:') else 'unchanged')

rows=[]
for i,s in enumerate(lines):
    summary,status=derive(s)
    rows.append(dict(source_row=i+1,complaint=s,silver_summary=summary,derivation=status,source='uploaded_raw',label_quality='automatic_silver_not_human'))
df=pd.DataFrame(rows)
# group on derived summary normalized to protect exact canonical cores from cross-split leakage
norm=lambda s:re.sub(r'\s+',' ',re.sub(r'[،؟!.,:؛]','',s)).strip()
df['group_key']=df.silver_summary.map(norm)
splitter=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42)
train_idx,test_idx=next(splitter.split(df,groups=df.group_key))
# second group split for validation
trainpart=df.iloc[train_idx]; sp=GroupShuffleSplit(n_splits=1,test_size=.125,random_state=42)
a,b=next(sp.split(trainpart,groups=trainpart.group_key)); train_ids=trainpart.index[a]; val_ids=trainpart.index[b]
df['split']='test';df.loc[train_ids,'split']='train';df.loc[val_ids,'split']='validation'
# Train a CPU classifier to learn which candidate of {raw, automatically shortened} should be returned.
# This is a lightweight supervised compression decision model, not generative summarization.
# To avoid trivial perfect label leakage, inputs are original complaint only, target whether reduction is possible.
y=(df.silver_summary.str.len()<df.complaint.str.len()-3).astype(int)
vec=TfidfVectorizer(analyzer='char',ngram_range=(2,4),min_df=3,max_features=55000,sublinear_tf=True,dtype=np.float32)
Xtr=vec.fit_transform(df.loc[train_ids,'complaint']); model=LogisticRegression(max_iter=250,random_state=42)
t0=time.perf_counter();model.fit(Xtr,y.loc[train_ids]);fit_seconds=time.perf_counter()-t0
metrics={}
for split,ids in [('train',train_ids),('validation',val_ids),('test',test_idx)]:
    yy=y.loc[ids];pred=model.predict(vec.transform(df.loc[ids,'complaint']))
    metrics[split]={'n':len(ids),'accuracy_vs_silver':round(float(accuracy_score(yy,pred)),4),'macro_f1_vs_silver':round(float(f1_score(yy,pred,average='macro',zero_division=0)),4)}
# model decides to use deterministic candidate; if no -> raw
pred=model.predict(vec.transform(df.complaint))
df['model_summary']=[r.silver_summary if p else r.complaint for r,p in zip(df.itertuples(),pred)]
df['original_chars']=df.complaint.str.len();df['summary_chars']=df.model_summary.str.len()
df['compression_fraction']=1-df.summary_chars/df.original_chars
out=ROOT/'sadat_summarization_cpu';out.mkdir(exist_ok=True)
df.to_csv(out/'derived_silver_dataset.csv',index=False,encoding='utf-8-sig')
joblib.dump({'vectorizer':vec,'model':model},out/'compression_decision_model.joblib')
# Human review required; balanced manual queue for qualitative review
rng=random.Random(2026);review=[]
for name,g in df.groupby('derivation'):
    review.extend(g.sample(n=min(15,len(g)),random_state=2026).to_dict('records'))
pd.DataFrame(review).to_csv(out/'human_review_queue.csv',index=False,encoding='utf-8-sig')
report={'raw_count':len(df),'unique_raw':df.complaint.nunique(),'unique_silver_summaries':df.silver_summary.nunique(),'group_count':df.group_key.nunique(),'derivation_counts':df.derivation.value_counts().to_dict(),'splits':df.split.value_counts().to_dict(),'avg_original_chars':round(float(df.original_chars.mean()),2),'avg_model_summary_chars':round(float(df.summary_chars.mean()),2),'avg_model_compression_fraction':round(float(df.compression_fraction.mean()),4),'fit_seconds':round(fit_seconds,3),'metrics':metrics,'warnings':['Automatic silver references are not independent ground truth.','Metrics reflect prediction of a rule-derived compression decision, NOT factual summary quality.','Near-duplicate semantic groups may still cross splits.','Training model is a compression decision classifier; actual phrase extraction is deterministic and is not a fine-tuned generative model.','Human review required before institutional deployment.']}
(out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(report,ensure_ascii=False,indent=2));print(df[['complaint','silver_summary','model_summary']].head(8).to_string(index=False))
