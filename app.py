import re, sqlite3, secrets, hashlib, hmac, datetime, tomllib
from pathlib import Path
import streamlit as st
import joblib

BASE=Path(__file__).resolve().parent
# The original theme file lives beside app.py in the flat distribution.
# Streamlit 1.65.0 is pinned because this bootstrap uses its config API.
with (BASE/'config.toml').open('rb') as theme_file:
    flat_theme=tomllib.load(theme_file)['theme']
flat_theme_changed=False
for theme_key,theme_value in flat_theme.items():
    option_key='theme.'+theme_key
    if st._config.get_option(option_key)!=theme_value:
        st._config.set_option(option_key,theme_value)
        flat_theme_changed=True
# A new script run sends the loaded theme to the browser before the UI.
if flat_theme_changed:
    st.rerun()
DB=BASE/'student_cases.sqlite3'
TASHKEEL=re.compile(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]')
def normalize_arabic(text):
    text=TASHKEEL.sub('',str(text)).replace('ـ','')
    text=re.sub('[إأآٱ]','ا',text).replace('ى','ي')
    return re.sub(r'\s+',' ',text).strip()

# Exact conservative extraction rules from the original research script.
PREFIX=[r'^أنا لاحظت إن\s*',r'^لما رجعت أتأكد لقيت إن\s*',r'^كنت بتابع الموضوع ولقيت إن\s*',r'^اللي حصل معايا إن\s*',r'^محتاج توضيح بخصوص الآتي:\s*',r'^عندي مشكلة وعايز حل:\s*',r'^عايز أعرف أتصرف إزاي[،,:]\s*',r'^عايز أعرف سبب إن\s*',r'^ممكن توضحوا موضوع إن\s*',r'^أنا مش فاهم ليه\s*',r'^ممكن أفهم ليه\s*',r'^هو المفروض أعمل إيه لو\s*',r'^هو ليه\s*',r'^بستفسر عن حاجة[،,:]\s*',r'^كنت عايز أسأل:\s*',r'^أنا محتاج مساعدة[،,:]\s*',r'^ممكن حد يفهمني[،,:]\s*',r'^يا جماعة\s*',r'^لو سمحت[،,:]?\s*',r'^من فضلكم[،,:]?\s*',r'^بعد إذنكم[،,:]?\s*',r'^برجاء الإفادة[،,:]?\s*',r'^برجاء المساعدة[،,:]?\s*',r'^أرجو المساعدة[،,:]?\s*',r'^أود الاستفسار[،,:]?\s*',r'^السلام عليكم[،,:]?\s*',r'^معلش[،,:]?\s*']
CLOSINGS=['ممكن مساعدة','أرجو مراجعة الموضوع','يا ريت تراجعوا ده','محتاج أعرف الخطوة الجاية','محتاج الموضوع يتراجع','محتاج رد لو سمحتم','أعمل إيه دلوقتي','أتصرف إزاي','ممكن توضيح','هل فيه حل متاح','ممكن أعرف الحل','مش عارف أتصرف إزاي','ومش عارف أبدأ منين','وأتمنى مراجعة الأمر','وأرجو توضيح الإجراء المناسب','ممكن تساعدوني أفهم السبب','يا ريت حد يوضح لي الخطوة الجاية','أتواصل مع مين بخصوص ده','هل في طريقة أتابع بيها','عايز أتأكد إن الموضوع اتسجل']
FILLERS=['ما عنديش معلومات زيادة غير اللي ظاهر قدامي','أنا مش متأكد إذا الخطأ من عندي ولا من النظام','مش عايز أعمل خطوة غلط وأضيع وقت','سألت قبل كده ولسه ما وصلت لرد واضح','جربت أستفسر بس الردود كانت مختلفة','كلمت حد وقال لي أتابع بعدين من غير تفاصيل','حاولت أراجع بنفسي لكن ما فهمتش اللي حصل','دخلت على حسابي تاني ونفس الوضع موجود','يهمني أعرف هل لازم أقدم طلب جديد','الموضوع مقلقني لأن عندي إجراء تاني مرتبط بيه','وده معطلني في اللي ورايا']
def derive(s):
    t=s.strip()
    for p in PREFIX:
        n=re.sub(p,'',t,count=1)
        if n!=t:t=n;break
    for f in FILLERS:t=re.sub(r'(?:[.؟!]\s*|،\s*)'+re.escape(f)+r'(?=[.؟!،]|$)','',t)
    for c in CLOSINGS:t=re.sub(r'(?:[،.؟!]\s*|\s+)'+re.escape(c)+r'[؟.!\s]*$','',t)
    t=re.sub(r'\s+',' ',t).strip(' ،.؟!:')
    if re.search(r'(?:مش عارف|مش قادر|ومحتاج|ومش|وعايز|لكن|بس)$',t):return s.strip()
    if not t or len(t)<12 or len(t)>len(s):return s.strip()
    return t

@st.cache_resource
def models():
    clf=joblib.load(BASE/'classifier.joblib')
    cmp=joblib.load(BASE/'compression.joblib')
    return clf,cmp
UNITS={'CARDS':('Cards Office','وحدة الكارنيهات'),'EXAMS':('Examinations Office','وحدة الامتحانات والكونترول'),'RECORDS':('Records & Registration','وحدة القيد والملفات'),'RESULTS':('Results Office','وحدة النتائج')}
def infer(body):
    clf,cmp=models()
    label=str(clf.predict([body])[0])
    vec=cmp['vectorizer'];decision=cmp['model'].predict(vec.transform([body]))[0]
    summary=derive(body) if int(decision)==1 else body
    return label,summary,bool(decision)
def connect():
    c=sqlite3.connect(DB);c.row_factory=sqlite3.Row;return c
def init():
    with connect() as c:
        c.execute('CREATE TABLE IF NOT EXISTS cases (id INTEGER PRIMARY KEY, reference TEXT UNIQUE, code_hash TEXT, name TEXT, email TEXT, body TEXT, summary TEXT, unit TEXT, compression INTEGER, created_at TEXT, rating INTEGER, comment TEXT)')
        c.execute('CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, case_id INTEGER, action TEXT, note TEXT, created_at TEXT)')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
def save(name,email,body,label,summary,compressed):
    ref='SD-'+datetime.datetime.now().strftime('%Y%m%d')+'-'+secrets.token_hex(4).upper()
    code=secrets.token_urlsafe(12)
    with connect() as c:
        cur=c.execute('INSERT INTO cases(reference,code_hash,name,email,body,summary,unit,compression,created_at) VALUES (?,?,?,?,?,?,?,?,?)',(ref,hashlib.sha256(code.encode()).hexdigest(),name,email,body,summary,label,int(compressed),now()))
        c.execute('INSERT INTO events(case_id,action,note,created_at) VALUES (?,?,?,?)',(cur.lastrowid,'Submitted','Complaint received and analyzed; suggested destination generated.',now()))
    return ref,code
def lookup(ref,code):
    with connect() as c:
        row=c.execute('SELECT * FROM cases WHERE reference=?',(ref.strip(),)).fetchone()
        if row and hmac.compare_digest(row['code_hash'],hashlib.sha256(code.encode()).hexdigest()):
            return dict(row),[dict(e) for e in c.execute('SELECT * FROM events WHERE case_id=? ORDER BY id',(row['id'],))]
    return None,None

def tr(en,ar):return ar if st.session_state.lang=='ar' else en
st.set_page_config(page_title='SADAT | Service Quality Center',page_icon='🎓',layout='wide')
init()
if 'lang' not in st.session_state:st.session_state.lang='en'
if 'draft' not in st.session_state:st.session_state.draft=None
if 'receipt' not in st.session_state:st.session_state.receipt=None
st.markdown('<style>'+ (BASE/'theme.css').read_text(encoding='utf-8') +'</style>',unsafe_allow_html=True)
with st.sidebar:
    st.image(str(BASE/'logo.jpg'),width=90)
    st.markdown('### SADAT')
    st.caption('SERVICE QUALITY CENTER')
    st.session_state.lang=st.segmented_control('Language / اللغة',['en','ar'],format_func=lambda x:'English' if x=='en' else 'العربية',default=st.session_state.lang,key='lang_widget')
    choices=['Overview','Submit Complaint','Track Complaint','Feedback']
    labels={'Overview':tr('Overview','الرئيسية'),'Submit Complaint':tr('Submit Complaint','تقديم شكوى'),'Track Complaint':tr('Track Complaint','متابعة الشكوى'),'Feedback':tr('Feedback','تقييم التجربة')}
    page=st.radio(tr('Navigation','التنقل'),choices,format_func=lambda x:labels[x],label_visibility='collapsed')
    st.caption(tr('Student experience · No staff workspace','واجهة الطالب فقط · بدون مساحة موظفين'))

def hero(title,sub):
    st.markdown(f'<div class="hero"><small>SADAT · SERVICE QUALITY CENTER</small><h1>{title}</h1><p>{sub}</p></div>',unsafe_allow_html=True)
if page=='Overview':
    hero(tr('Complaints & Service Quality Center','مركز الشكاوى وجودة الخدمات'),tr('Tell us what happened. Review the suggested destination and summary, then follow your submission.','احكي مشكلتك بحرية، راجع الملخص والجهة المقترحة، وبعدها تابع شكواك.'))
    a,b,c=st.columns(3)
    with connect() as con:
        count=con.execute('SELECT COUNT(*) FROM cases').fetchone()[0]
        rated=con.execute('SELECT COUNT(*) FROM cases WHERE rating IS NOT NULL').fetchone()[0]
    a.metric(tr('Submitted complaints','الشكاوى المقدمة'),count)
    b.metric(tr('Experience ratings','تقييمات التجربة'),rated)
    c.metric(tr('Workflow status','حالة المعالجة'),tr('Submission only','استلام فقط'))
    st.info(tr('The student-only version does not process or close complaints. Routing is a model suggestion, not a confirmed administrative assignment.','هذه النسخة لا تعالج الشكاوى أو تغلقها. الجهة المعروضة اقتراح من الموديل وليست تكليفًا إداريًا مؤكدًا.'))
elif page=='Submit Complaint':
    hero(tr('Submit a complaint','تقديم شكوى'),tr('Write naturally in Arabic. You do not need to choose a department.','اكتب الشكوى بطريقتك بالعربي، ومش مطلوب تختار الإدارة.'))
    with st.form('entry'):
        name=st.text_input(tr('Full name','الاسم الكامل'),max_chars=100)
        email=st.text_input(tr('Email address','البريد الإلكتروني'))
        body=st.text_area(tr('Describe your complaint','احكي مشكلتك'),height=190,max_chars=5000,placeholder='مثال: أنا قدمت طلب كارنيه من أسبوعين ولسه ما استلمتوش...')
        submitted=st.form_submit_button(tr('Analyze complaint','تحليل الشكوى'),type='primary')
    if submitted:
        if not name.strip() or not re.fullmatch(r'[^@\s]+@[^@\s]+\.[^@\s]+',email.strip()) or len(body.strip())<12:
            st.error(tr('Please enter a name, valid email and complaint of at least 12 characters.','اكتب الاسم وبريد صحيح وشكوى لا تقل عن 12 حرف.'))
        else:
            try:
                label,summary,compressed=infer(body.strip())
                st.session_state.draft=(name.strip(),email.strip(),body.strip(),label,summary,compressed)
                st.session_state.receipt=None
            except Exception as exc:
                st.error(tr('Model inference failed; complaint has not been saved.','فشل تشغيل الموديل؛ الشكوى لم تُحفظ.'))
                st.code(str(exc))
    if st.session_state.draft:
        name,email,body,label,summary,compressed=st.session_state.draft
        st.subheader(tr('Review before submission','راجع النتيجة قبل الإرسال'))
        unit=UNITS.get(label,('Unrecognized category','فئة غير معروفة'))
        st.write('**'+tr('Suggested destination','الجهة المقترحة')+':** '+unit[1 if st.session_state.lang=='ar' else 0])
        st.write('**'+tr('Complaint summary','ملخص الشكوى')+':**')
        st.info(summary)
        st.caption(tr('Compression decision: ','قرار الاختصار: ')+(tr('Applied','تم الاختصار') if compressed and summary!=body else tr('Original text retained','تم الاحتفاظ بالنص الأصلي')))
        st.warning(tr('Classification is a research baseline trained on synthetic data; verify before operational routing.','التصنيف موديل بحثي مدرب على بيانات اصطناعية، ويحتاج تحققًا قبل التوجيه التشغيلي.'))
        if st.button(tr('Confirm and save complaint','تأكيد وحفظ الشكوى'),type='primary'):
            st.session_state.receipt=save(*st.session_state.draft)
            st.session_state.draft=None
            st.rerun()
    if st.session_state.receipt:
        ref,code=st.session_state.receipt
        st.success(tr('Complaint saved successfully','تم حفظ الشكوى بنجاح'))
        st.caption(tr('Reference','رقم المتابعة'))
        st.code(ref)
        st.caption(tr('Private tracking code — save it now','رمز المتابعة السري — احتفظ به الآن'))
        st.code(code)
        st.info(tr('You will need both to track this complaint.','هتحتاج الرقم والرمز لمتابعة الشكوى.'))
elif page=='Track Complaint':
    hero(tr('Track your complaint','متابعة الشكوى'),tr('Use your reference and private tracking code.','استخدم رقم المتابعة والرمز السري.'))
    with st.form('track'):
        ref=st.text_input(tr('Reference','رقم المتابعة'))
        code=st.text_input(tr('Tracking code','رمز المتابعة'),type='password')
        find=st.form_submit_button(tr('Find complaint','عرض الشكوى'),type='primary')
    if find:
        row,events=lookup(ref,code)
        if row:
            st.success(tr('Received · Awaiting institutional processing','تم الاستلام · بانتظار المعالجة المؤسسية'))
            st.write('**'+tr('Suggested destination','الجهة المقترحة')+':** '+UNITS.get(row['unit'],('Unknown','غير معروف'))[1 if st.session_state.lang=='ar' else 0])
            st.write('**'+tr('Summary','الملخص')+':** '+row['summary'])
            st.write('**'+tr('Original complaint','الشكوى الأصلية')+':** '+row['body'])
            st.subheader(tr('Timeline','سجل الأحداث'))
            for event in events:st.write(f"{event['created_at']} · {event['action']} — {event['note']}")
            st.caption(tr('No staff updates or resolutions are generated in this student-only build.','لا توجد تحديثات موظفين أو حلول مُختلقة في هذه النسخة.'))
        else:st.error(tr('Reference or tracking code is incorrect.','رقم المتابعة أو الرمز غير صحيح.'))
else:
    hero(tr('Rate your submission experience','تقييم تجربة التقديم'),tr('Rate how easy it was to submit and track your complaint — not whether the problem was resolved.','قيّم سهولة تقديم الشكوى ومتابعتها، وليس حل المشكلة.'))
    with st.form('rating'):
        ref=st.text_input(tr('Reference','رقم المتابعة'))
        code=st.text_input(tr('Tracking code','رمز المتابعة'),type='password')
        rating=st.select_slider(tr('Experience rating','تقييم التجربة'),options=[1,2,3,4,5],value=5)
        comment=st.text_area(tr('Optional feedback','ملاحظات اختيارية'),max_chars=2000)
        send=st.form_submit_button(tr('Submit rating','إرسال التقييم'),type='primary')
    if send:
        row,_=lookup(ref,code)
        if not row:st.error(tr('Invalid tracking details.','بيانات المتابعة غير صحيحة.'))
        elif row['rating'] is not None:st.warning(tr('This complaint was already rated.','الشكوى دي اتقيّمت قبل كده.'))
        else:
            with connect() as c:
                c.execute('UPDATE cases SET rating=?,comment=? WHERE id=?',(rating,comment,row['id']))
                c.execute('INSERT INTO events(case_id,action,note,created_at) VALUES (?,?,?,?)',(row['id'],'Experience feedback','Student rated submission experience.',now()))
            st.success(tr('Thank you for your feedback.','شكرًا على تقييمك.'))
