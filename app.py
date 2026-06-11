import streamlit as st
import numpy as np
from PIL import Image
import os
import gdown
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import io

# ── Constants ──────────────────────────────────────────────────────────────────
CLASSES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

CLASS_INFO = {
    'glioma':     {'emoji':'🔴','label':'Gliome',             'color':'#ef4444','css':'result-glioma',
                   'desc':'Tumeur maligne des cellules gliales (~30% des tumeurs cérébrales).',
                   'severity':'Élevée','sev_color':'#ef4444',
                   'recommandation':'Consultation urgente en neurochirurgie recommandée.'},
    'meningioma': {'emoji':'🟡','label':'Méningiome',          'color':'#f59e0b','css':'result-meningioma',
                   'desc':'Tumeur des méninges, souvent bénigne et à croissance lente.',
                   'severity':'Modérée','sev_color':'#f59e0b',
                   'recommandation':'Suivi neurologique et imagerie de contrôle conseillés.'},
    'pituitary':  {'emoji':'🔵','label':'Tumeur hypophysaire', 'color':'#3b82f6','css':'result-pituitary',
                   'desc':"Tumeur de la glande pituitaire, généralement bénigne et traitable.",
                   'severity':'Faible','sev_color':'#3b82f6',
                   'recommandation':'Consultation endocrinologique et suivi IRM recommandés.'},
    'no_tumor':   {'emoji':'🟢','label':'Aucune tumeur',       'color':'#22c55e','css':'result-no_tumor',
                   'desc':"Aucune anomalie tumorale détectée sur l'IRM analysée.",
                   'severity':'Aucune','sev_color':'#22c55e',
                   'recommandation':'Aucune action urgente requise. Suivi de routine conseillé.'},
}

DRIVE_ID   = '1uGofq96oRk6E5_9vYhVqejarUijrpQMc'
MODEL_FILE = 'brain_tumor_mobilenet_finetuned.h5'

# ── PDF ────────────────────────────────────────────────────────────────────────
def generate_pdf(patient_name, pred_class, top_conf, probas, info, now):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            leftMargin=40, rightMargin=40,
                            topMargin=50, bottomMargin=40)
    styles = getSampleStyleSheet()
    elements = []

    title_style = ParagraphStyle('title', fontSize=18, fontName='Helvetica-Bold',
                                 spaceAfter=6, textColor=colors.HexColor('#4c1d95'))
    sub_style = ParagraphStyle('sub', fontSize=10, textColor=colors.HexColor('#8b7bb8'),
                               spaceAfter=16)
    elements.append(Paragraph("Rapport Médical — Brain Tumor Detection", title_style))
    elements.append(Paragraph("Généré automatiquement par NeuroScan AI · PFE Deep Learning", sub_style))

    data = [
        ['Patient',           patient_name if patient_name else 'Non renseigné'],
        ["Date d'analyse",    now],
        ['Modèle utilisé',    'MobileNetV2 Fine Tuning'],
        ['Accuracy modèle',   '96.21%'],
        ['Diagnostic',        f"{info['label']}"],
        ['Confiance',         f"{top_conf:.1f}%"],
        ['Sévérité',          info['severity']],
        ['Recommandation',    info['recommandation']],
    ]
    table = Table(data, colWidths=[150, 330])
    table.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (0, -1), colors.HexColor('#EDE9FE')),
        ('FONTNAME',      (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0), (-1, -1), 10),
        ('ROWBACKGROUNDS',(0, 0), (-1, -1), [colors.white, colors.HexColor('#F5F3FF')]),
        ('GRID',          (0, 0), (-1, -1), 0.4, colors.HexColor('#DDD6FE')),
        ('PADDING',       (0, 0), (-1, -1), 8),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("Probabilités par classe :", styles['Heading2']))
    elements.append(Spacer(1, 6))
    for i, cls in enumerate(CLASSES):
        pct = float(probas[i]) * 100
        bold = '<b>' if cls == pred_class else ''
        bold_end = '</b>' if cls == pred_class else ''
        elements.append(Paragraph(
            f"• {CLASS_INFO[cls]['label']} : {bold}{pct:.1f}%{bold_end}",
            styles['Normal']
        ))
    elements.append(Spacer(1, 24))

    disc_style = ParagraphStyle('disc', fontSize=8, textColor=colors.HexColor('#9ca3af'),
                                borderPad=6, backColor=colors.HexColor('#F5F3FF'),
                                borderColor=colors.HexColor('#DDD6FE'), borderWidth=0.5)
    elements.append(Paragraph(
        "⚠ Attention : Ce rapport est généré à des fins académiques (PFE). "
        "Il ne remplace pas un diagnostic médical professionnel. "
        "Consultez toujours un médecin qualifié.",
        disc_style
    ))
    doc.build(elements)
    buffer.seek(0)
    return buffer

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="NeuroScan AI", page_icon="🧠", layout="wide")

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── fond global ── */
.stApp { background-color: #F3F0FF; }
section[data-testid="stMain"] > div { background-color: #F3F0FF; }

/* ── cacher le disclaimer Streamlit par défaut ── */
[data-testid="stStatusWidget"] { display: none; }

/* ── header ── */
.app-header {
    background: linear-gradient(135deg, #5B21B6 0%, #7C3AED 55%, #A855F7 100%);
    padding: 2rem 2.6rem; border-radius: 18px; margin-bottom: 1.8rem;
    border: 1px solid #C4B5FD;
    box-shadow: 0 8px 32px rgba(124,58,237,0.22);
}
.app-header h1 { color: #fff; font-size: 1.85rem; font-weight: 700; margin: 0; letter-spacing:-0.3px; }
.app-header p  { color: #EDE9FE; margin: 0.45rem 0 0; font-size: 0.9rem; }

/* ── cards ── */
.card {
    background: #fff;
    border: 1px solid #DDD6FE;
    border-radius: 16px; padding: 1.5rem; margin-bottom: 1.1rem;
    box-shadow: 0 2px 14px rgba(124,58,237,0.07);
}
.card h3 { color: #5B21B6; font-size: 0.9rem; font-weight: 600; margin: 0 0 1rem; }

/* ── result boxes ── */
.result-glioma     { background:#FEF2F2; border:2px solid #ef4444; color:#991b1b; }
.result-meningioma { background:#FFFBEB; border:2px solid #f59e0b; color:#92400e; }
.result-pituitary  { background:#F5F3FF; border:2px solid #7C3AED; color:#5b21b6; }
.result-no_tumor   { background:#F0FDF4; border:2px solid #22c55e; color:#166534; }
.result-box {
    border-radius: 12px; padding: 1.2rem 1.5rem; text-align: center;
    font-size: 1.4rem; font-weight: 700; margin: 0.8rem 0;
}

/* ── conf bars ── */
.conf-bar-bg  { background:#EDE9FE; border-radius:999px; height:7px; margin:3px 0 9px; overflow:hidden; }
.conf-bar-fill{ height:7px; border-radius:999px; }

/* ── rapport ── */
.rapport-box {
    background:#F5F3FF; border:1px solid #DDD6FE; border-radius:14px;
    padding:1.4rem; margin-top:0.8rem;
}
.rapport-title { color:#7C3AED; font-size:0.95rem; font-weight:700; margin-bottom:0.9rem; }
.rapport-row {
    display:flex; justify-content:space-between; padding:0.38rem 0;
    border-bottom:1px solid #E9D5FF; font-size:0.82rem;
}
.rapport-label { color:#9c8fc0; }
.rapport-value { color:#4c1d95; font-weight:600; }

/* ── gcam label ── */
.gcam-label { font-size:0.74rem; color:#9c8fc0; text-align:center; margin-top:0.3rem; }

/* ── upload area ── */
[data-testid="stFileUploader"] {
    background:#F5F3FF; border:2px dashed #C4B5FD; border-radius:12px; padding:0.6rem;
}

/* ── disclaimer bottom ── */
.disclaimer-box {
    margin-top:1.5rem; padding:0.85rem 1.3rem;
    background:#F5F3FF; border-left:3px solid #7C3AED;
    border-radius:8px; font-size:0.78rem; color:#9c8fc0;
}

/* ── download button ── */
[data-testid="stDownloadButton"] button {
    background: linear-gradient(135deg,#5B21B6,#7C3AED) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.6rem 1.2rem !important;
    width: 100% !important;
    margin-bottom: 0.6rem;
}

/* ── text input ── */
[data-testid="stTextInput"] input {
    background: #F5F3FF !important;
    border: 1px solid #DDD6FE !important;
    border-radius: 10px !important;
    color: #4c1d95 !important;
    font-size: 0.88rem !important;
}
[data-testid="stTextInput"] label {
    color: #7C3AED !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        import tensorflow as tf
        if not os.path.exists(MODEL_FILE):
            url = f"https://drive.google.com/uc?id={DRIVE_ID}"
            with st.spinner(" Téléchargement du modèle..."):
                gdown.download(url, MODEL_FILE, quiet=False)
        return tf.keras.models.load_model(MODEL_FILE)
    except Exception as e:
        st.error(f"❌ Erreur : {e}")
        return None

def preprocess(img):
    img = img.convert('RGB').resize((224, 224))
    return np.expand_dims(np.array(img) / 255.0, axis=0)

def predict(model, arr):
    preds = model.predict(arr, verbose=0)[0]
    return CLASSES[int(np.argmax(preds))], preds

# ── Grad-CAM ───────────────────────────────────────────────────────────────────
def gradcam(model, arr, class_idx):
    try:
        import tensorflow as tf
        import cv2
        if not getattr(model, "_inbound_nodes", None):
            model(arr, training=False)
        gm = tf.keras.models.Model(
            inputs=model.input,
            outputs=[model.get_layer('Conv_1').output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_out, preds = gm(arr, training=False)
            loss = preds[:, class_idx]
        grads  = tape.gradient(loss, conv_out)
        pw     = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = (conv_out[0] @ pw[..., tf.newaxis]).numpy().squeeze()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        h       = cv2.resize(heatmap, (224, 224))
        h       = np.uint8(255 * h)
        colored = cv2.applyColorMap(h, cv2.COLORMAP_JET)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        orig    = np.array(arr[0] * 255, dtype=np.uint8)
        overlay = cv2.addWeighted(orig, 0.55, colored, 0.45, 0)
        return Image.fromarray(overlay)
    except Exception as e:
        st.warning(f"Grad-CAM non disponible : {e}")
        return None

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-header">
    <h1> Détection des Tumeurs Cérébrales</h1>
    <p>PFE · Deep Learning · MobileNetV2 Fine Tuning &nbsp;│&nbsp; 4 classes &nbsp;│&nbsp;
       <span style="color:#4ade80;font-weight:600">96.21% accuracy</span>
    </p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# COLUMNS
# ══════════════════════════════════════════════════════════════════════════════
col_l, col_r = st.columns([1, 1], gap="large")

# ── LEFT ──────────────────────────────────────────────────────────────────────
with col_l:
    st.markdown('<div class="card"><h3>🩻 Image IRM</h3>', unsafe_allow_html=True)

    patient_name = st.text_input(
        "Nom du patient (optionnel)",
        placeholder="Ex: Ahmed Benali"
    )

    uploaded = st.file_uploader(
        "Uploader une image IRM",
        type=['jpg', 'jpeg', 'png'],
        label_visibility="collapsed"
    )

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="IRM uploadée", use_container_width=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:2.5rem 1rem">
            <div style="font-size:2.8rem">🩻</div>
            <div style="font-size:0.87rem;margin-top:0.5rem;color:#9c8fc0">Aucune image sélectionnée</div>
            <div style="font-size:0.74rem;color:#C4B5FD;margin-top:3px">Formats : JPG, PNG</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ── RIGHT ─────────────────────────────────────────────────────────────────────
with col_r:
    st.markdown('<div class="card"><h3> Résultat de l\'analyse</h3>', unsafe_allow_html=True)

    if uploaded:
        model = load_model()
        if model:
            arr        = preprocess(img)
            pred_class, probas = predict(model, arr)
            pred_idx   = CLASSES.index(pred_class)
            info       = CLASS_INFO[pred_class]
            top_conf   = float(np.max(probas)) * 100

            # ── Résultat ──
            st.markdown(f"""
            <div class="result-box {info['css']}">
                {info['emoji']} &nbsp; {info['label'].upper()}
            </div>
            <p style='color:#9c8fc0;font-size:0.81rem;text-align:center;margin-bottom:0.8rem'>
                {info['desc']}
            </p>
            """, unsafe_allow_html=True)

            # ── Sévérité + Confiance ──
            if top_conf >= 90:   lv, lc = "Très élevée ", "#22c55e"
            elif top_conf >= 70: lv, lc = "Élevée 🟡",      "#f59e0b"
            else:                lv, lc = "Faible ",       "#ef4444"

            st.markdown(f"""
            <div style="display:flex;gap:0.6rem;justify-content:center;margin-bottom:1rem">
                <div style="background:#F5F3FF;border:1px solid #DDD6FE;border-radius:8px;
                            padding:0.4rem 0.9rem;font-size:0.77rem;color:#9c8fc0">
                    Sévérité : <span style="color:{info['sev_color']};font-weight:600">{info['severity']}</span>
                </div>
                <div style="background:#F5F3FF;border:1px solid #DDD6FE;border-radius:8px;
                            padding:0.4rem 0.9rem;font-size:0.77rem;color:#9c8fc0">
                    Confiance : <span style="color:{lc};font-weight:600">{top_conf:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Confidence bars ──
            st.markdown(
                "<div style='font-size:0.82rem;font-weight:600;color:#5B21B6;"
                "margin-bottom:6px'>Confiance par classe :</div>",
                unsafe_allow_html=True
            )
            for i, cls in enumerate(CLASSES):
                pct    = float(probas[i]) * 100
                ci     = CLASS_INFO[cls]
                is_top = cls == pred_class
                ls     = "font-weight:700;color:#4c1d95;" if is_top else "color:#9c8fc0;"
                st.markdown(f"""
                <div style="margin-bottom:4px">
                    <div style="display:flex;justify-content:space-between;{ls}font-size:0.8rem">
                        <span>{ci['emoji']} {ci['label']}</span><span>{pct:.1f}%</span>
                    </div>
                    <div class="conf-bar-bg">
                        <div class="conf-bar-fill" style="width:{pct:.1f}%;background:{ci['color']}"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Grad-CAM ──
            st.markdown(
                "<hr style='border:none;border-top:1px solid #EDE9FE;margin:0.9rem 0'>",
                unsafe_allow_html=True
            )
            st.markdown(
                "<div style='font-size:0.82rem;font-weight:600;color:#5B21B6;"
                "margin-bottom:6px'> Grad-CAM — Zone activée :</div>",
                unsafe_allow_html=True
            )
            overlay = gradcam(model, arr, pred_idx)
            if overlay:
                gc1, gc2 = st.columns(2)
                with gc1:
                    st.image(img.resize((224, 224)),
                             caption="IRM originale",
                             use_container_width=True)
                with gc2:
                    st.image(overlay,
                             caption="🌡️ Grad-CAM",
                             use_container_width=True)
                st.markdown(
                    '<div class="gcam-label">'
                    '🔴 Rouge = zone activée &nbsp;|&nbsp; 🔵 Bleu = zone peu activée'
                    '</div>',
                    unsafe_allow_html=True
                )
            else:
                st.info("Grad-CAM non disponible pour ce modèle.")

            # ── Rapport médical ──
            st.markdown(
                "<hr style='border:none;border-top:1px solid #EDE9FE;margin:0.9rem 0'>",
                unsafe_allow_html=True
            )
            now             = datetime.now().strftime("%d/%m/%Y à %H:%M")
            patient_display = patient_name if patient_name else "Non renseigné"

            pdf_buffer = generate_pdf(
                patient_name, pred_class, top_conf, probas, info, now
            )

            st.download_button(
                label="📄 Télécharger le rapport PDF",
                data=pdf_buffer,
                file_name=f"rapport_neuroscan_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

            st.markdown(f"""
            <div class="rapport-box">
                <div class="rapport-title"> Rapport médical</div>
                <div class="rapport-row">
                    <span class="rapport-label">Patient</span>
                    <span class="rapport-value">{patient_display}</span>
                </div>
                <div class="rapport-row">
                    <span class="rapport-label">Date d'analyse</span>
                    <span class="rapport-value">{now}</span>
                </div>
                <div class="rapport-row">
                    <span class="rapport-label">Modèle utilisé</span>
                    <span class="rapport-value">MobileNetV2 Fine Tuning</span>
                </div>
                <div class="rapport-row">
                    <span class="rapport-label">Accuracy du modèle</span>
                    <span class="rapport-value">96.21%</span>
                </div>
                <div class="rapport-row">
                    <span class="rapport-label">Diagnostic</span>
                    <span class="rapport-value" style="color:{info['color']}">
                        {info['emoji']} {info['label']}
                    </span>
                </div>
                <div class="rapport-row">
                    <span class="rapport-label">Confiance</span>
                    <span class="rapport-value" style="color:{lc}">{top_conf:.1f}%</span>
                </div>
                <div class="rapport-row" style="border-bottom:none">
                    <span class="rapport-label">Sévérité</span>
                    <span class="rapport-value" style="color:{info['sev_color']}">{info['severity']}</span>
                </div>
                <div style="margin-top:0.8rem;padding:0.65rem 0.9rem;background:#EDE9FE;
                            border-radius:8px;border-left:3px solid {info['color']};
                            font-size:0.8rem;color:#5b21b6;line-height:1.5">
                    <strong style="color:#4c1d95">Recommandation :</strong>
                    {info['recommandation']}
                </div>
                <div style="margin-top:0.7rem;font-size:0.71rem;color:#b0a0d4;text-align:center">
                    Attention: Ce rapport est généré à des fins académiques. Consultez un médecin qualifié.
                </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center;padding:5rem 1rem">
            <div style="font-size:3.2rem">🔬</div>
            <div style="font-size:0.92rem;margin-top:1rem;color:#9c8fc0">
                Uploadez une image IRM<br>pour démarrer l'analyse
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DISCLAIMER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="disclaimer-box">
     <strong style="color:#5b21b6">Avertissement médical :</strong>
    Cette application est développée à des fins académiques (PFE).
    Elle ne remplace pas un diagnostic médical professionnel.
    Consultez toujours un médecin qualifié pour toute décision médicale.
</div>
""", unsafe_allow_html=True)
