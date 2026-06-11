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

def generate_pdf(patient_name, pred_class, top_conf, probas, info, now):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    title_style = ParagraphStyle('title', fontSize=18, fontName='Helvetica-Bold',
                                  spaceAfter=10, textColor=colors.HexColor('#4c1d95'))
    elements.append(Paragraph(" Rapport Médical — Brain Tumor Detection", title_style))
    elements.append(Spacer(1, 10))

    # Info table
    data = [
        ['Patient',          patient_name if patient_name else 'Non renseigné'],
        ["Date d'analyse",   now],
        ['Modèle utilisé',   'MobileNetV2 Fine Tuning'],
        ['Accuracy modèle',  '96.21%'],
        ['Diagnostic',       f"{info['emoji']} {info['label']}"],
        ['Confiance',        f"{top_conf:.1f}%"],
        ['Sévérité',         info['severity']],
        ['Recommandation',   info['recommandation']],
    ]

    table = Table(data, colWidths=[150, 330])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#EDE9FE')),
        ('FONTNAME',   (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 11),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.white, colors.HexColor('#F5F3FF')]),
        ('GRID',       (0,0), (-1,-1), 0.5, colors.HexColor('#DDD6FE')),
        ('PADDING',    (0,0), (-1,-1), 8),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 20))

    # Probabilities
    elements.append(Paragraph("Probabilités par classe :", styles['Heading2']))
    for i, cls in enumerate(CLASSES):
        pct = float(probas[i]) * 100
        elements.append(Paragraph(
            f"• {CLASS_INFO[cls]['label']} : <b>{pct:.1f}%</b>",
            styles['Normal']
        ))

    elements.append(Spacer(1, 20))
    disclaimer = ParagraphStyle('disc', fontSize=9, textColor=colors.grey)
    elements.append(Paragraph(
        "Attention : Ce rapport est généré à des fins académiques (PFE). "
        "Il ne remplace pas un diagnostic médical professionnel.",
        disclaimer
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer



st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main {
    background-color:#F8F7FF;
}
.app-header {
    background: linear-gradient(
135deg,
#6D28D9 0%,
#7C3AED 50%,
#A855F7 100%
);
    padding: 2.2rem 2.8rem; border-radius: 20px; margin-bottom: 2rem;
    border: 1px solid #C4B5FD; box-shadow: 0 4px 32px rgba(124,58,237,0.18);
}
.app-header h1 { color: #ffffff; font-size: 2rem; font-weight: 700; margin: 0; }
.app-header p  { color: #ede9fe; margin: 0.5rem 0 0 0; font-size: 0.92rem; }

.st-key-card_irm, .st-key-card_result {
    background: white;
    border: 1px solid #DDD6FE;
    border-radius: 16px; padding: 1.6rem; margin-bottom: 1.2rem;
    box-shadow: 0 2px 16px rgba(124,58,237,0.08);
}
.st-key-card_irm h3, .st-key-card_result h3 { color: #4c1d95; font-size: 0.95rem; font-weight: 600; margin: 0 0 1rem 0; }

.result-glioma     { background: #FEF2F2; border: 2px solid #ef4444; color: #991b1b; }
.result-meningioma { background: #FFFBEB; border: 2px solid #f59e0b; color: #92400e; }
.result-pituitary  { background: #F5F3FF; border: 2px solid #7C3AED; color: #5b21b6; }
.result-no_tumor   { background: #F0FDF4; border: 2px solid #22c55e; color: #166534; }
.result-box {
    border-radius: 14px; padding: 1.4rem 1.6rem; text-align: center;
    font-size: 1.5rem; font-weight: 700; margin: 1rem 0;
}

.conf-bar-bg { background: #EDE9FE; border-radius: 999px; height: 8px; margin: 4px 0 10px 0; overflow: hidden; }
.conf-bar-fill { height: 8px; border-radius: 999px; }

.rapport-box {
    background: #F5F3FF; border: 1px solid #DDD6FE; border-radius: 14px;
    padding: 1.5rem; margin-top: 1rem;
}
.rapport-title { color: #7C3AED; font-size: 1rem; font-weight: 700; margin-bottom: 1rem; }
.rapport-row { display: flex; justify-content: space-between; padding: 0.4rem 0;
    border-bottom: 1px solid #E9D5FF; font-size: 0.83rem; }
.rapport-label { color: #8b7bb8; }
.rapport-value { color: #4c1d95; font-weight: 600; }

.gcam-label { font-size: 0.75rem; color: #8b7bb8; text-align: center; margin-top: 0.3rem; }

[data-testid="stFileUploader"] {
    background: #F5F3FF; border: 2px dashed #DDD6FE; border-radius: 14px; padding: 0.8rem;
}
</style>
""", unsafe_allow_html=True)

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

DRIVE_ID  = '1uGofq96oRk6E5_9vYhVqejarUijrpQMc'
MODEL_FILE = 'brain_tumor_mobilenet_finetuned.h5'

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
        st.error(f" Erreur {e}")
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
        # Le modèle est un Sequential contenant MobileNetV2 comme sous-modèle
        # (ou ses couches conv directement). On découpe juste après la dernière
        # couche convolutive pour récupérer la feature map à visualiser.
        layers = model.layers
        split_idx = None
        for i, layer in enumerate(layers):
            if isinstance(layer, tf.keras.Model) or 'conv' in layer.name.lower():
                split_idx = i
        if split_idx is None:
            raise ValueError("Couche convolutive introuvable pour Grad-CAM.")

        with tf.GradientTape() as tape:
            x = arr
            for layer in layers[:split_idx + 1]:
                x = layer(x, training=False)
            conv_out = x
            tape.watch(conv_out)
            for layer in layers[split_idx + 1:]:
                x = layer(x, training=False)
            loss = x[:, class_idx]
        grads = tape.gradient(loss, conv_out)
        pw = tf.reduce_mean(grads, axis=(0,1,2))
        heatmap = (conv_out[0] @ pw[..., tf.newaxis]).numpy().squeeze()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0: heatmap /= heatmap.max()
        h = cv2.resize(heatmap, (224,224))
        h = np.uint8(255 * h)
        colored = cv2.applyColorMap(h, cv2.COLORMAP_JET)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        orig = np.array(arr[0] * 255, dtype=np.uint8)
        overlay = cv2.addWeighted(orig, 0.55, colored, 0.45, 0)
        return Image.fromarray(overlay)
    except Exception as e:
        st.error(f"Grad-CAM error: {e}")
        return None

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1> Détection des Tumeurs Cérébrales</h1>
    <p>PFE · Deep Learning · MobileNetV2 Fine Tuning &nbsp;│&nbsp; 4 classes &nbsp;│&nbsp;
       <span style="color:#4ade80;font-weight:600">96.21% accuracy</span>
    </p>
</div>
""", unsafe_allow_html=True)

# ── Layout ─────────────────────────────────────────────────────────────────────
col_l, col_r = st.columns([1, 1], gap="large")

with col_l:
    with st.container(key="card_irm"):
        st.markdown("<h3> Image IRM</h3>", unsafe_allow_html=True)
        patient_name = st.text_input("Nom du patient (optionnel)", placeholder="Ex: Ahmed Benali")
        uploaded = st.file_uploader("Uploader une image IRM", type=['jpg','jpeg','png'], label_visibility="collapsed")
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption="IRM uploadée", use_container_width=True)
        else:
            st.markdown("""
            <div style="text-align:center;padding:2.5rem 1rem;color:#c4b5fd">
                <div style="font-size:3rem">🩻</div>
                <div style="font-size:0.88rem;margin-top:0.5rem;color:#8b7bb8">Aucune image sélectionnée</div>
                <div style="font-size:0.75rem;color:#c4b5fd">Formats : JPG, PNG</div>
            </div>
            """, unsafe_allow_html=True)

with col_r:
    with st.container(key="card_result"):
        st.markdown("<h3> Résultat de l'analyse</h3>", unsafe_allow_html=True)

        if uploaded:
            model = load_model()
            if model:
                arr = preprocess(img)
                pred_class, probas = predict(model, arr)
                pred_idx = CLASSES.index(pred_class)
                info = CLASS_INFO[pred_class]
                top_conf = float(np.max(probas)) * 100

                # Result badge
                st.markdown(f"""
                <div class="result-box {info['css']}">
                    {info['emoji']}  {info['label'].upper()}
                </div>
                <p style='color:#8b7bb8;font-size:0.82rem;text-align:center'>{info['desc']}</p>
                """, unsafe_allow_html=True)

                # Severity + Confidence
                if top_conf >= 90: lv, lc = "Très élevée ✅", "#22c55e"
                elif top_conf >= 70: lv, lc = "Élevée 🟡", "#f59e0b"
                else: lv, lc = "Faible ", "#ef4444"

                st.markdown(f"""
                <div style="display:flex;gap:0.6rem;justify-content:center;margin-bottom:1rem">
                    <div style="background:#F5F3FF;border:1px solid #DDD6FE;border-radius:8px;
                                padding:0.4rem 0.9rem;font-size:0.78rem;color:#8b7bb8">
                        Sévérité : <span style="color:{info['sev_color']};font-weight:600">{info['severity']}</span>
                    </div>
                    <div style="background:#F5F3FF;border:1px solid #DDD6FE;border-radius:8px;
                                padding:0.4rem 0.9rem;font-size:0.78rem;color:#8b7bb8">
                        Confiance : <span style="color:{lc};font-weight:600">{top_conf:.1f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Confidence bars
                st.markdown("**Confiance par classe :**")
                for i, cls in enumerate(CLASSES):
                    pct = float(probas[i]) * 100
                    ci = CLASS_INFO[cls]
                    is_top = cls == pred_class
                    ls = "font-weight:700;color:#4c1d95;" if is_top else "color:#8b7bb8;"
                    st.markdown(f"""
                    <div style="margin-bottom:4px">
                        <div style="display:flex;justify-content:space-between;{ls}font-size:0.81rem">
                            <span>{ci['emoji']} {ci['label']}</span><span>{pct:.1f}%</span>
                        </div>
                        <div class="conf-bar-bg">
                            <div class="conf-bar-fill" style="width:{pct:.1f}%;background:{ci['color']}"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Grad-CAM
                st.markdown("---")
                st.markdown("** Grad-CAM — Zone activée :**")
                overlay = gradcam(model, arr, pred_idx)
                if overlay:
                    gc1, gc2 = st.columns(2)
                    with gc1: st.image(img.resize((224,224)), caption="IRM originale", use_container_width=True)
                    with gc2: st.image(overlay, caption="🌡️ Grad-CAM", use_container_width=True)
                    st.markdown('<div class="gcam-label">🔴 Rouge = zone activée &nbsp;|&nbsp; 🔵 Bleu = zone peu activée</div>', unsafe_allow_html=True)
                else:
                    st.info("Grad-CAM non disponible.")

                # Rapport médical
                st.markdown("---")
                now = datetime.now().strftime("%d/%m/%Y à %H:%M")
                patient_display = patient_name if patient_name else "Non renseigné"

                pdf_buffer = generate_pdf(
                    patient_name,
                    pred_class,
                    top_conf,
                    probas,
                    info,
                    now
                )
                st.download_button(
                    label=" Télécharger le rapport PDF",
                    data=pdf_buffer,
                    file_name=f"rapport_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                st.markdown(f"""
                <div class="rapport-box">
                    <div class="rapport-title"> Rapport médical</div>
                    <div class="rapport-row"><span class="rapport-label">Patient</span><span class="rapport-value">{patient_display}</span></div>
                    <div class="rapport-row"><span class="rapport-label">Date d'analyse</span><span class="rapport-value">{now}</span></div>
                    <div class="rapport-row"><span class="rapport-label">Modèle utilisé</span><span class="rapport-value">MobileNetV2 Fine Tuning</span></div>
                    <div class="rapport-row"><span class="rapport-label">Accuracy du modèle</span><span class="rapport-value">96.21%</span></div>
                    <div class="rapport-row"><span class="rapport-label">Diagnostic</span><span class="rapport-value" style="color:{info['color']}">{info['emoji']} {info['label']}</span></div>
                    <div class="rapport-row"><span class="rapport-label">Confiance</span><span class="rapport-value" style="color:{lc}">{top_conf:.1f}%</span></div>
                    <div class="rapport-row"><span class="rapport-label">Sévérité</span><span class="rapport-value" style="color:{info['sev_color']}">{info['severity']}</span></div>
                    <div style="margin-top:0.8rem;padding:0.7rem;background:#EDE9FE;border-radius:8px;
                                border-left:3px solid {info['color']};font-size:0.81rem;color:#5b21b6">
                         <strong style="color:#4c1d95">Recommandation :</strong> {info['recommandation']}
                    </div>
                    <div style="margin-top:0.8rem;font-size:0.72rem;color:#a78bda;text-align:center">
                        Attention : Ce rapport est généré à des fins académiques (PFE). Consultez un médecin qualifié.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="text-align:center;padding:5rem 1rem;color:#c4b5fd">
                <div style="font-size:3.5rem">🔬</div>
                <div style="font-size:0.95rem;margin-top:1rem;color:#8b7bb8">
                    Uploadez une image IRM<br>pour démarrer l'analyse
                </div>
            </div>
            """, unsafe_allow_html=True)

# ── Disclaimer ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:1.5rem;padding:0.9rem 1.4rem;background:#F5F3FF;
     border-left:3px solid #7C3AED;border-radius:8px;font-size:0.79rem;color:#8b7bb8">
    Attention : <strong style="color:#5b21b6">Avertissement médical :</strong>
    Cette application est développée à des fins académiques (PFE).
    Elle ne remplace pas un diagnostic médical professionnel.
</div>
""", unsafe_allow_html=True)
