import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import base64
import textwrap
import streamlit.components.v1 as components


# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="SeismoSense",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ------------------------------------------------------------
# PROJECT CONFIGURATION
# ------------------------------------------------------------

MODEL_PATH = "models/earthquake_model_v2.keras"
NORMALIZATION_PATH = "models/normalization.npz"
GIF_PATH = "assets/seismosense.gif"


# ------------------------------------------------------------
# HTML RENDER HELPER
# ------------------------------------------------------------

def render_html(html):
    st.html(
        textwrap.dedent(html).strip()
    )


# ------------------------------------------------------------
# CUSTOM STYLING
# ------------------------------------------------------------

st.markdown(
    """
    <style>

    /* -------------------------------------------------- */
    /* Global */
    /* -------------------------------------------------- */

    .stApp {
        background:
            radial-gradient(
                circle at 50% 0%,
                rgba(255, 43, 176, 0.045),
                transparent 35%
            ),
            #08090d;

        color: #eee8ed;

        font-family:
            "Courier New",
            Courier,
            monospace;
    }

    .block-container {
        max-width: 1420px;
        padding-top: 1rem;
        padding-bottom: 3rem;
    }

    /* -------------------------------------------------- */
    /* Hide unnecessary Streamlit chrome */
    /* -------------------------------------------------- */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* -------------------------------------------------- */
    /* Sidebar */
    /* -------------------------------------------------- */

    section[data-testid="stSidebar"] {
        background: #090a0f;
        border-right: 1px solid #35152d;
        min-width: 280px;
    }

    section[data-testid="stSidebar"] {
        font-family:
            "Courier New",
            Courier,
            monospace;
    }

    /* -------------------------------------------------- */
    /* Section headings */
    /* -------------------------------------------------- */

    .section-title {
        color: #ff2bb0;
        font-family:
            "Courier New",
            Courier,
            monospace;

        font-size: 1.05rem;
        font-weight: 700;

        letter-spacing: 0.6px;

        margin-top: 1rem;
        margin-bottom: 0.35rem;
    }

    .section-title::before {
        content: "/ ";
        color: #ff4058;
    }

    .section-description {
        color: #a99fa8;

        font-family:
            "Courier New",
            Courier,
            monospace;

        font-size: 0.82rem;

        margin-bottom: 1rem;
    }

    /* -------------------------------------------------- */
    /* Hero */
    /* -------------------------------------------------- */

    .hero-subtitle {
        text-align: center;

        color: #d8b8ca;

        font-family:
            "Courier New",
            Courier,
            monospace;

        font-size: 0.82rem;

        letter-spacing: 1.2px;

        margin-top: 0.1rem;
    }

    .status-line {
        text-align: center;

        color: #ff4058;

        font-family:
            "Courier New",
            Courier,
            monospace;

        font-size: 0.65rem;

        letter-spacing: 1.5px;

        margin-top: 0.65rem;
    }

    /* -------------------------------------------------- */
    /* Info cards */
    /* -------------------------------------------------- */

    .info-card {
        background:
            linear-gradient(
                135deg,
                #0d0e14,
                #0a0b10
            );

        border: 1px solid #421d37;

        border-radius: 3px;

        padding: 0.95rem 1.1rem;

        min-height: 82px;

        overflow: hidden;

        box-shadow:
            inset 0 0 18px rgba(255, 43, 176, 0.02);
    }

    .card-value {
        color: #ff4fb3;

        font-family:
            "Courier New",
            Courier,
            monospace;

        font-size: 1.25rem;

        font-weight: 700;

        letter-spacing: 1px;

        overflow-wrap: anywhere;
    }

    .card-label {
        color: #817781;

        font-family:
            "Courier New",
            Courier,
            monospace;

        font-size: 0.66rem;

        text-transform: uppercase;

        letter-spacing: 1px;

        margin-top: 0.25rem;
    }

    /* -------------------------------------------------- */
    /* Terminal status */
    /* -------------------------------------------------- */

    .terminal-status {
        margin: 0.8rem 0 1rem 0;

        padding: 0.7rem 0.9rem;

        border-left: 2px solid #ff2bb0;

        background: #0c0d12;

        color: #887d87;

        font-family:
            "Courier New",
            Courier,
            monospace;

        font-size: 0.72rem;

        letter-spacing: 0.2px;
    }

    .terminal-status .accent {
        color: #ff4058;
    }

    /* -------------------------------------------------- */
    /* Prediction */
    /* -------------------------------------------------- */

    .prediction-card {
        background:
            radial-gradient(
                circle at center,
                rgba(255, 43, 176, 0.055),
                transparent 68%
            ),
            #0b0c11;

        border: 1px solid #ff2bb0;

        border-radius: 3px;

        padding: 1.5rem;

        text-align: center;

        margin-top: 0.7rem;

        box-shadow:
            0 0 18px rgba(255, 43, 176, 0.05),
            inset 0 0 18px rgba(255, 43, 176, 0.025);
    }

    .prediction-label {
        color: #9a8493;

        font-family:
            "Courier New",
            Courier,
            monospace;

        font-size: 0.68rem;

        text-transform: uppercase;

        letter-spacing: 2px;
    }

    .prediction-value {
        color: #ff4fb3;

        font-family:
            "Courier New",
            Courier,
            monospace;

        font-size: 2.5rem;

        font-weight: 700;

        letter-spacing: 4px;

        margin-top: 0.35rem;

        text-shadow:
            0 0 10px rgba(255, 43, 176, 0.2);
    }

    .prediction-confidence {
        color: #c1b5bf;

        font-family:
            "Courier New",
            Courier,
            monospace;

        font-size: 0.82rem;

        margin-top: 0.4rem;
    }

    /* -------------------------------------------------- */
    /* Workflow cards */
    /* -------------------------------------------------- */

    .workflow-card {
        background: #0c0d12;

        border: 1px solid #35182e;

        border-radius: 3px;

        padding: 1rem;

        min-height: 145px;

        height: 100%;

        overflow: hidden;

        overflow-wrap: break-word;

        word-break: normal;

        box-shadow:
            inset 0 0 15px rgba(255, 43, 176, 0.018);
    }

    .workflow-number {
        color: #ff4058;

        font-family:
            "Courier New",
            Courier,
            monospace;

        font-size: 0.68rem;

        font-weight: 700;

        letter-spacing: 1px;

        white-space: normal;
    }

    .workflow-title {
        color: #ff6bbb;

        font-family:
            "Courier New",
            Courier,
            monospace;

        font-size: 0.9rem;

        font-weight: 700;

        margin-top: 0.35rem;
    }

    .workflow-text {
        color: #918791;

        font-family:
            "Courier New",
            Courier,
            monospace;

        font-size: 0.71rem;

        line-height: 1.55;

        margin-top: 0.4rem;

        white-space: normal;

        overflow-wrap: break-word;
    }

    /* -------------------------------------------------- */
    /* Sidebar */
    /* -------------------------------------------------- */

    .sidebar-heading {
        color: #ff4fb3;

        font-family:
            "Courier New",
            Courier,
            monospace;

        font-size: 0.88rem;

        font-weight: 700;

        letter-spacing: 0.5px;
    }

    .sidebar-heading::before {
        content: "/ ";
        color: #ff4058;
    }

    .sidebar-text {
        color: #9a9099;

        font-family:
            "Courier New",
            Courier,
            monospace;

        font-size: 0.71rem;

        line-height: 1.6;
    }

    /* -------------------------------------------------- */
    /* File uploader */
    /* -------------------------------------------------- */

    section[data-testid="stFileUploaderDropzone"] {
        background: #0b0c11;

        border: 1px dashed #7a2855;

        border-radius: 3px;
    }

    section[data-testid="stFileUploaderDropzone"]:hover {
        border-color: #ff2bb0;

        background: #0e0c12;
    }

    /* Keep Streamlit's icon font intact inside the uploader.
       Applying Courier New to every child turns the upload icon
       ligature into visible text such as "upload". */

    section[data-testid="stFileUploaderDropzone"] {
        background: #0b0c11;
        border: 1px dashed #7a2855;
        border-radius: 3px;
        }

    section[data-testid="stFileUploaderDropzone"]:hover {
        border-color: #ff2bb0;
        background: #0e0c12;
        }

    /* -------------------------------------------------- */
    /* Buttons */
    /* -------------------------------------------------- */

    div.stButton > button[kind="primary"] {
        background: #ff2bb0;

        border: 1px solid #ff70bf;

        border-radius: 3px;

        color: #08090d;

        font-family:
            "Courier New",
            Courier,
            monospace;

        font-weight: 700;

        letter-spacing: 0.5px;

        box-shadow:
            0 0 12px rgba(255, 43, 176, 0.1);
    }

    div.stButton > button[kind="primary"]:hover {
        background: #ff4fb3;

        border-color: #ff8aca;

        color: #08090d;

        box-shadow:
            0 0 18px rgba(255, 43, 176, 0.18);
    }

    /* -------------------------------------------------- */
    /* Progress */
    /* -------------------------------------------------- */



    /* -------------------------------------------------- */
    /* Metrics */
    /* -------------------------------------------------- */

    [data-testid="stMetricLabel"] {
        color: #817781 !important;

        font-family:
            "Courier New",
            Courier,
            monospace;
    }

    [data-testid="stMetricValue"] {
        color: #ff4fb3 !important;

        font-family:
            "Courier New",
            Courier,
            monospace;
    }

    /* -------------------------------------------------- */
    /* Alerts */
    /* -------------------------------------------------- */

    div[data-testid="stAlert"] {
        border-radius: 3px;

        font-family:
            "Courier New",
            Courier,
            monospace;
    }

    /* -------------------------------------------------- */
    /* Divider */
    /* -------------------------------------------------- */

    hr {
        border-color: #281723;
    }

    /* -------------------------------------------------- */
    /* Footer */
    /* -------------------------------------------------- */

    .footer {
        text-align: center;

        color: #665b65;

        font-family:
            "Courier New",
            Courier,
            monospace;

        font-size: 0.64rem;

        letter-spacing: 0.4px;

        padding-top: 1.8rem;
    }

    .footer-accent {
        color: #ff4058;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# ANIMATED LOGO
# ------------------------------------------------------------

def display_gif(path):

    try:
        with open(path, "rb") as f:
            gif_data = base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        st.warning(
            "Logo GIF not found. Expected: assets/seismosense.gif"
        )
        return

    html = f"""
    <div style="
        display:flex;
        flex-direction:column;
        align-items:center;
        text-align:center;
        padding:0;
        margin:0;
    ">

        <img
            src="data:image/gif;base64,{gif_data}"
            width="150"
            height="105"
            style="
                object-fit:contain;
                display:block;
                margin:0;
                background:transparent;
            "
        >

        <div style="
            margin-top:-3px;
            font-family:'Courier New', monospace;
            font-size:42px;
            font-weight:700;
            letter-spacing:-1px;
            color:#ff4fb3;
            text-shadow:0 0 12px rgba(255,43,176,0.18);
        ">
            SeismoSense
        </div>

        <div style="
            margin-top:2px;
            font-family:'Courier New', monospace;
            font-size:13px;
            letter-spacing:1px;
            color:#d8b8ca;
        ">
            AI-POWERED SEISMIC SIGNAL CLASSIFICATION
        </div>

        <div style="
            margin-top:4px;
            font-family:'Courier New', monospace;
            font-size:8px;
            letter-spacing:2px;
            color:#ff4058;
        ">
            [ SYSTEM READY ]
        </div>

    </div>
    """

    components.html(
        textwrap.dedent(html).strip(),
        height=195,
        scrolling=False
    )


# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH
    )


# ------------------------------------------------------------
# LOAD NORMALIZATION PARAMETERS
# ------------------------------------------------------------

@st.cache_resource
def load_normalization():

    normalization = np.load(
        NORMALIZATION_PATH
    )

    mean = normalization["mean"].reshape(2)
    std = normalization["std"].reshape(2)

    return mean, std


# ------------------------------------------------------------
# HERO
# ------------------------------------------------------------

display_gif(GIF_PATH)

render_html(
    """
    <div class="hero-subtitle">
        TURNING WAVEFORMS INTO INSIGHTS.
    </div>

    <div class="status-line">
        &gt; SEISMOSENSE.EXE &nbsp; | &nbsp; CNN-LSTM ENGINE &nbsp; | &nbsp; READY
    </div>
    """
)

st.divider()


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

with st.sidebar:

    render_html(
        """
        <div class="sidebar-heading">ABOUT</div>

        <div class="sidebar-text">
        SeismoSense is a deep learning application for
        classifying two-channel seismic waveforms as
        <b>Signal</b> or <b>Noise</b>.
        </div>
        """
    )

    st.divider()

    render_html(
        """
        <div class="sidebar-heading">MODEL PIPELINE</div>

        <div class="sidebar-text">

        <span style="color:#ff4058;">01</span>
        &nbsp; Waveform Input

        <br>

        <span style="color:#ff4058;">02</span>
        &nbsp; Downsampling<br>
        &nbsp;&nbsp;&nbsp;&nbsp;500 → 125 samples

        <br>

        <span style="color:#ff4058;">03</span>
        &nbsp; Normalization

        <br>

        <span style="color:#ff4058;">04</span>
        &nbsp; CNN Feature Extraction

        <br>

        <span style="color:#ff4058;">05</span>
        &nbsp; LSTM Temporal Modeling

        <br>

        <span style="color:#ff4058;">06</span>
        &nbsp; Signal / Noise Classification

        </div>
        """
    )

    st.divider()

    render_html(
        """
        <div class="sidebar-heading">PERFORMANCE</div>
        """
    )

    st.metric(
        "Test Accuracy",
        "86.82%"
    )

    st.metric(
        "ROC-AUC",
        "0.9461"
    )

    st.caption(
        "Measured on the held-out test set."
    )

    st.divider()

    render_html(
        """
        <div class="sidebar-heading">INFO</div>

        <div class="sidebar-text">

        CNN-LSTM seismic waveform classifier.

        <br><br>

        Built with:
        <br><br>
        TensorFlow
        <br><br>
        Streamlit

        <br><br>

        <span style="color:#ff4058;">
        &gt; SYSTEM READY.
        </span>

        </div>
        """
    )


# ------------------------------------------------------------
# UPLOAD SECTION
# ------------------------------------------------------------

render_html(
    """
    <div class="section-title">
        UPLOAD A SEISMIC WAVEFORM
    </div>

    <div class="section-description">
        Upload a CSV containing 500 samples across two seismic channels.
    </div>
    """
)

uploaded_file = st.file_uploader(
    "Choose a waveform CSV",
    type=["csv"],
    help="The CSV must contain exactly 500 rows and 2 numerical columns."
)


# ------------------------------------------------------------
# NO FILE STATE
# ------------------------------------------------------------

if uploaded_file is None:

    render_html(
        """
        <div class="terminal-status">
            <span class="accent">&gt;</span>
            WAITING FOR WAVEFORM INPUT...
        </div>
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        render_html(
            """
            <div class="info-card">
                <div class="card-value">500</div>
                <div class="card-label">Samples</div>
            </div>
            """
        )

    with col2:
        render_html(
            """
            <div class="info-card">
                <div class="card-value">2</div>
                <div class="card-label">Channels</div>
            </div>
            """
        )

    with col3:
        render_html(
            """
            <div class="info-card">
                <div class="card-value">CNN-LSTM</div>
                <div class="card-label">Architecture</div>
            </div>
            """
        )

    render_html(
        """
        <div class="terminal-status">
            EXPECTED FORMAT
            &nbsp; :: &nbsp;
            500 ROWS × 2 NUMERICAL COLUMNS
            &nbsp; :: &nbsp;
            CSV
        </div>
        """
    )

    st.stop()


# ------------------------------------------------------------
# READ CSV
# ------------------------------------------------------------

try:

    df = pd.read_csv(uploaded_file)

except Exception as e:

    st.error(
        f"Unable to read CSV: {e}"
    )

    st.stop()


# ------------------------------------------------------------
# VALIDATE SHAPE
# ------------------------------------------------------------

if df.shape != (500, 2):

    st.error(
        f"Invalid waveform shape: {df.shape}"
    )

    render_html(
        """
        <div class="terminal-status">
            <span class="accent">&gt;</span>
            EXPECTED 500 ROWS × 2 NUMERICAL COLUMNS.
        </div>
        """
    )

    st.stop()


# ------------------------------------------------------------
# CHECK MISSING VALUES
# ------------------------------------------------------------

if df.isnull().values.any():

    st.error(
        "Missing values detected in waveform."
    )

    render_html(
        """
        <div class="terminal-status">
            <span class="accent">&gt;</span>
            REMOVE MISSING VALUES AND TRY AGAIN.
        </div>
        """
    )

    st.stop()


# ------------------------------------------------------------
# CONVERT TO NUMPY
# ------------------------------------------------------------

try:

    waveform = df.values.astype(
        np.float32
    )

except ValueError:

    st.error(
        "Waveform values must be numerical."
    )

    st.stop()


# ------------------------------------------------------------
# WAVEFORM INFORMATION
# ------------------------------------------------------------

render_html(
    """
    <div class="section-title">
        WAVEFORM INFORMATION
    </div>
    """
)

col1, col2, col3 = st.columns(3)

with col1:
    render_html(
        """
        <div class="info-card">
            <div class="card-value">500</div>
            <div class="card-label">Samples</div>
        </div>
        """
    )

with col2:
    render_html(
        """
        <div class="info-card">
            <div class="card-value">2</div>
            <div class="card-label">Channels</div>
        </div>
        """
    )

with col3:
    render_html(
        """
        <div class="info-card">
            <div class="card-value">CSV</div>
            <div class="card-label">Input Format</div>
        </div>
        """
    )


# ------------------------------------------------------------
# WAVEFORM VISUALIZATION
# ------------------------------------------------------------

render_html(
    """
    <div class="section-title">
        WAVEFORM VISUALIZATION
    </div>

    <div class="section-description">
        Visualization of the two input seismic channels.
    </div>
    """
)

fig, ax = plt.subplots(
    figsize=(12, 4)
)

fig.patch.set_facecolor("#0b0c11")
ax.set_facecolor("#0b0c11")

ax.plot(
    waveform[:, 0],
    label="Channel 1",
    linewidth=1.2,
    color="#ff4058"
)

ax.plot(
    waveform[:, 1],
    label="Channel 2",
    linewidth=1.2,
    color="#ff2bb0"
)

ax.set_xlabel(
    "Time Sample",
    color="#9d929b"
)

ax.set_ylabel(
    "Amplitude",
    color="#9d929b"
)

ax.set_title(
    "INPUT SEISMIC WAVEFORM",
    color="#ff6bbb",
    fontsize=11,
    fontfamily="monospace"
)

ax.tick_params(
    colors="#807680"
)

for spine in ax.spines.values():
    spine.set_color("#3a2634")

ax.grid(
    alpha=0.12,
    color="#ff2bb0"
)

ax.legend(
    frameon=False,
    labelcolor="#c8b8c5"
)

fig.tight_layout()

st.pyplot(
    fig,
    use_container_width=True
)

plt.close(fig)


# ------------------------------------------------------------
# ANALYSIS SECTION
# ------------------------------------------------------------

st.divider()

render_html(
    """
    <div class="section-title">
        ANALYZE WAVEFORM
    </div>

    <div class="section-description">
        Run the trained CNN-LSTM engine on the uploaded waveform.
    </div>
    """
)

if st.button(
    "🔎  DETECT SIGNAL",
    type="primary",
    use_container_width=True
):

    # --------------------------------------------------------
    # MODEL INFERENCE
    # --------------------------------------------------------

    with st.spinner(
        "PROCESSING WAVEFORM..."
    ):

        try:

            model = load_model()

            mean, std = load_normalization()

            # Downsample
            processed_waveform = waveform[::4, :]

            # Normalize
            processed_waveform = (
                processed_waveform - mean
            ) / std

            # Add batch dimension
            processed_waveform = np.expand_dims(
                processed_waveform,
                axis=0
            )

            # Model prediction
            probability = model.predict(
                processed_waveform,
                verbose=0
            )[0][0]

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )

            st.stop()


    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    if probability >= 0.5:

        prediction = "Signal"
        confidence = probability

    else:

        prediction = "Noise"
        confidence = 1 - probability


    # --------------------------------------------------------
    # PREDICTION RESULT
    # --------------------------------------------------------

    st.divider()

    render_html(
        """
        <div class="section-title">
            PREDICTION RESULT
        </div>
        """
    )

    render_html(
        f"""
        <div class="prediction-card">

            <div class="prediction-label">
                Classification
            </div>

            <div class="prediction-value">
                {prediction.upper()}
            </div>

            <div class="prediction-confidence">
                Model confidence:
                <b>{confidence * 100:.2f}%</b>
            </div>
            
            &gt; Model output based on the trained CNN-LSTM classifier.
            This system performs seismic signal/noise classification
            and is not an earthquake early-warning system.

        </div>
        """
    )


    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    render_html(
        """
        <div style="
            margin-top:1rem;
            margin-bottom:0.4rem;
            color:#bcaeba;
            font-family:'Courier New', monospace;
            font-size:0.72rem;
            font-weight:700;
            letter-spacing:1px;
        ">
            CONFIDENCE
        </div>
        """
    )

    render_html(
    f"""
    <div style="
        margin-top:0.45rem;
        height:8px;
        width:100%;
        background:#191019;
        border:1px solid #35152d;
        border-radius:2px;
        overflow:hidden;
    ">
        <div style="
            width:{confidence * 100:.2f}%;
            height:100%;
            background:linear-gradient(
                90deg,
                #ff4058,
                #ff2bb0,
                #ff72ba
            );
            box-shadow:0 0 8px rgba(255,43,176,0.25);
        "></div>
    </div>
    """
    )


    # --------------------------------------------------------
    # TERMINAL OUTPUT
    # --------------------------------------------------------

    render_html(
        f"""
        <div class="terminal-status">
            <span class="accent">&gt;</span>
            MODEL OUTPUT
            &nbsp; :: &nbsp;
            {prediction.upper()}
            &nbsp; :: &nbsp;
            {confidence * 100:.2f}% CONFIDENCE
        </div>
        """
    )


# ------------------------------------------------------------
# HOW IT WORKS
# ------------------------------------------------------------

st.divider()

render_html(
    """
    <div class="section-title">
        HOW SEISMOSENSE WORKS
    </div>

    <div class="section-description">
        From raw waveform to model classification.
    </div>
    """
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    render_html(
        """
        <div class="workflow-card">

            <div class="workflow-number">
                01 // UPLOAD
            </div>

            <div class="workflow-title">
                Input
            </div>

            <div class="workflow-text">
                Upload a two-channel seismic waveform
                containing 500 samples.
            </div>

        </div>
        """
    )

with col2:
    render_html(
        """
        <div class="workflow-card">

            <div class="workflow-number">
                02 // PROCESS
            </div>

            <div class="workflow-title">
                Preprocessing
            </div>

            <div class="workflow-text">
                The waveform is downsampled from
                500 to 125 samples and normalized.
            </div>

        </div>
        """
    )

with col3:
    render_html(
        """
        <div class="workflow-card">

            <div class="workflow-number">
                03 // ANALYZE
            </div>

            <div class="workflow-title">
                CNN + LSTM
            </div>

            <div class="workflow-text">
                CNN layers extract waveform features
                while the LSTM models temporal patterns.
            </div>

        </div>
        """
    )

with col4:
    render_html(
        """
        <div class="workflow-card">

            <div class="workflow-number">
                04 // CLASSIFY
            </div>

            <div class="workflow-title">
                Output
            </div>

            <div class="workflow-text">
                The trained model classifies the
                waveform as Signal or Noise.
            </div>

        </div>
        """
    )


# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

render_html(
    """
    <div class="footer">
        <span class="footer-accent">SEISMOSENSE</span>
        &nbsp; v1.0.0
        &nbsp; | &nbsp;
        CNN-LSTM SEISMIC SIGNAL CLASSIFICATION
        &nbsp; | &nbsp;
        <span class="footer-accent">SYSTEM READY.</span>
    </div>
    """
)

