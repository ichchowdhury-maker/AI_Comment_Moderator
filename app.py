import streamlit as st
import joblib
import scipy.sparse
import requests
import secrets
from urllib.parse import urlencode


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Comment Moderator",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# PRIVACY POLICY PAGE
# IMPORTANT: KEEP THIS BEFORE FACEBOOK CALLBACK
# =========================================================

if st.query_params.get("page") == "privacy":

    st.title("🔒 Privacy Policy")

    st.write("Last updated: September 2026")

    st.markdown("""
## 1. Information We Access

AI Comment Moderator may access information necessary to
connect and manage Facebook Pages that the user is authorized
to manage.

This may include:

- Facebook Page name
- Facebook Page ID
- Authorization information
- Comments required for moderation

## 2. How We Use Information

Information is used only to provide the application's
functionality, including:

- Connecting Facebook Pages
- Analyzing comments
- Detecting Toxic, Spam, Promo and Normal comments
- Performing authorized moderation actions

## 3. Facebook Data

The application uses Meta/Facebook APIs for Facebook Page
integration.

We do not sell Facebook user data or use Facebook data for
advertising purposes.

## 4. Access Tokens

Facebook access tokens are sensitive credentials and are used
only for authorized Facebook API operations.

Users should never share their Facebook passwords or access
tokens with other people.

## 5. Data Storage

The application is designed to minimize the storage of
personal information.

Information may be processed when necessary to provide
moderation functionality.

## 6. Data Sharing

We do not sell or rent personal information to third parties.

## 7. Data Security

Reasonable technical measures are used to protect application
credentials and information.

However, no internet service can guarantee absolute security.

## 8. User Control

Users can stop using the application and remove its Facebook
permissions through their Facebook account settings.

## 9. Changes to This Privacy Policy

This Privacy Policy may be updated when the application or
its features change.

## 10. Contact

For questions about this Privacy Policy or AI Comment Moderator,
please contact the application developer.

## 11. Third-Party Services

The application uses Meta/Facebook services and APIs.

Their use is also subject to Meta's applicable terms and policies.
""")

    st.divider()

    st.caption(
        "AI Comment Moderator | Privacy Policy"
    )

    st.stop()


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 17px;
        opacity: 0.75;
        margin-bottom: 25px;
    }

    .status-box {
        padding: 18px;
        border-radius: 12px;
        margin-top: 10px;
    }

    .facebook-box {
        padding: 20px;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# META CONFIGURATION
# =========================================================

GRAPH_API_VERSION = "v26.0"


def get_secret(name, default=""):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


META_APP_ID = get_secret("META_APP_ID")
META_APP_SECRET = get_secret("META_APP_SECRET")
META_CONFIG_ID = get_secret("META_CONFIG_ID")
META_REDIRECT_URI = get_secret("META_REDIRECT_URI")


# =========================================================
# SESSION STATE
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "facebook_user_token" not in st.session_state:
    st.session_state.facebook_user_token = None

if "facebook_pages" not in st.session_state:
    st.session_state.facebook_pages = []

if "selected_page" not in st.session_state:
    st.session_state.selected_page = None

if "oauth_state" not in st.session_state:
    st.session_state.oauth_state = None

if "facebook_connected" not in st.session_state:
    st.session_state.facebook_connected = False


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "models/comment_moderator_model.joblib"
    )

    word_vectorizer = joblib.load(
        "models/word_tfidf_vectorizer.joblib"
    )

    char_vectorizer = joblib.load(
        "models/char_tfidf_vectorizer.joblib"
    )

    return model, word_vectorizer, char_vectorizer


try:

    model, word_vectorizer, char_vectorizer = load_model()

except Exception:

    st.error("❌ Model could not be loaded.")

    st.info(
        "Make sure these files exist inside the models folder:"
    )

    st.code(
        """
models/
├── comment_moderator_model.joblib
├── word_tfidf_vectorizer.joblib
└── char_tfidf_vectorizer.joblib
        """
    )

    st.stop()


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_comment(text):

    text = str(text).strip()

    word_features = word_vectorizer.transform([text])

    char_features = char_vectorizer.transform([text])

    features = scipy.sparse.hstack(
        [word_features, char_features]
    )

    probabilities = model.predict_proba(features)[0]

    classes = model.classes_

    prediction_index = probabilities.argmax()

    label = classes[prediction_index]

    confidence = probabilities[prediction_index] * 100

    probability_dict = {
        classes[i]: probabilities[i] * 100
        for i in range(len(classes))
    }

    return label, confidence, probability_dict


# =========================================================
# FACEBOOK LOGIN URL
# =========================================================

def create_facebook_login_url():

    if not META_APP_ID:
        return None

    if not META_CONFIG_ID:
        return None

    if not META_REDIRECT_URI:
        return None

    state = secrets.token_urlsafe(32)

    st.session_state.oauth_state = state

    params = {
        "client_id": META_APP_ID,
        "redirect_uri": META_REDIRECT_URI,
        "config_id": META_CONFIG_ID,
        "response_type": "code",
        "state": state
    }

    return (
        "https://www.facebook.com/"
        f"{GRAPH_API_VERSION}/dialog/oauth?"
        + urlencode(params)
    )


# =========================================================
# FACEBOOK TOKEN EXCHANGE
# =========================================================

def exchange_code_for_user_token(code):

    url = (
        f"https://graph.facebook.com/"
        f"{GRAPH_API_VERSION}/oauth/access_token"
    )

    params = {
        "client_id": META_APP_ID,
        "client_secret": META_APP_SECRET,
        "redirect_uri": META_REDIRECT_URI,
        "code": code
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    try:
        data = response.json()
    except Exception:
        data = {
            "error": "Invalid response from Facebook."
        }

    if response.status_code != 200:
        return None, data

    return data.get("access_token"), data


# =========================================================
# GET USER'S PAGES
# =========================================================

def get_managed_pages(user_access_token):

    url = (
        f"https://graph.facebook.com/"
        f"{GRAPH_API_VERSION}/me/accounts"
    )

    params = {
        "access_token": user_access_token,
        "fields": "id,name,access_token",
        "limit": 100
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    try:
        data = response.json()
    except Exception:
        data = {
            "error": "Invalid response from Facebook."
        }

    if response.status_code != 200:
        return [], data

    return data.get("data", []), data


# =========================================================
# HANDLE FACEBOOK CALLBACK
# =========================================================

query_params = st.query_params

facebook_code = query_params.get("code")
facebook_state = query_params.get("state")
facebook_error = query_params.get("error")


if facebook_error:

    st.error(
        "❌ Facebook Login was cancelled or failed."
    )

    st.session_state.oauth_state = None

    st.query_params.clear()


elif facebook_code:

    expected_state = st.session_state.get(
        "oauth_state"
    )

    if not expected_state or facebook_state != expected_state:

        st.error(
            "❌ Security check failed. "
            "Please try Facebook Login again."
        )

        st.query_params.clear()

    elif not META_APP_SECRET:

        st.error(
            "❌ META_APP_SECRET is not configured."
        )

    else:

        with st.spinner(
            "Connecting your Facebook account..."
        ):

            try:

                user_token, token_response = (
                    exchange_code_for_user_token(
                        facebook_code
                    )
                )

                if not user_token:

                    st.error(
                        "❌ Could not obtain Facebook access token."
                    )

                    st.json(token_response)

                else:

                    st.session_state.facebook_user_token = (
                        user_token
                    )

                    pages, page_response = (
                        get_managed_pages(
                            user_token
                        )
                    )

                    if pages:

                        st.session_state.facebook_pages = pages

                        st.session_state.facebook_connected = True

                        st.success(
                            "✅ Facebook connected successfully!"
                        )

                    else:

                        st.warning(
                            "Facebook login worked, "
                            "but no managed Pages were returned."
                        )

                        st.info(
                            "Make sure the Facebook account "
                            "has the required Page permissions."
                        )

                        if page_response:
                            st.json(page_response)

            except Exception as e:

                st.error(
                    "❌ Facebook connection failed."
                )

                st.caption(str(e))

        st.session_state.oauth_state = None

        st.query_params.clear()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">'
    '🛡️ AI Comment Moderator'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered comment moderation system for detecting '
    'Toxic, Spam, Promo and Normal comments.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# FACEBOOK CONNECT SECTION
# =========================================================

st.subheader("🔵 Facebook Page Connection")

if st.session_state.facebook_connected:

    st.success(
        "✅ Facebook account connected"
    )

    pages = st.session_state.facebook_pages

    if pages:

        page_names = [
            page.get("name", "Unnamed Page")
            for page in pages
        ]

        selected_page_name = st.selectbox(
            "Select a Page you manage",
            page_names
        )

        selected_index = page_names.index(
            selected_page_name
        )

        selected_page = pages[selected_index]

        st.session_state.selected_page = selected_page

        st.info(
            f"Connected Page: "
            f"**{selected_page.get('name', 'Unnamed Page')}**"
        )

        st.caption(
            "Page connection is ready for the next "
            "comment moderation/API integration step."
        )

    else:

        st.warning(
            "No Pages found for this Facebook account."
        )

else:

    st.write(
        "Connect your Facebook account to select a "
        "Page that you manage."
    )

    login_url = create_facebook_login_url()

    if login_url:

        st.link_button(
            "🔵 Connect Facebook",
            login_url,
            use_container_width=True
        )

    else:

        st.warning(
            "Facebook Login is not configured yet."
        )

        st.caption(
            "Configure META_APP_ID, META_CONFIG_ID "
            "and META_REDIRECT_URI in Streamlit Secrets."
        )


st.divider()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Moderation Settings")

    toxic_threshold = st.slider(
        "🔴 Toxic Threshold",
        min_value=50,
        max_value=99,
        value=70,
        step=1
    )

    spam_threshold = st.slider(
        "🟠 Spam Threshold",
        min_value=50,
        max_value=99,
        value=70,
        step=1
    )

    promo_threshold = st.slider(
        "🔵 Promo Threshold",
        min_value=50,
        max_value=99,
        value=70,
        step=1
    )

    st.divider()

    st.header("📊 Categories")

    st.write("🔴 **TOXIC**")
    st.caption(
        "Abusive, insulting or threatening comments."
    )

    st.write("🟠 **SPAM**")
    st.caption(
        "Spam, scam or suspicious messages."
    )

    st.write("🔵 **PROMO**")
    st.caption(
        "Promotional or advertising comments."
    )

    st.write("🟢 **NORMAL**")
    st.caption(
        "Normal comments and conversations."
    )

    st.divider()

    st.header("🤖 Model")

    st.caption(
        "TF-IDF + Logistic Regression"
    )

    st.caption(
        "4-Class Comment Classification"
    )


# =========================================================
# TOP STATISTICS
# =========================================================

total_comments = len(
    st.session_state.history
)

toxic_count = sum(
    1
    for x in st.session_state.history
    if x["label"] == "TOXIC"
)

spam_count = sum(
    1
    for x in st.session_state.history
    if x["label"] == "SPAM"
)

promo_count = sum(
    1
    for x in st.session_state.history
    if x["label"] == "PROMO"
)

normal_count = sum(
    1
    for x in st.session_state.history
    if x["label"] == "NORMAL"
)


stat1, stat2, stat3, stat4, stat5 = st.columns(5)


with stat1:
    st.metric(
        "Total",
        total_comments
    )

with stat2:
    st.metric(
        "🔴 Toxic",
        toxic_count
    )

with stat3:
    st.metric(
        "🟠 Spam",
        spam_count
    )

with stat4:
    st.metric(
        "🔵 Promo",
        promo_count
    )

with stat5:
    st.metric(
        "🟢 Normal",
        normal_count
    )


st.divider()


# =========================================================
# COMMENT INPUT
# =========================================================

st.subheader("💬 Analyze a Comment")

comment = st.text_area(
    "Enter comment",
    placeholder="Write a Facebook comment here...",
    height=150,
    label_visibility="collapsed"
)


button_col1, button_col2 = st.columns(2)


with button_col1:

    analyze = st.button(
        "🔍 Analyze Comment",
        use_container_width=True
    )


with button_col2:

    clear = st.button(
        "🗑️ Clear History",
        use_container_width=True
    )


# =========================================================
# CLEAR HISTORY
# =========================================================

if clear:

    st.session_state.history = []

    st.rerun()


# =========================================================
# ANALYSIS
# =========================================================

if analyze:

    if not comment.strip():

        st.warning(
            "⚠️ Please enter a comment before analyzing."
        )

    else:

        try:

            label, confidence, probabilities = (
                predict_comment(comment)
            )

            # ---------------------------------------------
            # MODERATION DECISION
            # ---------------------------------------------

            if label == "TOXIC":

                if confidence >= toxic_threshold:

                    decision = "HIDE / REVIEW"
                    decision_icon = "🔴"

                else:

                    decision = "MANUAL REVIEW"
                    decision_icon = "🟡"

            elif label == "SPAM":

                if confidence >= spam_threshold:

                    decision = "HIDE / REVIEW"
                    decision_icon = "🟠"

                else:

                    decision = "MANUAL REVIEW"
                    decision_icon = "🟡"

            elif label == "PROMO":

                if confidence >= promo_threshold:

                    decision = "ALLOW / REVIEW"
                    decision_icon = "🔵"

                else:

                    decision = "MANUAL REVIEW"
                    decision_icon = "🟡"

            else:

                decision = "ALLOW"
                decision_icon = "🟢"


            # ---------------------------------------------
            # SAVE HISTORY
            # ---------------------------------------------

            st.session_state.history.insert(
                0,
                {
                    "comment": comment.strip(),
                    "label": label,
                    "confidence": confidence,
                    "decision": decision
                }
            )


            # ---------------------------------------------
            # RESULT
            # ---------------------------------------------

            st.divider()

            st.subheader(
                "🤖 AI Analysis"
            )

            result1, result2, result3 = st.columns(3)


            with result1:

                if label == "TOXIC":

                    st.error(
                        f"🔴 {label}"
                    )

                elif label == "SPAM":

                    st.warning(
                        f"🟠 {label}"
                    )

                elif label == "PROMO":

                    st.info(
                        f"🔵 {label}"
                    )

                else:

                    st.success(
                        f"🟢 {label}"
                    )


            with result2:

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )


            with result3:

                st.metric(
                    "Moderation",
                    f"{decision_icon} {decision}"
                )


            # ---------------------------------------------
            # CONFIDENCE
            # ---------------------------------------------

            st.write("### Confidence")

            st.progress(
                min(
                    max(
                        confidence / 100,
                        0.0
                    ),
                    1.0
                )
            )


            # ---------------------------------------------
            # COMMENT
            # ---------------------------------------------

            st.write(
                "### 💬 Comment"
            )

            st.info(
                comment.strip()
            )


            # ---------------------------------------------
            # CATEGORY PROBABILITIES
            # ---------------------------------------------

            st.write(
                "### 📊 Category Probabilities"
            )

            probability_order = [
                "NORMAL",
                "TOXIC",
                "SPAM",
                "PROMO"
            ]

            for category in probability_order:

                if category in probabilities:

                    value = probabilities[category]

                    st.write(
                        f"**{category}: "
                        f"{value:.2f}%**"
                    )

                    st.progress(
                        min(
                            max(
                                value / 100,
                                0.0
                            ),
                            1.0
                        )
                    )


        except Exception as e:

            st.error(
                "❌ Something went wrong while "
                "analyzing the comment."
            )

            st.caption(str(e))


# =========================================================
# FACEBOOK PAGE STATUS
# =========================================================

if st.session_state.selected_page:

    st.divider()

    st.subheader(
        "📘 Connected Facebook Page"
    )

    page = st.session_state.selected_page

    page_col1, page_col2 = st.columns(2)

    with page_col1:

        st.write(
            f"**Page:** "
            f"{page.get('name', 'Unknown')}"
        )

    with page_col2:

        st.write(
            f"**Page ID:** "
            f"{page.get('id', 'Unknown')}"
        )


# =========================================================
# HISTORY
# =========================================================

if st.session_state.history:

    st.divider()

    st.subheader(
        "📜 Recent Analysis"
    )

    for index, item in enumerate(
        st.session_state.history[:10],
        start=1
    ):

        with st.expander(
            f"{index}. {item['label']} • "
            f"{item['confidence']:.2f}%"
        ):

            st.write(
                f"**Comment:** "
                f"{item['comment']}"
            )

            st.write(
                f"**Category:** "
                f"{item['label']}"
            )

            st.write(
                f"**Confidence:** "
                f"{item['confidence']:.2f}%"
            )

            st.write(
                f"**Decision:** "
                f"{item['decision']}"
            )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🛡️ AI Comment Moderator | "
    "Machine Learning Based Moderation System"
)
```
