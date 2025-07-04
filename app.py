import streamlit as st
import os
from PIL import Image
from model import predict_ecg_class

# ---------------------- Setup ----------------------
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------------- Session State ----------------------
if "page" not in st.session_state:
    st.session_state.page = "login"
if "username" not in st.session_state:
    st.session_state.username = ""
if "users" not in st.session_state:
    st.session_state.users = {}  # username: password

# ---------------------- Login Page ----------------------
def login_page():
    st.markdown(
        "<h2 style='text-align: center;'>🩺 Welcome to <span style='color:#E91E63'>ECG Diagnostic AI</span></h2>",
        unsafe_allow_html=True,
    )
    st.markdown("<h4 style='text-align: center;'>Please login to continue</h4>", unsafe_allow_html=True)
    st.markdown("###")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form", clear_on_submit=False):
            st.markdown("#### 🔐 User Login")
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter password")

            col_login, col_signup = st.columns(2)
            login_clicked = col_login.form_submit_button("🔓 Login")
            signup_clicked = col_signup.form_submit_button("📝 Sign Up")

            if login_clicked:
                if username in st.session_state.users and st.session_state.users[username] == password:
                    st.session_state.username = username
                    st.session_state.page = "welcome"
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password.")

            if signup_clicked:
                st.session_state.page = "signup"
                st.rerun()

# ---------------------- Signup Page ----------------------
def signup_page():
    st.markdown(
        "<h2 style='text-align: center;'>📝 Create a New Account</h2>",
        unsafe_allow_html=True,
    )
    st.markdown("<h4 style='text-align: center;'>Join ECG Diagnostic AI</h4>", unsafe_allow_html=True)
    st.markdown("###")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("signup_form", clear_on_submit=False):
            new_username = st.text_input("👤 Username", placeholder="Choose a username")
            new_password = st.text_input("🔒 Password", type="password", placeholder="Enter a secure password")
            confirm_password = st.text_input("🔒 Confirm Password", type="password", placeholder="Re-enter password")

            col_submit, col_back = st.columns(2)
            create = col_submit.form_submit_button("✅ Register")
            back = col_back.form_submit_button("🔙 Back to Login")

            if create:
                if new_username in st.session_state.users:
                    st.warning("⚠️ Username already exists. Try another.")
                elif new_password != confirm_password:
                    st.warning("⚠️ Passwords do not match.")
                elif not new_username or not new_password:
                    st.warning("⚠️ All fields are required.")
                else:
                    st.session_state.users[new_username] = new_password
                    st.success("🎉 Account created successfully!")
                    st.session_state.page = "login"
                    st.rerun()

            if back:
                st.session_state.page = "login"
                st.rerun()

# ---------------------- Welcome Page ----------------------
def welcome_page():
    st.markdown(
        f"<h2 style='text-align: center;'>👋 Welcome, <span style='color:#00bfa5'>{st.session_state.username}</span></h2>",
        unsafe_allow_html=True,
    )
    st.markdown("<h4 style='text-align: center;'>Upload your ECG image to get an AI-based diagnosis</h4>",
                unsafe_allow_html=True)
    st.markdown("###")

    uploaded_file = st.file_uploader("📤 Upload ECG Image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        st.image(Image.open(file_path), caption="📷 Uploaded ECG", use_container_width=True)
        st.markdown("")

        if st.button("🔍 Predict"):
            label, gradcam_overlay, ig_overlay, explanation = predict_ecg_class(file_path)
            # Show prediction
            st.subheader("🩺 Prediction")
            st.success(f"Predicted Class: **{label}**")

            # Show textual explanation
            st.subheader("🧠 Explanation")
            st.info(explanation)

            # Show Grad-CAM visualization
            st.subheader("🔍 Grad-CAM Highlight")
            st.image(gradcam_overlay, caption="Model Focus Area (Grad-CAM)", use_container_width =True)

            # Show Integrated Gradients overlay
            st.subheader("⚡ Integrated Gradients")
            st.image(ig_overlay, caption="Attribution Map (Integrated Gradients)", use_container_width =True)

    st.markdown("---")
    col1, col2, col3 = st.columns([4, 2, 4])
    with col2:
        if st.button("🚪 Logout"):
            st.session_state.page = "login"
            st.session_state.username = ""
            st.rerun()

# ---------------------- Routing ----------------------
if st.session_state.page == "login":
    login_page()
elif st.session_state.page == "signup":
    signup_page()
elif st.session_state.page == "welcome":
    welcome_page()
