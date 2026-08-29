import streamlit as st

ALLOWED_USERS = {
    "laliparth@gmail.com",
    "zaiddyd42@gmail.com",
    "shreyagoyal1733@gmail.com",
    "dharneet08@gmail.com",
    "shounak6425@gmail.com"
    "guptatanush763@gmail.com"
}


def login_screen():
    st.title("Intelligent Border Video Analytics Platform")
    st.subheader("Secure Surveillance Portal")

    st.button("Login with Google", on_click=st.login)


if not st.user.is_logged_in:
    login_screen()
    st.stop()


if st.user.email not in ALLOWED_USERS:
    st.error("You are not authorized to access this system.")
    st.button("Logout", on_click=st.logout)
    st.stop()


st.title("Intelligent Border Video Analytics Platform")

st.success(f"Welcome, {st.user.name}!")

st.write("We will add our app here...")

if st.button("Logout"):
    st.logout()
